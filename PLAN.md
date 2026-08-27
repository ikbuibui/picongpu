# PMacc Event System Replacement Plan

## Status

Implementation is in progress. The Caravan completion core (Phase 1) is
implemented and tested. The Phase 0 local inventory and CPU baseline are
recorded in `docs/CARAVAN_PHASE0.md`; target GPU measurements remain open.
Phase 2 has started with the dedicated `MpiRuntime`, nonblocking barrier and
all-reduce progress, point-to-point futures, receive metadata, buffer leases,
and MPI-owned Cartesian communicators with immutable topology snapshots. PMacc
can now attach its environment to the runtime and initialize `CommunicatorMPI`
from an immutable snapshot without application-thread topology calls. Legacy
`TaskSendMPI` and `TaskReceiveMPI` can poll Caravan point-to-point futures while
unchanged PMacc startup remains available during migration. Migrating the
example entry points and remaining MPI operations is next; target-GPU validation
is deferred as described in the migration gate below.

This plan targets the PMacc event system.
Breaking PMacc and PIConGPU interfaces is allowed during the migration.

## Goals

1. Make dependencies explicit and locally understandable.
2. Run every application MPI call on exactly one dedicated thread.
3. Keep MPI progress active while simulation threads create work or perform CPU work.
4. Allow task submission from multiple threads.
5. Prevent arbitrary shared task mutation; mutable native state has one owner.
6. Preserve native GPU queue concurrency and GPU-aware MPI support.
7. Replace hand-written task state machines with operation chains.
8. Propagate errors and retain buffer lifetimes correctly.
9. Remove the global manager, transaction stack, observer system, and task IDs.
10. Reduce scheduling overhead and total implementation size.

## Non-goals

- A general work-stealing task runtime.
- Automatic dependency discovery from arbitrary pointers.
- Arbitrary concurrent mutation of submitted tasks.
- Requiring `MPI_THREAD_SERIALIZED` or `MPI_THREAD_MULTIPLE`.
- Adding HPX, Taskflow, oneTBB, stdexec, or another runtime dependency.
- Preserving the current PMacc event-system API.
- Supporting direct MPI calls from simulation or plugin threads.

## Fixed design decisions

### One MPI thread, using `MPI_THREAD_FUNNELED`

The process main thread becomes the MPI executor thread. It calls
`MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`, starts the simulation on a
separate application thread, and then runs the MPI progress loop until
shutdown.

`MPI_THREAD_FUNNELED` is the minimum correct level for this process model:
the process is multithreaded, but only the thread that initialized MPI calls
MPI. `MPI_THREAD_SINGLE` is not sufficient for a multithreaded process, and
`MPI_THREAD_SERIALIZED` is unnecessary because MPI calls never move between
threads.

The MPI executor thread owns all MPI operations, including:

- initialization and finalization;
- communicator and datatype creation and destruction;
- topology queries;
- point-to-point communication;
- collectives and barriers;
- signal collectives;
- MPI-enabled third-party library entry points;
- error handling that requires MPI, including `MPI_Abort`.

No runtime mutex serializes MPI calls because there is only one caller.
Providing a higher thread level does not change this contract.

The executable structure becomes conceptually:

```cpp
int main(int argc, char** argv)
{
    return pmacc::MpiRuntime::run(
        argc,
        argv,
        [&]
        {
            return runApplication();
        });
}
```

`MpiRuntime::run()` performs this sequence:

```text
process main / MPI thread             application thread
-------------------------             ------------------
MPI_Init_thread(FUNNELED)
start application thread       -----> initialize simulation
run MPI executor loop          <----- submit MPI commands
progress active requests       <----> create GPU/MPI work
receive shutdown request       <----- application returns
finish or fail active work
free MPI-owned resources
join application thread
MPI_Finalize
return application result
```

If the MPI implementation provides less than `MPI_THREAD_FUNNELED`, startup
fails. It must not warn and continue.

### Single-owner mutation

Commands are immutable after submission. Each native execution domain owns
its mutable state:

- the MPI thread owns MPI requests, communicators, and statuses;
- a device executor owns alpaka queues, native device events, and pending
  device completions;
- producer threads own local `Flow` objects while constructing work.

Multiple threads may submit commands and share completion handles. They may
not directly mutate active commands or executor-native state. Cancellation or
priority changes, if later needed, are commands sent to the owner.

### Explicit dependencies

Every operation accepts dependencies and returns a completion handle.
Registering an operation must establish its predecessor relationship before
any externally visible side effect starts.

There is no implicit current transaction and no dependency behavior hidden in
buffer accessors.

### Caravan: PMacc-independent runtime

The replacement library is named **Caravan**. It uses the C++ namespace
`caravan` and is developed inside this repository as isolated CMake targets:

```text
caravan::core              Event, Future<T>, whenAll(), Flow, shutdown
    +-- caravan::mpi       MpiRuntime and MpiExecutor; depends on MPI
    `-- caravan::alpaka    DeviceExecutor; depends on alpaka
             ^
           PMacc           buffers, topology, fields, particles, exchanges
             ^
         PIConGPU
```

The runtime targets must not include PMacc or PIConGPU headers. Their public
interfaces use generic lifetime tokens, pointers, extents, and opaque native
resource descriptors rather than PMacc buffer or topology types. PMacc owns
the adapters for its buffers and topology and the composition of field,
particle, and exchange operations.

Do not promise a stable public ABI or create a separate package during the
migration. External extraction is deferred until the migration is complete or
a second real consumer needs it.

## Target architecture

```text
                         thread-safe completion states
                      +-------------------------------+
                      | Event, Future<T>, whenAll()   |
                      +-------------------------------+
                            ^                 ^
                            |                 |
producer threads            |                 |
+------------------+        |                 |
| simulation       |--MPSC--+--> DeviceExecutor --> alpaka queues/events
| plugins          |        |
| helper threads   |--MPSC------> MpiExecutor    --> MPI requests
+------------------+
```

There is no central scheduler that scans all tasks. Executors track only their
own queued and active native operations. Completion directly releases
successors to their target executor.

## Core types

### `Event` and `Future<T>`

`Event` is a direct handle to shared completion state. `Future<T>` adds an
immutable result.

```cpp
enum class CompletionState : std::uint8_t
{
    pending,
    ready,
    failed,
    cancelled
};

class Event;

template<typename T>
class Future;

Event readyEvent();
Event whenAll(std::span<Event const> events);
```

Required behavior:

- exactly one terminal transition;
- thread-safe observation and continuation registration;
- error propagation to dependent work;
- `whenAll()` completes only after every input is terminal, recording the first
  failure or cancellation and publishing it after the final input completes;
- a failed `whenAll()` is therefore still a quiescence boundary, not a
  fail-fast notification;
- a blocking wait for non-executor threads;
- no task IDs or global lookup;
- no user callback execution from a destructor;
- no callback invocation while holding an event-state lock;
- no recursive inline continuation chains;
- an already-ready event requires no allocation.

Initial implementation:

- `std::shared_ptr` for state ownership;
- `std::mutex` for subscriber registration;
- `std::condition_variable` for blocking waits;
- one atomic terminal state;
- one flat counter for `whenAll()`.

Prompt fatal-error notification, if needed, must use a separate runtime failure
signal rather than weakening `whenAll()` completion semantics. Initially retain
the first observed error; aggregate multiple errors only if diagnostics require
it. Do not begin with a lock-free state machine or custom allocator. Add a slab or
intrusive ownership only if profiles identify shared-state allocation as a
material cost.

### Completion phases for native device work

A device event may be usable by another device queue before it has completed
on the host. Device completion state therefore has two milestones:

```text
created -> submitted/native fence available -> completed
                                      `-------> failed
```

A device consumer may use the native fence at `submitted`. MPI and host
consumers require `completed` unless a future stream-aware transport explicitly
supports native device dependencies.

### `Flow`

`Flow` replaces the useful part of transactions: a local current frontier.

```cpp
class Flow
{
public:
    Flow fork() const;
    void join(Event event);
    Event done() const;

    template<typename T_Executor, typename T_Operation>
    Event then(T_Executor& executor, T_Operation&& operation);

private:
    Event m_tail;
};
```

Rules:

- `Flow` is a local value and is not shared concurrently;
- `fork()` copies the current frontier;
- `join()` uses `whenAll()`;
- `then()` explicitly submits an operation after the current frontier;
- low-level code may pass `Event` dependencies directly without using `Flow`;
- no global or thread-local transaction stack exists.

Example:

```cpp
Flow main{stepStart};

auto communication = main.fork();
communication.then(device, packBoundary);
communication.then(mpi, sendBoundary);

main.then(device, updateCore);
main.join(communication.done());
main.then(device, updateBorder);
```

## MPI executor

### Public API

Application code uses opaque communicator, peer, tag, and datatype
descriptors. Raw `MPI_Comm`, `MPI_Request`, `MPI_Status`, and `MPI_Datatype`
do not leave the MPI implementation layer.

Representative API:

```cpp
class MpiExecutor
{
public:
    Future<SendResult> send(
        Event dataReady,
        BufferLease buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator);

    Future<ReceiveResult> receive(
        Event bufferAvailable,
        BufferLease buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator);

    template<typename T>
    Future<T> allReduce(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        ReduceOperation operation,
        CommunicatorId communicator);

    Event barrier(Event predecessor, CommunicatorId communicator);
};
```

`ReceiveResult` owns copied status information such as source, tag, and byte or
element count. It is not a pointer borrowed during a callback.

### Submission queue

Use a mutex-protected `std::deque` plus `std::condition_variable` first.
Producers push commands and the MPI thread drains them in batches. This is
simpler than a custom lock-free MPSC queue and should already remove the
current map, set, allocation, and polling overhead. Replace it only with
benchmark evidence.

Commands whose dependencies are incomplete register a continuation that
posts the command when ready. They do not occupy the active MPI request list.
Receives should depend only on availability of their destination buffer so
they can be posted as early as correctness permits.

### Active request storage and progress

Store `MPI_Request` by value in contiguous executor-owned storage. Use
`MPI_Testsome` to progress active requests in batches.

```text
drain newly ready commands
start MPI_Isend/Irecv/Iallreduce/Ibarrier operations
MPI_Testsome(active requests)
process statuses and receive counts
complete futures
compact inactive entries
repeat
```

Completion only publishes event state and posts ready successor commands. It
does not run simulation code on the MPI thread.

The progress policy is configurable:

- default: spin while requests are active, assuming one CPU core is reserved;
- optional: spin, then yield, for oversubscribed development systems;
- sleep on the condition variable only when no MPI requests are active.

The default and backoff values must be benchmarked on target systems. Do not
use `MPI_Waitsome` in the main loop because it can prevent newly submitted
commands from being started.

### Communicator ownership

The MPI thread creates and owns all communicators. Application threads receive
plain immutable topology data and opaque `CommunicatorId` values.

`GridController` data needed outside the MPI thread is copied into an
immutable snapshot:

```text
global rank
world size
Cartesian coordinates
neighbor ranks
periodicity
host-local rank
```

No accessor returns a raw MPI communicator.

Collective submission order remains the application's responsibility. The MPI
executor preserves FIFO submission order per communicator and adds debug
sequence checks, but it cannot make inconsistent collective control flow
between ranks correct.

### MPI-enabled third-party libraries

Inventory openPMD, ADIOS, HDF5, and other libraries that may call MPI
internally. Any MPI-enabled entry point must execute on the MPI thread.

A temporary internal `invokeOnMpiThread` migration facility may run a bounded
callable on the MPI thread. It is not a public general task API. Long blocking
library operations stall progress, so they are permitted only at an explicit
MPI quiescence point where no simulation request is active. Replace temporary
uses with asynchronous executor operations where practical.

### Enforcement

After MPI migration, CI rejects MPI usage outside an explicit allowlist under
the MPI implementation directory. This includes direct calls hidden in
plugins and tests.

Debug builds record the MPI owner thread ID and assert it at every internal
MPI entry point.

`MPI_Abort` is requested through the MPI executor. If the MPI thread itself is
unusable, the fallback is process termination, not an MPI call from another
thread.

## Device executor

Use one device executor owner per accelerator initially. It owns:

- alpaka queues;
- native device events;
- event pooling;
- queued launch commands;
- pending device completions.

All producer threads may submit work. Only the owner calls backend queue and
event APIs. This avoids relying on backend thread-safety and avoids driver
contention from many submitters.

Dependency lowering:

```text
same queue dependency       -> FIFO ordering, no host wait
different queue, same device -> native queue wait
MPI or host dependency      -> post command after Event completion
```

The executor must continue submitting work ahead of device completion. A GPU
operation must not host-wait merely because its predecessor is unfinished on
the device.

Initially record a completion event for each exported asynchronous operation.
After correctness is established, coalesce events within a same-queue `Flow`
and record fences only at cross-queue, cross-domain, join, and host-wait
boundaries.

CPU alpaka backends need a separate policy because a queue operation may run
work on the submitting thread. Reuse the same event API, but do not assume the
GPU executor's polling and thread-count policy is optimal for CPU execution.

## Buffer lifetime and access

Every asynchronous command retains the underlying allocation until native
completion.

```cpp
struct BufferView
{
    AllocationHandle allocation;
    void* data;
    Extents extents;
};
```

`BufferLease` or `AllocationHandle` should be intrusive or shared ownership of
the allocation, not ownership of the high-level simulation object.

Consequences:

- MPI sends cannot outlive their source allocation;
- receives cannot outlive their destination allocation;
- buffer destruction waits only when required by that allocation, not on a
  global frontier;
- GPU-aware MPI explicitly depends on the last use of its device buffer;
- buffer accessors no longer call a global `startOperation()`.

Do not initially implement automatic release-build hazard scheduling. Add a
debug-only read/write access annotation system after the explicit dependency
API is stable. It should detect unordered overlapping accesses without adding
release-build serialization.

## Communication composition

### Send

A host-staged send is an ordinary chain:

```text
data ready
-> pack/copy into contiguous device buffer
-> copy to host buffer
-> MPI_Isend
-> send complete
```

A GPU-aware send omits the host copy:

```text
data ready
-> pack into contiguous device buffer if needed
-> wait for device completion on MPI thread
-> MPI_Isend(device pointer)
-> send complete
```

There is no `TaskSend`, child observer, or `TaskSendMPI` state machine.

### Receive

```text
receive buffer available
-> MPI_Irecv
-> immutable ReceiveResult with received count
-> resize metadata
-> host-to-device copy if needed
-> unpack into destination view
-> receive complete
```

The receive may be posted independently of unrelated compute work. GPU-aware
MPI receives depend on destination-buffer availability rather than the whole
simulation frontier.

### Fields

Field exchange becomes:

```text
fork one Flow per direction
send direction: pack -> send
receive direction: receive -> unpack/insert
whenAll(direction completions)
```

The caller joins communication with core computation before border work.
There is no parent field task that polls child IDs.

### Particles

Particle exchange retains its required dynamic chunk loop:

```text
pack chunk
-> obtain count
-> send chunk
-> repeat if chunk was full
```

and:

```text
receive chunk
-> insert chunk
-> repeat if chunk was full
-> fill border gaps after all directions finish
```

Implement the first version with explicit continuations. `Future<T>` may gain a
C++20 coroutine adapter later to express these loops as normal control flow,
but the coroutine adapter must use the same event and executor core rather
than introduce another scheduler.

## Error handling and shutdown

Executor operations complete as ready, failed, or cancelled. Dependent work
propagates predecessor failure by default and does not execute.

Rules:

- no exception escapes an executor thread;
- `Future<T>::result()` and blocking waits report stored errors;
- no blocking wait is allowed from an executor on work that requires that
  executor;
- communicator error handlers use `MPI_ERRORS_RETURN` where possible;
- shutdown rejects new application submissions;
- executors drain or explicitly fail queued work;
- active MPI requests are completed or handled by the documented fatal path;
- device queues finish before native resources are destroyed;
- MPI resources are freed and `MPI_Finalize` runs on the MPI thread;
- cooperative tasks cannot escape shutdown accounting.

Application-thread failure is sent to the MPI thread. The MPI thread then
coordinates clean shutdown or invokes `MPI_Abort` according to failure type.

## Legacy components to remove

The completed migration removes:

- `pmacc::Manager`;
- `TransactionManager` and `Transaction`;
- the global transaction API;
- `ITask`, `DeviceTask`, and `MPITask`;
- `Factory`, `FieldFactory`, and `ParticleFactory` task allocation;
- `EventNotify`, `IEvent`, `IEventData`, and `EventType`;
- integer `EventTask` IDs;
- `TaskLogicalAnd`;
- `TaskSendMPI` and `TaskReceiveMPI`;
- `TaskSend` and `TaskReceive`;
- field and particle parent polling tasks;
- direct MPI access from `CommunicatorMPI` callers;
- heap-allocated `MPI_Request` objects;
- `mpiBlocking()` and manual event-system pumping around collectives.

The native queue and event wrappers may be retained temporarily, but ownership
moves into `DeviceExecutor` and manual intrusive event handling should be
simplified where possible.

## Migration phases

The migration has a hard project boundary:

- Phases 0 through 6 change only the Caravan targets, PMacc, PMacc
  tests, and `share/pmacc/examples/gameOfLife2D` and
  `share/pmacc/examples/heatEquation2D`.
- No migration work is done under `include/picongpu` or `share/picongpu`
  during these phases. PIConGPU compatibility is not preserved: its build may
  break as legacy PMacc interfaces are removed.
- Do not add or retain adapters solely to keep untouched PIConGPU code working.
  Delete each legacy component as soon as its last migrated PMacc user is gone.
- Phase 7 cannot start until the PMacc completion gate at the end of Phase 6
  passes. Phase 7 then restores PIConGPU by porting it to the new APIs rather
  than restoring removed compatibility interfaces.

Each phase's in-scope runtime, PMacc, and PMacc-example targets must build and
test before the next phase starts. Target-GPU measurements may remain pending
through implementation, but must be collected from the recorded baseline
revision before the Phase 6 exit gate and performance comparison. This deferral
does not relax CPU tests, available CUDA compile checks, or phase-local
correctness gates. PIConGPU is required to build again only at the end of
Phase 7. Do not maintain two independent long-lived runtimes; adapters are
temporary PMacc migration tools and are deleted immediately after their last
in-scope user is ported.

### Phase 0: PMacc inventory and baseline

1. Inventory every direct MPI call in PMacc, PMacc examples, and PMacc tests.
   Defer the PIConGPU and enabled third-party inventory to Phase 7.
2. Classify calls as bootstrap, topology, point-to-point, collective, signal,
   shutdown, or error handling.
3. Inventory every `EventTask`, transaction, manager wait, and custom task
   state machine used by PMacc and its examples.
4. Record CPU-serial and CUDA compile baselines for PMacc and both target
   examples. Also record one untouched PIConGPU build, behavior, and full-step
   performance baseline for later comparison, without inventorying or changing
   PIConGPU source.
5. Add focused current-behavior integration tests for:
   - device operation ordering;
   - fork/join halo exchange;
   - host-staged MPI exchange;
   - GPU-aware MPI exchange where hardware is available;
   - field exchange;
   - multi-chunk particle exchange;
   - signal barriers;
   - shutdown with outstanding work.
6. Record reproducible output checks for `gameOfLife2D` and `heatEquation2D`,
   including their multi-rank paths.
7. Benchmark current host submission cost, manager CPU time, MPI ping-pong,
   halo exchange overlap, and both example runtimes.

Implementation exit criterion: the CPU behavior baseline, inventories,
migration paths, and hardware-independent regression tests are recorded. This
permits PMacc migration work to proceed.

Deferred target criterion: target CUDA and GPU-aware MPI behavior and
performance baselines are recorded from the baseline revision before the
Phase 6 exit gate. Phase 7 cannot begin without them.

### Phase 1: Completion core

1. Implement `Event`, `Future<T>`, failure propagation, and `whenAll()` in
   `caravan::core`.
2. Implement target-executor continuation posting without inline recursive
   callback execution.
3. Add a deterministic inline test executor.
4. Add multithreaded tests for completion/registration races.
5. Add blocking-wait deadlock guards.

Exit criterion: ThreadSanitizer passes completion-core tests, every state
transition is exactly once, and `whenAll()` uses one node rather than a tree.

### Phase 2: Dedicated MPI runtime

1. Refactor PMacc example startup so process main owns MPI and the simulation
   runs on an application thread.
2. Request and require `MPI_THREAD_FUNNELED`.
3. Implement the MPI submission queue, executor loop, contiguous active request
   storage, and `MPI_Testsome` progress in `caravan::mpi`.
4. Move MPI initialization, topology setup, communicator management, and
   finalization into the MPI thread.
5. Expose immutable topology snapshots and opaque communicator IDs.
6. Implement point-to-point futures and receive results.
7. Implement nonblocking collective and barrier futures.
8. Route PMacc signal handling through the MPI executor.
9. Temporarily adapt legacy `TaskSendMPI` and `TaskReceiveMPI` to submit to the
   MPI executor and poll only the returned future.
10. Route remaining direct MPI operations in PMacc and its examples through
    migration wrappers on the MPI thread.
11. Add a PMacc-scoped CI direct-MPI allowlist. PIConGPU is out of scope until
    Phase 7 and receives no compatibility exemption.

Exit criterion: no PMacc example, PMacc task, or PMacc helper thread calls MPI;
the process main thread continues progressing requests while the application
thread sleeps or computes.

### Phase 3: Device executor

1. Implement one device executor owner and thread-safe submission in
   `caravan::alpaka`.
2. Move alpaka queue and native event ownership into it.
3. Implement same-queue FIFO and cross-queue native-wait dependency lowering.
4. Implement device completion publication and cross-domain completion.
5. Adapt existing kernel, copy, fill, and size operations to return new Events.
6. Preserve CPU backend behavior with an explicit backend policy.

Exit criterion: producer threads can submit device work concurrently, native
GPU dependencies do not host-wait, and device event completion on migrated
PMacc paths does not depend on the legacy manager.

### Phase 4: Explicit `Flow` and buffer leases

1. Implement local `Flow` sequencing, fork, and join.
2. Remove event-system hooks from buffer accessors.
3. Add allocation leases to asynchronous buffer views.
4. Port basic kernel, copy, set-value, and size-transfer call sites.
5. Add debug checks for executor-thread blocking waits and invalid lifetimes.

Exit criterion: a representative PMacc example step uses explicit Flows and
no global transaction state for device ordering.

### Phase 5: Generic PMacc communication and examples

1. Port `Exchange` send and receive to operation chains.
2. Preserve host staging, device double buffering, and GPU-aware MPI.
3. Return immutable receive counts.
4. Port `GridBuffer::asyncCommunication` to direction Flows and flat joins.
5. Validate per-direction buffer-reuse dependencies across time steps.
6. Port `gameOfLife2D` to explicit fork/join Flows.
7. Port `heatEquation2D` to explicit communication and compute Flows, including
   gather and reduction operations.
8. Run both examples on their CPU paths and compile their CUDA paths. Exercise
   the existing multi-rank configurations.

Exit criterion: both PMacc examples use the new runtime and pass their recorded
behavior checks. Legacy send and receive tasks are retained only if a remaining
Phase 6 PMacc migration step still needs them, never for PIConGPU compatibility.

### Phase 6: Complete PMacc and pass the PIConGPU entry gate

1. Replace PMacc field send/receive parent tasks with direction Flows.
2. Replace PMacc particle send/receive tasks with continuation-based chunk
   loops.
3. Join all receive directions before field insertion or particle gap filling.
4. Add exact-capacity and multi-chunk stress tests.
5. Port remaining PMacc reductions, gather operations, signals, examples, and
   tests.
6. Remove `FieldFactory`, `ParticleFactory`, and their task classes as soon as
   their migrated PMacc replacements pass, without retaining PIConGPU
   compatibility wrappers.
7. Delete Manager, transactions, legacy tasks, observers, IDs, factories,
   event pumping, and all PMacc migration adapters after their last PMacc use.
8. Run the complete PMacc unit and integration test suite, `gameOfLife2D`, and
   `heatEquation2D` with CPU execution and CUDA compile validation.
9. Compare PMacc and example behavior and performance with the Phase 0
   baselines.

Exit criterion and PIConGPU entry gate: PMacc and both target examples use the
new runtime without manager polling or global transaction state, all PMacc
tests pass, MPI progress is owned by the process main thread, and the legacy
runtime has been deleted. No PIConGPU source has been changed, and PIConGPU may
be broken by the removed PMacc interfaces. Only then may Phase 7 begin.

### Phase 7: PIConGPU inventory and migration

1. Inventory and classify every direct MPI call, event-system use, manager
   wait, transaction, and custom task state machine in PIConGPU and enabled
   third-party integrations.
2. Use the untouched Phase 0 PIConGPU baseline as the migration reference; do
   not restore deleted legacy interfaces merely to reproduce it.
3. Port PIConGPU field and particle call sites to the already tested PMacc
   operation-chain APIs.
4. Port reductions, gather, checkpoints, signals, plugins, diagnostics,
   examples, and tests.
5. Move MPI-enabled external-library calls onto the MPI thread at documented
   quiescence points or replace them with async operations.
6. Remove all raw communicator access from public APIs.
7. Enable the final CI rule forbidding MPI outside the implementation layer.

Exit criterion: all supported PMacc and PIConGPU configurations use the new
runtime and obey single-thread MPI ownership.

### Phase 8: Final cleanup and documentation

1. Remove any migration-only wrappers introduced while porting PIConGPU.
2. Remove stale legacy includes, tests, documentation, and build rules.
3. Update PMacc and PIConGPU documentation and examples.
4. Verify shutdown no longer depends on singleton destruction order.

Exit criterion: no legacy event-system source, compatibility API, or migration
wrapper remains.

### Phase 9: Profile-driven optimization

Only after correctness and migration:

1. Batch executor queue drains and native submissions.
2. Coalesce same-queue device events at dependency boundaries.
3. Replace shared-state allocation with a slab if profiles justify it.
4. Replace mutex queues only if contention is measured.
5. Tune and document MPI polling, yielding, and CPU affinity.
6. Evaluate more than one device submission owner only if one owner is a
   measured bottleneck.

Exit criterion: no full-step performance regression and measured scheduling or
communication improvements over the Phase 0 baseline.

## Testing strategy

### Unit tests

- completion before and after continuation registration;
- simultaneous completion and registration from many threads;
- exactly-once continuation scheduling;
- flat `whenAll()` for zero, one, and many dependencies;
- `whenAll()` waits for all inputs after an early failure or cancellation;
- failure propagation;
- executor shutdown with queued commands;
- buffer lease lifetime;
- invalid wait from owner executor thread;
- Flow fork and join semantics.

### MPI integration tests

Run with at least one, two, and four ranks:

- thread ownership assertion for every MPI operation;
- eager and rendezvous-sized send/receive;
- wildcard receive status and received count;
- bidirectional neighbor exchange;
- nonblocking collectives;
- separate signal communicator;
- progress while the application thread sleeps or performs CPU work;
- new submissions while requests are active;
- shutdown with requests in flight;
- propagated MPI failure where practical;
- repeated communicator creation/destruction on the MPI thread.

### Device integration tests

- same-queue ordering;
- cross-queue waits;
- GPU-to-MPI and MPI-to-GPU dependencies;
- host-staged and GPU-aware paths;
- producer contention from multiple threads;
- event and allocation lifetime after high-level buffer destruction.

### PMacc example regression tests

These gate all PIConGPU source changes:

- `gameOfLife2D` fork/join and multi-rank halo exchange;
- `heatEquation2D` communication, gather, reduction, and multi-rank output;
- field halo exchange;
- particle exchange with empty, partial, exact-capacity, and multi-chunk data.

### PIConGPU regression tests

These start only after the Phase 6 PMacc gate passes:

- field and particle communication;
- checkpoint and plugin synchronization;
- CPU serial backend runtime tests;
- CPU threaded backend tests;
- CUDA compile-only validation on the local no-GPU machine;
- CUDA runtime tests in CI or on suitable hardware.

Use ThreadSanitizer on the completion core, submission queues, fake executors,
and CPU backend where supported.

## Performance acceptance criteria

1. MPI makes progress while simulation threads do not call the runtime.
2. MPI ping-pong latency and bandwidth remain within 5 percent of equivalent
   direct nonblocking MPI for representative message sizes.
3. No full simulation-step regression greater than 2 percent on the primary
   GPU configuration without an understood and accepted cause.
4. Host scheduling time is lower than the legacy manager for representative
   kernel and exchange counts.
5. No scan cost proportional to all outstanding operations.
6. Join cost is linear in the submitted dependencies with one completion node,
   not a heap-allocated binary task tree.
7. GPU operations depending only on GPU work are enqueued without waiting for
   host-observed completion.
8. The final implementation has a net reduction in event/communication runtime
   code and removes all legacy task classes.

## Main risks and mitigations

### MPI must remain on the process main thread

Mitigation: the process main thread is the MPI executor, and the existing
simulation entry point moves to an application thread. Test startup and
shutdown on Open MPI, MPICH, and target system MPI implementations.

### Dedicated progress consumes a CPU core

Mitigation: reserve and optionally pin one core per rank in production. Provide
a yield policy for oversubscribed development runs. Benchmark before selecting
poll defaults.

### MPI-enabled external libraries may block progress

Mitigation: inventory them before implementation, execute them only on the MPI
thread, require MPI quiescence for blocking calls, and prefer asynchronous
library APIs where available.

### Cross-domain dependencies can accidentally host-synchronize GPU work

Mitigation: represent native device submission separately from host completion
and lower device-to-device dependencies to native queue waits.

### Executor queues can become a launch bottleneck

Mitigation: batch drain, retain a single owner for correctness, and add shards
only after profiling. Do not start with lock-free queues.

### Buffer views may outlive high-level objects

Mitigation: every asynchronous command owns an allocation lease until native
completion. Add focused lifetime tests.

### Collective ordering can still deadlock

Mitigation: preserve FIFO order per communicator, use nonblocking collectives,
add debug sequence metadata, and keep collective control flow explicit in the
simulation stages.

## Definition of done

The replacement is complete when:

- process main is the sole MPI-calling thread under `MPI_THREAD_FUNNELED`;
- no other source file can call MPI directly;
- MPI progresses independently of simulation-thread polling;
- task submission is safe from multiple threads;
- submitted commands are immutable and executor-owned;
- all dependencies are explicit Events or Futures;
- no global transaction stack or task manager remains;
- no destructor drives task completion;
- no field or particle enum state machine exists solely for async sequencing;
- errors and buffer lifetimes propagate through completion objects;
- CPU and GPU test matrices pass;
- performance meets the acceptance criteria;
- legacy event-system code is deleted rather than retained as compatibility
  scaffolding;
- the runtime targets have no dependency on PMacc or PIConGPU headers.
