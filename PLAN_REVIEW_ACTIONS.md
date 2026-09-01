# Caravan Event-System Rework: Review Findings and Near-Term Actions

## Purpose and relationship to `PLAN.md`

This document supplements [`PLAN.md`](PLAN.md). It turns the implementation
review of commit `3073aba0a37ee5763521ee79f838f8e71ef85daf` (2026-08-29)
into concrete work items, design gates, and acceptance tests.

The architecture in `PLAN.md` remains the target. In particular, this review
does **not** reopen the following settled decisions:

- Caravan is sender-oriented and semantically aligned with P2300;
- primitive backend operations are lazy and begin native work on operation start;
- `Event` and `Future<T>` are eager, type-erased migration/interop bridges;
- PMacc owns its application async scope and optional single-thread control run
  loop;
- `MpiContext` is a native progress/lifecycle authority, not a general scheduler;
- Caravan core owns operation-state lifetime, not arbitrary application storage;
- resource-access dependency inference is optional and remains outside core;
- accelerator-native dependencies remain backend-local until a real second
  interop path justifies a cross-backend protocol; and
- no global Caravan manager, task hierarchy, worker pool, or mandatory resource
  scheduler is introduced.

The current implementation is best described as the end of the representative
Phase 5 prototype: the hardware-independent Caravan architecture and one PMacc
compute path exist, but Phase 6 communication migration and Phase 7 legacy-system
removal remain open.

---

## Priority and ordering

| ID | Priority | Action | Blocks |
|---|---|---|---|
| A1 | P0 correctness | Fix eager runtime-sized `whenAll` terminal precedence | General use of dynamic joins |
| A2 | P0 design/correctness | Define and implement deterministic collective initiation order | Phase 6 multi-rank communication |
| A3 | P0 execution safety | Make PMacc continuation placement explicit and hard to misuse | More sender composition in PMacc |
| A4 | P1 lifetime/progress | Define non-hanging `AsyncScope` and `Context::wait` contracts | General scope use outside the prototype |
| A5 | P1 API boundary | Separate normal MPI headers from native extension APIs | Treating `caravan::mpi` as a library boundary |
| A6 | P1 migration cleanup | Remove eager predecessor-taking `MpiContext` members | Sender API becoming the single normal MPI API |
| A7 | P1 interoperability | Perform a real standard-execution adapter/model spike | Long-term P2300 compatibility claims |
| A8 | P2 performance | Measure current allocation and dispatch costs | Optimization decisions only |
| M1 | Phase 6 | Finish `gameOfLife2D` communication composition | Phase 6 exit |
| M2 | Phase 6 | Port `heatEquation2D` | Phase 6 exit |
| M3 | Phase 6/7 | Port generic PMacc communication and remove adapters | PIConGPU entry gate |
| V1 | Hardware gate | Run target GPU, HIP, GPU-aware MPI, and performance validation | Phase 7 exit |

Complete A1-A3 before expanding sender-based PMacc communication. A4-A7 may
proceed in parallel with M1 once their public contracts are fixed. Do not begin
PIConGPU source migration before M1-M3 and the Phase 7 gate in `PLAN.md` pass.

---

# Correctness and semantic blockers

## A1: Fix eager runtime-sized `whenAll` precedence

### Problem

The documented quiescent join result precedence is:

```text
failed > stopped > ready
```

The current eager `detail::WhenAllState::arrive()` records a failure only while
the accumulated result is `ready`. Therefore this completion order:

```cpp
first.setStopped();
second.setFailed(error);
```

incorrectly produces `stopped`. The typed sender `whenAll` stores error and
stopped state independently and does not have this specific defect.

### Required implementation

- Track failure independently of stopped state, or allow a later failure to
  replace a previously recorded stopped result.
- Preserve quiescence: the output must remain pending until every input is
  terminal.
- Preserve the current non-promise about which failure is retained when several
  inputs fail concurrently.
- Do not request sibling cancellation from the eager runtime-sized join; it is a
  quiescence primitive.

### Required tests

- ready + ready -> ready;
- stopped + ready -> stopped;
- failed + ready -> failed;
- failed followed by stopped -> failed;
- stopped followed by failed -> failed;
- failure before the last sibling still leaves the join pending;
- the same precedence cases with concurrent completion where practical.

### Done when

- all cases above pass;
- the implementation still invokes no callbacks while holding the result lock;
  and
- ThreadSanitizer coverage remains clean where available.

---

## A2: Make managed collective initiation order deterministic

### Problem

The dedicated MPI context currently allocates a collective lane ticket when a
sender operation calls `MpiContext::Impl::submit()`. This orders collectives by
the time their primitive operation states start.

That is insufficient for this graph:

```cpp
auto first = caravan::letValue(
    firstDependency,
    [&] { return caravan::mpi::allReduce(mpi, firstBuffers, communicator); });

auto second = caravan::letValue(
    secondDependency,
    [&] { return caravan::mpi::allReduce(mpi, secondBuffers, communicator); });
```

If `secondDependency` becomes ready first, `second` can receive the first lane
ticket. If readiness differs between ranks, ranks can initiate collectives in
different orders.

The existing test completes the first dependency first and therefore does not
exercise the dangerous inversion.

### Design gate

Before Phase 6, choose and document one correctness model:

1. **Explicit serial composition baseline.** PMacc composes collectives in one
   explicit per-communicator chain. This is simplest but may serialize completion,
   not merely initiation.
2. **Explicit managed collective sequence.** A PMacc/Caravan-MPI composition
   object reserves logical order before dependency readiness, retires skipped
   entries on error/stopped, and releases the next entry immediately after native
   collective initiation rather than after completion.

The recommended long-term model is option 2, implemented as an explicit
dependency-planning layer above primitive `mpi::request` senders. It must not add
an `Event predecessor` field back to the MPI request engine.

One possible shape is:

```cpp
auto lane = pmaccCollectives.forCommunicator(communicator);

auto first = lane.submit(
    std::move(firstDependency),
    [&] { return caravan::mpi::allReduce(mpi, firstBuffers, communicator); });

auto second = lane.submit(
    std::move(secondDependency),
    [&] { return caravan::mpi::allReduce(mpi, secondBuffers, communicator); });
```

`lane.submit` is a dependency planner. Primitive `caravan::mpi::allReduce`
remains lazy, predecessor-free, and independently usable.

### Required semantics

- Logical order is defined independently of dependency-ready timing.
- Every rank using a managed communicator observes the same logical collective
  sequence.
- A failed or stopped predecessor retires its reserved sequence entry without
  initiating MPI and without permanently blocking following entries.
- Point-to-point operations are not serialized behind the collective lane.
- The next collective may initiate after the prior collective has been initiated;
  waiting for prior completion is not required unless the application graph says
  so.
- Native MPI calls still occur only on the selected MPI authority.
- Raw/native expert invocation that performs collectives has an explicit
  caller-managed or lane-integrated ordering contract.

### Required tests

- Construct two managed collectives in logical order, make the second dependency
  ready first, and verify native initiation still follows logical order.
- Vary dependency-ready timing differently on different ranks.
- Fail the first dependency and verify the following collective initiates.
- Stop the first dependency and verify the following collective initiates.
- Keep a point-to-point receive/send active between the two collective entries and
  verify it progresses independently.
- Run the inversion tests with at least two and four ranks.

### Done when

- the selected semantic model is recorded in `PLAN.md` or MPI documentation;
- the inversion tests pass on multiple ranks; and
- no predecessor `Event` is stored in `MpiContext`, `NativeSubmission`, or the
  generic request engine.

---

## A3: Make continuation execution placement explicit

### Problem

`then` executes its callable on the thread that delivers the predecessor's value
completion. Therefore:

```cpp
asyncContext.spawn(
    caravan::then(
        caravan::mpi::send(mpi, buffer, peer, tag),
        userCode));
```

can execute `userCode` on the MPI progress worker. `Context::spawn()` currently
wraps the already-composed sender in `continuesOn`; that only transfers the final
completion and cannot move an inner `then` that has already executed. The same
issue exists for alpaka host-callback completion.

This behavior is consistent with P2300 execution placement, but it is too easy to
violate PMacc's rule that arbitrary application callbacks must not execute on MPI
or device completion authorities.

### Required API decision

Provide and document one obvious PMacc operation that transfers a sender to the
control scheduler before application continuations are attached. For example:

```cpp
auto onControl = asyncContext.onControl(
    caravan::mpi::send(mpi, buffer, peer, tag));

auto handled = caravan::then(
    std::move(onControl),
    [](caravan::SendResult result) { handleOnPmaccControlThread(result); });

auto done = asyncContext.spawn(std::move(handled));
```

`onControl()` may initially be a thin wrapper over `continuesOn(sender,
context.scheduler())`. Do not make `MpiContext` or an accelerator completion
authority a public scheduler.

### Required documentation

Document these two distinct patterns:

```cpp
// No user continuation: spawn may transfer only final completion.
auto done = asyncContext.spawn(backendSender);

// User continuation: transfer before attaching then/letValue application code.
auto handled = caravan::then(
    asyncContext.onControl(backendSender),
    userCode);
```

Backend-to-backend composition that should retain native execution placement must
remain possible without an unnecessary control-loop transfer.

### Required tests

- `then(onControl(mpiSender), f)` runs `f` on the run-loop thread, never the MPI
  worker.
- `then(onControl(alpakaSender), f)` runs `f` on the run-loop thread, never the
  alpaka completion callback thread when those differ.
- `letValue(alpakaSender, mpiFactory)` still crosses directly through the intended
  host-completion boundary and does not first execute on the PMacc run loop unless
  explicitly requested.
- Nested `then`/`letValue` examples document where each callable runs.

### Done when

- PMacc sender examples use the placement operation consistently;
- tests assert thread placement; and
- no documentation implies that `Context::spawn(then(sender, f))` moves `f` to
  the control loop.

---

# Lifetime, progress, and public API boundaries

## A4: Define `AsyncScope` destruction and `Context::wait` contracts

### Problem

`AsyncScope::~AsyncScope()` currently calls `join().wait()`. A raw scope can
therefore block forever if one of its operations has posted completion to a
manually driven run loop and no thread drives that loop.

`pmacc::async::Context` avoids that case for its own spawned operations by driving
its run loop while waiting. However, `Context::wait(Event)` accepts any Event. If
the Event becomes ready without posting a control-loop task after `runOne()` has
begun blocking, the wait can also stall.

### Required decision

Choose and document an explicit scope lifetime contract. The recommended contract
is:

- spawning is eager and operation states are owned until terminal receiver
  completion;
- `join()` closes the scope and returns quiescent completion;
- the owner explicitly provides progress while joining;
- raw `AsyncScope` destruction does not silently attempt unbounded progress;
- destruction of a non-quiescent or unjoined scope is diagnosed according to a
  documented debug/release policy; and
- `pmacc::async::Context` performs its explicit progress-aware join before its
  member scope is destroyed.

If blocking destruction is retained temporarily, document the exact progress
precondition and add a diagnostic for scopes containing control-loop-bound work
without an active driver.

### Required implementation/tests

- Add observable internal state sufficient to distinguish open, joining, joined,
  and non-quiescent destruction.
- Test destruction after explicit successful join.
- Test the documented behavior for destruction with pending work.
- Test a scope containing `continuesOn(..., loop.scheduler())` with no independent
  loop driver; it must diagnose rather than hang the test suite.
- Restrict or document `Context::wait` to Events whose terminal progress is
  guaranteed to wake/post to that context.
- Add a regression where an Event completes concurrently with the transition into
  the run-loop wait.

### Done when

- no supported scope/context usage can silently hang because its required run loop
  is not driven; and
- PMacc shutdown remains quiescent and deterministic.

---

## A5: Enforce the normal/native MPI header split

### Problem

The intended public layering is:

```text
caravan/mpi.hpp
    context and normal typed sender operations

caravan/mpi/native.hpp
    NativeMpiContext, NativeRequestBatch, request/invoke escape hatches
```

Currently `mpi/operations.hpp` includes `mpi/native.hpp`, so normal users
transitively see the native extension surface and `mpi.h`.

### Required implementation

- Move shared request-sender machinery needed by typed operations into an internal
  `caravan/mpi/detail/...` header or a concrete non-native public sender type.
- Keep `mpi::send`, `receive`, reductions, gathers, barrier, and communicator
  operations available from `caravan/mpi.hpp`.
- Require an explicit include of `caravan/mpi/native.hpp` for
  `NativeMpiContext`, `NativeRequestBatch`, `mpi::request`, `mpi::invoke`, and
  `mpi::invokeBlocking`.
- Do not duplicate the native request progress engine.
- Do not add a parallel Caravan type hierarchy for all MPI types.

### Required tests

- A translation unit including only `caravan/mpi.hpp` can use every normal typed
  operation.
- A translation unit using native request/invocation APIs must include
  `caravan/mpi/native.hpp` explicitly.
- Core-only tests do not require MPI headers or linkage.
- MPI-native tests still use the same request engine as typed convenience
  operations.

### Done when

- header dependency checks demonstrate the intended split; and
- the normal public header no longer exposes native extension declarations merely
  through transitive inclusion.

---

## A6: Remove predecessor-taking eager `MpiContext` operations

### Problem

The public `MpiContext` currently exposes both the intended lazy free functions
and migration-era eager members:

```cpp
// Intended normal API
auto caravan::mpi::allReduce(MpiContext&, ...);
auto caravan::mpi::barrier(MpiContext&, ...);

// Migration API still public
Future<AllReduceResult> MpiContext::allReduce(Event predecessor, ...);
Event MpiContext::barrier(Event predecessor, ...);
```

This makes `Event predecessor` appear to be a permanent backend dependency model.

### Required migration

1. Inventory every call to the eager `MpiContext` member operations.
2. Convert each call to typed sender composition plus an explicit
   `AsyncScope`/PMacc context boundary.
3. Keep any unavoidable legacy adaptation in a named PMacc migration header, not
   as a normal `MpiContext` member.
4. Delete the eager members after the last call site is ported; breaking PMacc
   interfaces is allowed by `PLAN.md`.
5. Delete `submitWhenReady` and predecessor-taking native Event/Future helpers
   when their final compatibility user disappears.

### Done when

- normal `MpiContext` exposes topology/lifecycle authority only;
- all normal operations are lazy `caravan::mpi` sender factories; and
- no backend queue or native submission stores an `Event` predecessor.

---

# Standard execution alignment

## A7: Perform a real P2300/`std::execution` compatibility spike

### Current limitation

The custom API matches the P2300 lifecycle shape but does not model standard
execution concepts. In particular, it currently has:

- nested custom `completion_signatures` rather than standard completion function
  signatures;
- member `.connect()` and `.start()` rather than customization-point operations;
- no receiver environment or `get_env`;
- no completion scheduler/domain attributes;
- a `RunLoopScheduler::post()` handle rather than a scheduler whose `schedule()`
  returns a sender;
- no stop-token propagation;
- only `std::exception_ptr` error completions;
- non-pipeable camelCase algorithms; and
- explicit `caravan::alpaka::then(left, right)` fusion rather than standard domain
  transformation/customization.

Do not describe current sender types as source-compatible P2300 senders.

### Spike scope

Using one production-quality P2300 implementation or the available standard
execution implementation, build exactly this chain:

```cpp
namespace ex = std::execution; // or selected implementation namespace

auto chain
    = caravan::alpaka::kernel(queue, workDiv, kernel, args...)
    | ex::let_value(
          [&]
          {
              return caravan::mpi::send(mpi, buffer, peer, tag);
          })
    | ex::continues_on(controlLoop.get_scheduler())
    | ex::then(userContinuation);
```

Exercise it through standard-style eager lifetime management:

```cpp
auto result = ex::spawn_future(std::move(chain), scope.get_token());
auto value = std::this_thread::sync_wait(std::move(result));
scope.close();
std::this_thread::sync_wait(scope.join());
```

Equivalent names from the selected implementation are acceptable for the spike.

### Questions the spike must answer

- Can MPI and alpaka senders directly model the standard sender contract, or is a
  thin adapter boundary preferable?
- What minimum environment data is needed for continuation placement and stop
  propagation?
- Can alpaka-native FIFO/event fusion be implemented as a sender domain
  transformation without introducing alpaka types into `caravan::core`?
- Can the standard run loop replace Caravan's run loop on supported host
  toolchains?
- Can a counting scope replace `AsyncScope` without changing MPI/alpaka backend
  implementations?
- What CUDA and HIP translation limitations arise from standard sender expression
  types?
- What compile-time, diagnostic, binary-size, allocation, and runtime costs are
  measured?
- Where are deliberate `Event` type-erasure firebreaks still required?

### Compatibility policy until the spike completes

- Keep adding only the minimal generic algorithms required by migration.
- Do not reproduce broad environment/domain/customization machinery speculatively.
- Do not stabilize the current `caravan::Sender` concept as an independent public
  ecosystem.
- Preserve backend implementation independence from the selected composition
  syntax.

### Done when

- the representative chain runs on CPU and compiles through supported CUDA and
  HIP translation paths where available;
- direct modeling versus adapter strategy is documented;
- concrete blockers, if any, are recorded; and
- adopting standard composition would not require rewriting MPI progress or
  alpaka native integration.

### Outcome

Completed with NVIDIA/stdexec `nvhpc-25.09`. A thin adapter runs the representative
CPU chain without backend changes, and the chain translates with nvcc 13.3. The
stdexec async-scope path does not instantiate under that nvcc version; HIP was not
available. Decisions, measurements, diagnostics, and reproduction steps are in
[`docs/CARAVAN_STDEXEC_SPIKE.md`](docs/CARAVAN_STDEXEC_SPIKE.md).

---

## A8: Measure before optimizing the custom core

The current implementation deliberately uses simple ownership and queues. Known
costs include:

- a `shared_ptr`/heap allocation for transferred values in `continuesOn`;
- `std::function` storage in the run loop and eager continuation lists;
- one type-erased, shared operation object plus `unordered_map` entry per
  `AsyncScope::spawn`; and
- shared eager state for `Event`/`Future` bridges.

### Required measurements

- sender construction, connect, start, and ready-completion latency;
- `continuesOn` allocation count and latency for void and small-value completion;
- `AsyncScope::spawn` allocation count and contention;
- runtime-sized Event join allocation/counting cost;
- PMacc control-loop wakeup and batching cost;
- MPI request submission/progress overhead relative to direct nonblocking MPI;
- full representative step cost relative to the legacy Manager.

### Optimization rule

Do not replace shared state, mutexes, `std::function`, or queues until a profile
shows material cost. Any replacement must retain exactly-once completion,
operation-state lifetime, non-recursive dispatch, and backend independence.

---

# Phase 6 migration actions

## M1: Finish the `gameOfLife2D` sender graph

### Current mixed path

The example currently performs:

```cpp
auto communication = read->asyncCommunication(EventTask{}); // legacy
auto core = asyncContext.spawn(evo.runAsync<CORE>(...));     // Caravan

communication.waitForFinished();
asyncContext.wait(core);

auto border = asyncContext.spawn(evo.runAsync<BORDER>(...));
asyncContext.wait(border);
```

### Target graph

The target dependency graph is:

```text
core computation -----------+
                             +--> border computation --> step completion
halo communication ---------+
```

Representative API shape:

```cpp
auto core = evo.runAsync<CORE>(
    queue,
    pmacc::async::retain(readBox, readOwnedView),
    pmacc::async::retain(writeBox, writeOwnedView));

auto communication = read->asyncCommunicationSender(
    mpi,
    queue,
    /* explicit borrowed/owned views */);

auto step = caravan::letValue(
    caravan::whenAll(std::move(core), std::move(communication)),
    [&]
    {
        return evo.runAsync<BORDER>(
            queue,
            pmacc::async::retain(readBox, readOwnedView),
            pmacc::async::retain(writeBox, writeOwnedView));
    });

asyncContext.wait(asyncContext.spawn(std::move(step)));
```

Exact types may differ, but the dependency graph and ownership must remain
explicit.

### Runtime-sized direction boundary

If active directions or branch types require runtime type erasure, use a flat
Event boundary deliberately:

```cpp
std::vector<caravan::Event> directionTails;
for(auto direction : activeDirections)
    directionTails.push_back(asyncContext.spawn(exchangeSender(direction)));

auto communication = caravan::whenAll(directionTails);
```

Do not build a binary Event/task tree. A fully lazy runtime-range sender may be
added later only if it materially simplifies the call site.

### Required tests

- one-rank and four-rank recorded output hashes remain unchanged;
- core overlaps communication where the backend permits it;
- border starts only after core and communication are terminal;
- all active directions are quiescent before buffer reuse;
- operation-owned allocations survive wrapper destruction;
- borrowed storage preconditions are documented/tested;
- no `EventTask`, transaction API, or Manager wait remains in the example step;
- failure and stopped propagation do not start border work.

### Done when

- `gameOfLife2D` uses only Caravan/PMacc async composition for the complete step;
  and
- its communication does not depend on legacy polling tasks.

---

## M2: Port `heatEquation2D`

### Required migration

- Replace split transaction events with explicit sender/Event branches.
- Express halo communication, local computation, gather, reduction, and output
  dependencies locally.
- Route typed MPI operations through normal `caravan::mpi` senders.
- Use a flat dynamic join for runtime-sized branch collections.
- Remove the final global transaction wait.
- Keep output/PNG and reduction host code on the selected PMacc control scheduler.

### Required tests

- four-rank, 1000-step residual remains `4.58358` for the recorded baseline
  configuration;
- gather/reduction results match the baseline;
- no legacy transaction or Manager API remains in the example;
- shutdown with in-flight communication is quiescent; and
- performance is compared with the Phase 0 baseline.

### Done when

- both Phase 6 target examples use the new architecture end to end.

---

## M3: Port generic PMacc communication and remove migration adapters

### Work packages

1. Port `Exchange` send and receive operations to explicit sender chains.
2. Port `GridBuffer::asyncCommunication` to one branch per direction and a flat
   join.
3. Preserve host staging, double buffering, early receive posting, and GPU-aware
   MPI.
4. Return immutable receive metadata/counts.
5. Validate buffer reuse across directions and time steps.
6. Replace field polling task classes and factories.
7. Replace particle enum/polling state machines with sender continuation,
   coroutine, or standard-execution-compatible chunk loops over the same backend
   operations.
8. Port remaining reductions, gathers, signals, tests, and helpers.
9. Delete `TaskSendMPI`, `TaskReceiveMPI`, `TaskSignal`, and other migration
   adapters after their last user is removed.
10. Delete Manager, transaction, observer, task-ID, logical-and, and polling task
    infrastructure only after complete PMacc regression passes.

### Required regression cases

- field exchange in every supported direction;
- particle exchange: empty, partial, exact-capacity, and multi-chunk;
- host-staged and GPU-aware paths;
- signal/checkpoint barriers;
- gather and reductions;
- buffer wrapper destruction before explicitly retained native completion;
- borrowed buffer lifetime contract violations where diagnosable;
- shutdown with queued and in-flight work;
- CPU serial, threaded CPU, CUDA compile/runtime, and multi-rank configurations.

### Done when

- the Phase 7 PIConGPU entry gate from `PLAN.md` passes;
- no legacy PMacc event runtime remains; and
- Caravan targets still contain no PMacc or PIConGPU headers.

---

# Hardware and performance validation

## V1: Close the deferred hardware gates

The following cannot be inferred from CPU tests or CUDA translation alone:

- target accelerator runtime behavior;
- HIP translation/runtime behavior;
- queue callback behavior and thread placement on real GPU backends;
- same-device cross-queue native waits;
- GPU-aware MPI ordering and lifetime;
- MPI progress and core reservation policy on target systems;
- host submission cost and legacy Manager comparison; and
- full-step overlap/performance.

### Required matrix

| Configuration | Required validation |
|---|---|
| CPU serial | core, MPI, alpaka, PMacc examples |
| Threaded CPU/runtime | thread placement, races, progress |
| CUDA translation | representative typed chains and PMacc examples |
| CUDA runtime | queue ordering, callbacks, MPI boundary, lifetime |
| HIP translation/runtime | same coverage as CUDA where supported |
| Multi-rank CPU | 1, 2, and 4 rank MPI tests |
| GPU-aware MPI | send/receive, buffer reuse, shutdown |
| Primary production system | performance acceptance criteria from `PLAN.md` |

### Done when

- all remaining Phase 0/2/4 hardware gates are recorded;
- performance deviations are understood and accepted; and
- no hardware result requires introducing a global scheduler/manager or a
  speculative cross-backend native-event abstraction.

---

# Target public API after Phase 7

The custom migration implementation may remain temporarily, but normal user code
should see one coherent model.

## Generic composition

```cpp
auto sender = backendOperation(...);
auto chained = caravan::letValue(std::move(sender), successorFactory);
auto joined = caravan::whenAll(std::move(first), std::move(second));
auto placed = caravan::continuesOn(std::move(joined), controlScheduler);
```

## MPI

```cpp
auto sent = caravan::mpi::send(mpi, buffer, peer, tag, communicator);
auto received = caravan::mpi::receive(mpi, buffer, peer, tag, communicator);
auto reduced = caravan::mpi::allReduce(mpi, input, output, type, op, communicator);
```

No normal operation accepts an `Event predecessor`.

## Eager/dynamic boundary

```cpp
auto completion = asyncContext.spawn(std::move(sender));
auto value = asyncContext.spawnFuture<Result>(std::move(valueSender));
auto all = caravan::whenAll(std::span<caravan::Event const>{events});
asyncContext.wait(all);
```

## Native MPI extension

```cpp
#include <caravan/mpi/native.hpp>

auto native = caravan::mpi::request<Result>(
    mpi,
    initiateRequests,
    decodeCompletion,
    collectiveOrderingContract);

auto thirdParty = caravan::mpi::invokeBlocking(
    mpi,
    blockingMpiEnabledCall);
```

Dependencies for native/blocking invocation are composed outside the primitive.

## Optional resource planner

If and only if Phase 11 is justified:

```cpp
auto completion = resources.submit(
    operationSender,
    read(fieldE),
    write(fieldB));
```

This remains a removable dependency planner and does not own execution or
application storage.

---

# Review completion checklist

## Must pass before expanding Phase 6 communication

- [x] A1 eager `whenAll` precedence fixed and tested.
- [x] A2 collective ordering semantic selected and inversion-tested on multiple
      ranks.
- [x] A3 PMacc control-continuation placement API and tests added.

## Must pass before treating Caravan headers as a stable internal library boundary

- [x] A4 scope destruction/progress contract defined and tested.
- [x] A5 normal/native MPI header split enforced.
- [x] A6 eager predecessor-taking `MpiContext` API removed or confined to a
      named temporary PMacc adapter.
- [x] A7 standard-execution spike completed and compatibility claims updated.

## Must pass before the PIConGPU entry gate

- [x] M1 `gameOfLife2D` complete graph migrated.
- [ ] M2 `heatEquation2D` migrated.
- [ ] M3 generic PMacc communication and remaining helpers migrated.
- [ ] Legacy Manager, transactions, tasks, observers, and adapters deleted.
- [ ] PMacc CPU/GPU/multi-rank regression matrix passes.
- [ ] V1 target hardware and performance gates pass or deviations are explicitly
      accepted.

---

# Explicit non-actions

Do not respond to these findings by:

- adding Event predecessors back to primitive MPI/alpaka APIs;
- making `MpiContext` a scheduler for application callbacks;
- making the PMacc run loop a Caravan singleton or dependency database;
- implementing a general Caravan thread pool or task hierarchy;
- implementing resource inference before a measured use case exists;
- adding automatic dependency discovery from pointers;
- moving application allocation ownership into Caravan core;
- designing a generic cross-backend native-event protocol from alpaka alone;
- wrapping every MPI routine in a parallel Caravan type/API hierarchy; or
- reimplementing all P2300 environment/domain machinery before the real
  interoperability spike establishes what the backends require.
