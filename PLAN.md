# Caravan Library and PMacc Event-System Migration Plan

## Status

The implementation review and concrete near-term corrective actions for commit
`3073aba0a37ee5763521ee79f838f8e71ef85daf` are tracked in
[`PLAN_REVIEW_ACTIONS.md`](PLAN_REVIEW_ACTIONS.md). That document supplements this
plan; it does not replace the architecture or migration phases below.

Caravan is being developed inside the PMacc/PIConGPU repository while the
legacy PMacc event system is still in use.

The implementation completed so far remains valuable and is not discarded:

- the completion core (`Event`, `Future<T>`, terminal-state handling, continuation
  registration, blocking waits, and flat joins) is implemented and tested;
- the Phase 0 PMacc inventory and CPU baseline are recorded in
  `docs/CARAVAN_PHASE0.md`; target-GPU measurements remain open;
- the current MPI implementation has a dedicated worker that owns the MPI
  lifecycle and progresses nonblocking requests;
- nonblocking barrier, all-reduce, root-reduce, fixed-gather, variable-gather,
  point-to-point futures, receive metadata, communicator creation, topology
  snapshots, and buffer lifetime retention exist;
- PMacc can attach its environment to the current MPI runtime and routes several
  signal, reduction, gather, barrier, and point-to-point paths through it;
- PMacc now has an explicit async context owning a Caravan scope and manually
  driven run loop, and the `gameOfLife2D` local core/border step uses lazy kernel
  senders with explicit allocation retention; buffer accessors no longer mutate
  transaction state;
- the native MPI extension can transfer arbitrary nonblocking requests and their
  retained lifetimes to the progress engine;
- the MPI engine no longer stores predecessor Events; lazy senders submit directly,
  while temporary eager Event adapters subscribe above the backend;
- blocking MPI-context submissions already provide an escape path for blocking
  MPI-enabled operations and third-party libraries;
- the minimal typed sender vocabulary and completion-signature model are
  implemented: typed `then`, value-forwarding `letValue`, fixed-arity sender
  `whenAll`, sender/operation concepts, and Event/sender bridges;
- the manually driven `RunLoop` now exposes a cheap `RunLoopScheduler`, and
  `continuesOn` consumes that scheduler;
- the first alpaka prototype lazily submits caller-defined kernel/copy/fill batches
  to a borrowed queue, retains explicit captures through native completion, preserves
  same-queue FIFO without a host wait, and publishes completion through an alpaka
  host callback;
- legacy `TaskSendMPI` and `TaskReceiveMPI` can temporarily consume Caravan
  completion handles during migration.

The architecture described below changes the *role* of this implementation:
the dedicated MPI worker becomes the first MPI progress/lifecycle policy rather
than a defining property of Caravan, and the current `DeviceExecutor` direction
becomes an alpaka backend policy rather than a universal execution model.

Breaking PMacc and PIConGPU interfaces is allowed during the migration. Do not
retain compatibility scaffolding merely to avoid porting users of the legacy
system.

The immediate implementation order is:

1. **Implemented:** complete the minimal typed sender vocabulary (`then`,
   value-forwarding `letValue`, fixed-arity sender `whenAll`, and one coherent
   sender concept/completion-signature representation).
2. **Implemented:** split the manually driven `RunLoop` from its cheap
   `RunLoopScheduler`, and make `continuesOn` consume the scheduler rather than the
   loop as an "executor".
3. **Implemented:** rename `MpiExecutor` to `MpiContext` and expose it as the MPI
   backend authority while keeping its dedicated worker and progress implementation;
4. **Implemented:** make the typed `mpi::send`, `receive`, collectives, and barrier
   sender factories part of the normal public MPI header, leaving
   request/invoke/native context in an extension/native header;
5. **Implemented:** remove predecessor handling from the MPI engine; temporary
   Event-taking wrappers compose or subscribe above the backend;
6. **CPU prototype implemented; target accelerator runtime validation remains:**
   complete the first real alpaka/device sender architecture test; and only then
7. **Phase 5 representative path implemented:** replace PMacc polling task chains
   with local sender composition and retain Event only at unavoidable legacy
   boundaries; and
8. **Phase 6 M1/M2 implemented:** port generic grid-buffer exchange branches and
   both target example step graphs to explicit sender/Event composition.

---

## Library vision

Caravan is a lightweight C++ library for integrating and composing asynchronous
operations across heterogeneous execution systems.

The design is **sender-oriented and P2300-aligned from this migration onward**,
even where the supported compiler/library stack still requires a small custom
implementation. New backend primitives should be expressible as lazy operations
whose native side effects begin only when the operation is started. Dependencies
should normally be expressed by composition rather than by passing an `Event
predecessor` argument into every backend call.

Caravan defines or bridges common semantics for:

- asynchronous operation start and terminal completion;
- value, error, and stopped completion channels;
- explicit composition and runtime-sized quiescent joins;
- operation-state lifetime until terminal completion;
- eager/type-erased completion handles at runtime boundaries;
- completion subscription and execution-context transfer across runtime boundaries;
- backend-local native dependency preservation;
- interoperability with P2300/C++ standard execution and external task runtimes.

Caravan does **not** generally own application resources merely because an
operation refers to them. Application storage may be borrowed, explicitly owned
by a particular operation, or retained by an application/backend-specific handle.
The core guarantee is that Caravan-owned asynchronous operation state remains valid
until that operation is terminal.

Resource-access dependency inference is an optional layer, not a core invariant.
A resource layer may accept declarations such as `read(A)` and `write(B)`, infer
predecessor relationships, and produce ordinary sender/completion dependencies.
It must remain pluggable and must not turn Caravan core into a mandatory
resource-aware scheduler.

Backend contexts, application async scopes, or external execution systems own
shutdown and accounting for the work they submit. Caravan core does not maintain a
global registry of all outstanding application work.

Caravan should avoid owning how available parallelism is exploited. Scheduling,
worker management, work stealing, CPU parallelism, accelerator execution, and
native progress belong to pluggable execution systems and backend policies.
Possible users or adapters include alpaka, SYCL, Kokkos, Taskflow, HPX, oneTBB,
OpenMP tasks, and implementations of C++ standard execution.

The intended long-term layering is:

```text
                 optional resource dependency layer
                  read/write access declarations
                    dependency inference only
                              |
                              v
                         application
                    PMacc / PIConGPU / other
                              |
                sender/P2300-style composition
                 + async scopes / run loops
                              |
                     +--------+---------+
                     |      Caravan     |
                     |------------------|
                     | backend senders  |
                     | Event bridge     |
                     | dynamic joins    |
                     | interop          |
                     +--------+---------+
                              |
               +--------------+----------------+
               |              |                |
              MPI         accelerator      CPU/task runtime
               |              |                |
        progress policy   alpaka/SYCL/      std::execution/
                          Kokkos/...        HPX/TBB/Taskflow/...
```

`Event`/`Future<T>` remain useful during migration as eager, type-erased completion
bridges for already-started work, runtime-sized containers, and imperative PMacc
interfaces. They are not intended to become a second permanent async programming
model that competes with senders.

Caravan is not required to remain the highest-level composition API forever. If
C++ `std::execution` or a production-quality P2300 implementation satisfies
PMacc's generic composition, async-scope, run-loop, and dynamic-join needs,
Caravan should be able to move underneath it and become primarily an
interoperability/backend library plus optional resource-dependency utilities.

---

## Goals

1. Make asynchronous dependencies explicit and locally understandable.
2. Make new asynchronous backend APIs sender-oriented/P2300-compatible: operation
   description is separate from start, and dependencies are expressed primarily by
   composition rather than predecessor parameters.
3. Remove PMacc's global manager, transaction stack, task IDs, observer system,
   polling task hierarchy, and hidden buffer-access side effects.
4. Keep Caravan core responsible for the lifetime of its own async operation state,
   not for generic ownership of application buffers/fields/particles.
5. Support explicit application async scopes so dynamically spawned work can be
   joined/quiesced without a global Caravan registry.
6. Permit PMacc to use a single-threaded run loop and its scheduler for host
   continuations and progress integration without making that loop a Caravan
   singleton or semantic manager.
7. Keep generic task scheduling and CPU parallelism outside Caravan.
8. Make resource-access dependency inference optional and pluggable; a resource
   layer may generate dependencies but must not be required by backend APIs.
9. Make accelerator support pluggable; alpaka is the first backend, not a core
   architectural assumption.
10. Make MPI integration pluggable; the dedicated-thread `MPI_THREAD_FUNNELED`
    implementation is the first policy, not a universal requirement.
11. Keep MPI progress active independently of simulation-thread polling in the
    PMacc production configuration.
12. Avoid reimplementing MPI. Centralize integration around native nonblocking
    requests plus generic MPI-context invocation.
13. Preserve backend-native accelerator dependency mechanisms so accelerator-to-
    accelerator work does not unnecessarily synchronize through the host.
14. Use portable host-visible completion at cross-backend boundaries initially;
    add a generic native cross-backend dependency protocol only after at least one
    real second interop path demonstrates the required semantics.
15. Make borrowed/owned application-resource semantics explicit at operation or
    PMacc API boundaries; retain storage only when the specific operation/API has
    chosen ownership.
16. Permit future integration with C++ standard execution without requiring a
    rewrite of MPI or accelerator backends.
17. Allow task runtimes to be attached as execution choices without translating
    all application work into a Caravan-specific task type.
18. Keep whole-application supervision and structured lifetime outside
    `caravan::core`; backend contexts account only for native work they must finish
    or destroy safely.
19. Reduce scheduling/progress overhead and total PMacc event/communication code.
20. Keep Caravan independent of PMacc and PIConGPU headers.

---

## Non-goals

- A general work-stealing runtime.
- A mandatory Caravan thread pool for arbitrary application work.
- A Caravan-specific general `Task<T>` abstraction.
- A Caravan-specific general scheduler hierarchy that competes with
  `std::execution`.
- A second incompatible sender/receiver model.
- Owning all CPU parallelism used by PMacc or PIConGPU.
- Replacing alpaka, SYCL, Kokkos, Taskflow, HPX, oneTBB, or similar systems.
- A global Caravan `Manager` that owns every operation in the application.
- Making one PMacc run loop the only valid execution/progress mechanism.
- Automatic dependency discovery from arbitrary pointers.
- Mandatory resource-access dependency inference in Caravan core.
- Making Caravan responsible for generic ownership/reclamation of application
  buffers, fields, particle containers, or arbitrary user objects.
- A new MPI API that mirrors every MPI routine with renamed Caravan types.
- Hiding all native backend types merely for abstraction purity.
- Requiring every backend to support the same native synchronization features.
- Requiring one owner thread for every execution domain.
- Requiring `MPI_THREAD_FUNNELED` for every possible Caravan MPI configuration.
- Promising cancellation of already-submitted MPI or accelerator operations.
- Preserving the current PMacc event-system API.
- Stabilizing a public ABI or extracting a separate package before the PMacc and
  PIConGPU migration demonstrates a second real consumer or a stable boundary.

---

## Architectural principles

### 1. Separate operation description, start, completion, execution placement, progress, and ownership

These are distinct concepts and must not be conflated.

```text
operation description
    What asynchronous work would be performed?

start
    When do externally visible/native side effects begin?

completion
    How does the operation report value/error/stopped terminal state?

execution placement
    Which execution system decides where runnable continuation/callable work runs?

progress
    What mechanism advances already-started native asynchronous work?

operation-state lifetime
    What implementation state must remain alive until completion?

application-resource ownership
    Who guarantees that borrowed or owned application storage remains valid?
```

Examples:

- a sender describes work before `start()`;
- Taskflow/HPX/oneTBB or a standard scheduler may decide where CPU work executes;
- a SYCL/alpaka queue determines accelerator submission semantics;
- an MPI progress policy may repeatedly call `MPI_Testsome`;
- the sender operation state retains its `MPI_Request` bookkeeping and receiver;
- a PMacc buffer allocation may be borrowed or explicitly captured by that
  operation without becoming a generic Caravan-owned resource.

No generic Caravan abstraction should imply that these responsibilities must be
implemented by the same object or thread.

### 2. Sender/P2300 semantics guide new APIs now

New primitive async backend APIs should be designed as if they were sender
factories even when the migration implementation is temporarily custom.

Prefer:

```cpp
auto s = caravan::mpi::send(buffer);
auto work = previous | let_value([&] { return caravan::mpi::send(buffer); });
```

over:

```cpp
Event e = caravan::mpi::send(previous, buffer);
```

A primitive operation should normally be lazy: creating/composing it must not
start MPI, enqueue device work, or otherwise create an externally visible side
effect. Native initiation occurs when the connected operation state is started.

Do not expose implementation-specific sender expression types from stable PMacc
interfaces; use `auto`, local composition, type erasure, or an eager `Event` bridge
where an imperative/runtime-dynamic boundary requires it.

### 3. Submitted/started operation state is immutable in dependency shape

Once an operation is started, its predecessor relation, operation parameters, and
ownership/borrowing choices are fixed. Mutable native execution state has a
backend-defined synchronization authority. That authority may be one dedicated
thread, an external runtime, a thread-safe queue/runtime object, serialized caller
access, or another documented mechanism.

The generic Caravan invariant is **well-defined synchronization authority**, not
**one owner thread per domain**.

### 4. Dependencies are composition; resource inference is optional

The default Caravan model uses explicit sender/completion composition. There is no
implicit current transaction, global dependency cursor, hidden `startOperation()`
call, or buffer accessor that silently changes scheduling.

An optional resource layer may provide:

```cpp
submit(kernel(), read(A), write(B));
```

and infer the necessary predecessor set. That layer then produces ordinary
sender/completion dependencies. It is a dependency planner/tracker, not the owner
of generic execution placement.

The resource layer must be pluggable: users must also be able to use Caravan with
fully explicit composition and no resource registry.

### 5. Generic task scheduling stays external

Caravan must not require arbitrary CPU work to become a Caravan task. Do not
introduce a public general abstraction such as `caravan::Task<T>`, a bespoke
scheduler hierarchy, or a general Caravan thread pool unless a future type is a
direct model/adapter of standard execution and has a concrete interoperability
need.

A task belongs to the runtime executing it. Caravan participates through senders,
completion bridges, schedulers/run loops supplied by consumers, and backend
adapters at boundaries.

### 6. `std::execution` is the long-term generic composition target

P2300-aligned semantics are an architectural constraint from Phase 2 onward,
not merely a late optional adapter exercise. The completed interoperability spike
is recorded in [`docs/CARAVAN_STDEXEC_SPIKE.md`](docs/CARAVAN_STDEXEC_SPIKE.md):
the migration senders require a thin adapter and are not source-compatible P2300
senders.

In particular:

- backend primitives should be representable as senders;
- operation state should naturally correspond to connect/start lifetime;
- value/error/stopped should map directly to receiver completion channels;
- execution placement must be explicit and separable from native progress;
- use standard-style `starts_on`/`continues_on` concepts rather than relying on the
  thread that happens to observe native completion;
- use standard async scopes (`counting_scope`/`simple_counting_scope`-style
  semantics or equivalent) for dynamic structured lifetime when practical;
- use `run_loop`-style schedulers for manually driven single-thread execution when
  practical;
- make it possible to bridge an eager Caravan completion into a sender;
- make it possible to spawn/retain a sender and obtain an eager completion handle
  where runtime type erasure is useful;
- keep MPI and accelerator backend logic independent of the composition syntax.

A third-party P2300 implementation is optional during the migration. Caravan must
not require one before the supported PMacc/PIConGPU toolchains can use it
reliably, but custom APIs should not diverge semantically without a measured need.

Before adding more backend or PMacc composition code, the custom migration layer
must provide one small, typed vocabulary: `then`, value-forwarding `letValue`,
fixed-arity sender `whenAll`, and a coherent sender concept/completion-signature
representation used by those algorithms and backend senders. Do not add the rest
of P2300 speculatively; add environment/query machinery only when continuation
placement or a real backend requires it.

### 7. `Event` is an eager, type-erased completion bridge, not the universal async model

`Event` represents terminal state of work that has already been started/spawned.
It is deliberately different from a lazy sender expression.

Use it where one or more of these properties are valuable:

- cheap sharing;
- runtime type erasure across MPI, accelerator, CPU-runtime, and host operations;
- dynamic storage in containers;
- runtime-sized quiescent joins;
- migration of imperative PMacc APIs;
- bridging already-started native work into another runtime.

Required properties remain exactly-once terminal transition, thread-safe
observation/registration, no user callback from destructors, no callback while
holding an event-state lock, no recursive inline continuation chain, and an
allocation-free already-ready fast path.

`Future<T>` is similarly an eager/type-erased migration/interop facility for a
shared immutable result. It should not grow into a competing general asynchronous
value model if sender value channels satisfy the use case.

### 8. Terminal completion protects Caravan/backend operation state, not arbitrary user storage

A completion handle must not become terminal while native work represented by
that handle can still access **operation state that the operation itself owns or
has explicitly retained**.

This does not imply that Caravan automatically owns every application object
mentioned by an operation. Borrowed storage follows a documented lifetime
precondition. Explicitly owned/captured storage remains alive because the
operation state owns that handle. Backend-affine native resources remain owned and
destroyed by the backend authority.

`stopped` means the operation did not produce a successful result because stop was
honored at a point where the backend could honor it. It does not promise physical
revocation of an already-issued MPI request/kernel/device command.

### 9. Runtime-sized quiescent joins are first-class

PMacc has dynamic collections of asynchronous communication/device operations.
Caravan must retain a flat runtime-sized join that waits until all inputs are
terminal even after one fails or stops, then reports failed > stopped > ready.

The eager migration form may remain:

```cpp
Event joinAll(std::span<Event const> events);
```

A sender-oriented form may later accept a runtime range/type-erased set of child
operations. This is intentionally a quiescence primitive and need not have exactly
the same sibling-stop semantics as standard `when_all`.

### 10. `Flow` is a migration convenience, not a foundation

`Flow` is a local imperative cursor for migrating PMacc transaction-style code.
It must be layered on sender/Event operations and no backend API may require it.

```cpp
Flow main{stepStart};
main.then(...);
auto branch = main.fork();
...
main.join(branch.done());
```

Long term, sender pipelines, coroutines, async scopes, or an external task runtime
should replace most generic `Flow` composition. If forgotten joins recur, provide
a sealed `parallel()`/`forkJoin()` helper or move directly to scoped composition.

### 11. Async scopes own dynamic structured lifetime; core does not globally supervise

Dynamically spawned asynchronous work needs an explicit owner. PMacc should use an
application/migration async scope with semantics comparable to standard execution
scopes: spawn work into the scope, prevent destruction until spawned work is
quiescent, and join at well-defined stage/shutdown boundaries.

The scope is not a global Caravan singleton and is not required by backend APIs.
A future standard `counting_scope`/`simple_counting_scope` or equivalent may replace
the migration implementation without rewriting MPI/alpaka backends.

Backend contexts still account for native work required to destroy that backend
safely; application scopes account for application-level structured lifetime.

### 12. A PMacc run loop is allowed and useful, but it is not the new Manager

PMacc may use one manually driven, single-threaded `RunLoop` for host
continuations, control-plane work, and integration with blocking waits. Scheduling
is exposed through a cheap copyable `RunLoopScheduler` obtained from the loop;
`continuesOn` consumes that scheduler, not the loop itself. If a usable
standard/P2300 `run_loop` exists, prefer it; otherwise the migration implementation
should intentionally match that scheduler-shaped semantic split.

The loop may execute ready host continuations and invoke registered progress
sources while waiting, but it must not:

- own every Caravan operation globally;
- become the dependency database;
- scan all application operations by task ID;
- execute all GPU/MPI work serially;
- become required by Caravan MPI/alpaka APIs.

The control plane may be single-threaded while GPU queues, MPI, and external CPU
runtimes remain concurrently active.

### 13. Resource-access dependency inference is a separate optional layer

A future `caravan::resource`-style target may maintain stable resource identities,
read/write access state, and access leases that remain active until the associated
operation completes. It may infer dependencies such as writer->reader and
readers->writer and then compose/submit ordinary Caravan/P2300-style operations.

It must not implicitly imply ownership of underlying application storage. A resource control
block may need to outlive an access lease; the field/buffer allocation itself may
still be owned by PMacc, PIConGPU, a backend, or explicit operation state.

Start conservatively with logical resources. Split compound objects such as
`Buffer::data` and `Buffer::size` only when their semantics and measured overlap
justify separate synchronization identities.

### 14. Keep native dependency semantics backend-local until real cross-backend interop exists

Portable correctness at a boundary between independent backends uses host-visible
completion initially.

A backend should preserve and exploit its own native dependency representation
internally. For example, an alpaka backend may keep queue/event information needed
for alpaka-to-alpaka ordering without exposing a generic `NativeDependency` type
from `caravan::core`.

Do not design a type-erased cross-backend dependency protocol during the initial
migration. Such a protocol must account for completion, memory visibility, native
resource lifetime, device/context identity, and target-consumption semantics; one
backend is insufficient evidence for the right abstraction.

Introduce a generic import/export capability only when a concrete second path can
be implemented and tested.

```text
within one backend        -> backend-native dependency mechanism
cross-backend boundary    -> portable host-visible completion
future measured interop   -> smallest capability justified by real requirements
```

Backends may expose different native capability levels.

---

## Target library structure

Initial targets should remain small and independently usable:

```text
caravan::core
    minimal sender concept and completion-signature representation
    typed then()/letValue()/fixed-arity sender whenAll()
    RunLoop + cheap RunLoopScheduler and continuesOn()
    Event / Future<T> eager completion bridges
    runtime-sized Event whenAll()/readyEvent()
    terminal/error/stopped semantics
    operation-state/completion utilities
    sender/completion bridge utilities
    continuation dispatch hooks
    optional migration Flow convenience

caravan::mpi
    sender-oriented native MPI async operations
    generic request initiation/progress
    generic MPI-context invocation
    MPI progress/lifecycle policies

caravan::alpaka
    sender-oriented alpaka operation adapters
    supplied queue/event adaptation
    backend-local native dependency chaining

optional future targets
    caravan::resource
        resource identities
        read/write access leases
        dependency inference only
    caravan::stdexec_interop
    caravan::sycl
    caravan::kokkos
    task-runtime adapters only when a real use case needs them
```

The current public header layout mirrors these layers without introducing a
`caravan::core` namespace:

```text
caravan/
|-- core.hpp                 # public core umbrella
|-- core/
|   |-- sender.hpp           # sender vocabulary and generic algorithms
|   |-- eager.hpp            # Event, Future, Promise, bridges, syncWait
|   |-- run_loop.hpp         # RunLoop + RunLoopScheduler
|   `-- async_scope.hpp      # AsyncScope
|-- mpi.hpp                  # normal MPI public umbrella
`-- mpi/
    |-- context.hpp          # MpiContext and common MPI-facing types
    |-- operations.hpp       # typed MPI sender factories
    `-- native.hpp           # native request/invoke escape hatches
```

`core.hpp` includes the complete generic/eager API. `mpi.hpp` exposes the normal
MPI context and typed operations; native integrations include `mpi/native.hpp`
explicitly.

Caravan targets must not include PMacc or PIConGPU headers.

PMacc owns domain-specific composition and application policy:

```text
PMacc
    application async scopes
    optional run_loop/control scheduler
    optional choice/use of caravan::resource
    buffer/allocation ownership policy
    topology policy
    exchange direction/tag mapping
    buffer/view adaptation
    fields
    particles
    signals
    simulation flows
```

PIConGPU builds on the PMacc APIs after PMacc migration is complete.

---

## Core async/completion model

### Sender-oriented primitive operations

New backend primitives should be implementable as lazy operations. Constructing a
primitive should not itself start MPI or enqueue device work. Starting the
connected operation state initiates the native operation through the backend's
synchronization authority.

The implementation used during migration may be custom, but the conceptual model
should remain:

```text
sender/factory
    -> connect(receiver/environment)
    -> operation state
    -> start
    -> native initiation
    -> native progress/completion
    -> value/error/stopped receiver completion
```

Do not add predecessor parameters merely because `Event` is currently convenient.
Compose predecessor work outside the primitive. An eager `Event` API may be
provided as a wrapper that spawns/starts a sender into an explicit scope.

### Minimum generic sender vocabulary

Complete this vocabulary before extending MPI or migrating more PMacc call sites:

- one sender concept based on a coherent completion-signature representation;
- `then(sender, f)`, invoking `f` with the predecessor values and publishing its
  void or non-void result through the resulting sender;
- `letValue(sender, f)`, forwarding predecessor values to `f` and connecting the
  sender returned by `f` without type-erasing the successor operation;
- fixed-arity sender `whenAll(...)`, combining typed value completions while
  propagating error/stopped completion coherently; and
- the existing runtime-sized Event `whenAll(span<Event>)` as the separate eager,
  type-erased quiescence primitive.

Completion signatures are the single source of truth for algorithm result typing
and backend sender declarations. The initial representation need only cover the
channels Caravan actually supports: `set_value(Ts...)`,
`set_error(std::exception_ptr)`, and `set_stopped()`. Do not build a broad query,
environment, or customization framework ahead of a concrete backend need.

### `Event` and `Future<T>` eager bridge

Representative migration interface:

```cpp
enum class CompletionState : std::uint8_t
{
    pending,
    ready,
    failed,
    stopped
};

class Event;
template<typename T> class Future;

Event readyEvent();
Event joinAll(std::span<Event const> events);
```

Required behavior:

- exactly one terminal transition;
- thread-safe observation and continuation registration;
- predecessor failure/stopped propagation in composition helpers by default;
- `joinAll()` completes only after all inputs are terminal;
- failed > stopped > ready precedence for quiescent joins;
- no promise of race-defined first failure;
- terminal completion implies represented native work no longer accesses
  operation-owned/explicitly-retained state;
- blocking waits are available to threads that do not provide required native
  progress themselves, or are implemented by driving the configured PMacc run
  loop/progress integration;
- invalid blocking waits from a backend's own required progress authority are
  diagnosed;
- no task IDs or global lookup;
- no callback execution from destructors;
- no callback invocation under an internal event-state lock;
- no recursive inline continuation chains.

Initial `Event` implementation may continue using `std::shared_ptr`, a mutex,
condition variable, and compact terminal state. Optimization comes only after
profiling. `Future<T>` remains a shared immutable eager result for migration and
runtime boundaries, not a replacement for sender value channels.

### Completion/continuation placement

Native completion and continuation execution are separate. An MPI request may
finish on the MPI progress authority without running arbitrary PIConGPU code on
that thread. Sender composition should use receiver environment/scheduler transfer
or an explicit `continues_on`-style boundary to place application continuations on
the PMacc run loop or another chosen runtime.

For the eager Event bridge, continuation registration similarly accepts a target
dispatch mechanism; completion only makes the continuation eligible.

### Async scopes and backend-local accounting

`caravan::core` does not maintain a global registry of all submitted operations.
Each backend context accounts for native work/resources it must finish or destroy
safely.

PMacc owns an explicit application async scope during migration. Dynamically
spawned sender work is attached to that scope, and stage/shutdown code joins the
scope. Prefer standard/P2300 async-scope semantics where supported; a temporary
PMacc implementation must remain replaceable by them.

A PMacc `RunLoopScheduler` may place host continuations while its owning `RunLoop`
is manually driven during waits. The scheduler is an execution choice owned by
PMacc, not a Caravan global manager.

---

## MPI integration

### Scope and public layering

`caravan::mpi` is an asynchronous integration layer over native MPI, not a new MPI
API and not a general scheduler. The ordinary public MPI header exposes the typed
sender factories applications should normally use: `send`, `receive`, reductions,
gathers, barrier, and similar operations. Generic request submission,
`invoke`/`invokeBlocking`, `NativeMpiContext`, and native request ownership belong
in an explicitly native/extension header. Public header placement must make the
sender path obvious without hiding the native escape hatch.

The hard part is implemented once:

```text
lazy MPI operation description
       |
start operation state in MPI-valid context
       |
initiate native nonblocking operation
       |
store one or more MPI_Request values in backend-owned state
       |
progress requests
       |
decode copied completion/status information
       |
publish value/error/stopped completion
```

Common send/receive/collective helpers are thin sender factories over the generic
request engine, not independent progress state machines.

### Native MPI types

The MPI target may use native MPI concepts such as `MPI_Comm`, `MPI_Datatype`,
`MPI_Op`, `MPI_Status`, and `MPI_Request` where appropriate. Do not create a
parallel Caravan hierarchy mirroring the complete MPI type system.

PMacc may still expose opaque IDs or topology snapshots at its own boundary to
prevent direct MPI access from simulation code.

### Generic nonblocking request sender

The central primitive should support one or more requests created by arbitrary
native initiation code without taking an explicit predecessor argument.

Conceptually:

```cpp
auto request(Initiate initiate, Complete complete /* + explicit captures */);
```

The returned object is sender-like. `connect` constructs operation state; `start`
posts initiation to the selected MPI synchronization authority. The initiate hook
creates `MPI_Request` values and transfers native request ownership into the MPI
progress engine. Completion decodes copied status/result information and completes
the receiver.

Any application storage needed by the request is either borrowed under an explicit
lifetime precondition or explicitly captured/owned by the operation state. This is
not represented by a generic mandatory `LifetimeSet` in Caravan core.

A migration wrapper may spawn this sender into a PMacc async scope and return an
`Event`/`Future<T>` for imperative code.

### Immediate MPI-context invocation

Some MPI operations do not produce a request. Provide one lazy sender-like context
invocation mechanism for short operations that must execute where MPI calls are
permitted:

```cpp
auto invoke(Callable&& mpiCall);  // sender of T
```

Use it for topology/resource queries or communicator setup when no nonblocking
request representation exists. It must not become an escape hatch for arbitrary
expensive application work.

### Blocking MPI-context invocation

Keep one sender-like mechanism for blocking MPI operations and MPI-enabled
third-party libraries that cannot expose nonblocking requests:

```cpp
auto invokeBlocking(Callable&& blockingMpiCall);  // sender of T
```

Dependencies are composed outside the primitive. The MPI backend must not silently
wait for every previously active request: doing so can create dependency cycles.
If a third-party call requires specific outstanding operations to finish, PMacc
composes those dependencies explicitly before the blocking invocation.

Once a blocking call enters a one-thread MPI authority, no other MPI call can run
on that authority until it returns. This physical exclusion is a policy consequence,
not an implicit dependency on unrelated active MPI operations.

### MPI progress and lifecycle are policies

Separate operation semantics, request storage/completion decoding, progress
strategy, and MPI lifecycle strategy.

The first production policy remains the current dedicated worker:

```text
DedicatedThreadMpiPolicy
    worker calls MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)
    worker performs all MPI calls
    worker progresses active requests
    worker frees owned MPI resources
    worker calls MPI_Finalize()
```

This remains the PMacc production configuration during migration because it
provides independent progress and a simple MPI threading contract. The architecture
must not prevent future attach/MULTIPLE/external-runtime/Sessions-based policies,
but do not implement them without a consumer.

### The MPI context/progress authority is not a scheduler

Rename the current `MpiExecutor` to `MpiContext` and expose it as the MPI backend
runtime authority. Keep essentially its current dedicated worker, progress,
lifecycle, and resource-destruction implementation; the change is its public role,
not a rewrite of a good progress engine. `NativeMpiContext` remains the short-lived
native view used by extension hooks.

The progress thread is an implementation authority for MPI initiation/progress and
native MPI resource destruction. It must not be exposed as the place arbitrary
`then` callbacks execute.

When MPI completes a receiver, application continuation execution is transferred
to the scheduler/environment selected by composition (for PMacc, often its
run-loop/control scheduler or another runtime). This is a required P2300 alignment
property to test explicitly.

### Active request storage and progress

For the dedicated-thread policy:

- keep `MPI_Request` by value in contiguous owner-controlled storage;
- use batched progress such as `MPI_Testsome`;
- drain newly startable submissions in batches;
- do not keep dependency-blocked application operations in the active native
  request set; composition decides when their operation state is started;
- post receives as early as their destination/lifetime contract allows;
- spin while active requests exist where a core is reserved;
- optionally yield/back off for oversubscribed development systems;
- sleep only when no progress-requiring request is active.

Do not replace simple queues with lock-free structures without measurement.

### MPI resource handling and collective ordering

The dedicated policy may centrally create/free communicators and other MPI-native
resources to enforce its one-caller contract. Generic Caravan MPI APIs should not
pretend those resources are backend-neutral.

Collective ordering remains an application correctness responsibility across
ranks. PMacc uses the explicit managed-sequence model: it reserves logical order
with `mpi::CollectiveLane::submit()` when building the graph, before predecessor
readiness. Failed and stopped predecessors retire their entries without initiating
MPI. The lane releases the next entry after native operation start, not completion,
and does not serialize point-to-point operations. Every rank must submit and start
the same managed sequence on a communicator. Primitive typed senders remain lazy
and independently usable; code mixing unmanaged or expert collectives with a
managed lane is responsible for their relative ordering.

---

## Accelerator backend model

### No universal `DeviceExecutor`

Caravan core must not require all accelerators to be controlled by one
Caravan-owned submission thread. Each accelerator backend documents native
resources, ownership/borrowing, synchronization authority, operation start,
completion representation, backend-local dependency capabilities, and portable
host-completion fallback.

### alpaka is the first backend

Start with sender-oriented adaptation of caller-supplied alpaka queues/events. The
native alpaka queue is already an asynchronous execution/submission mechanism, so
Caravan should not add another owner thread/queue by default.

A primitive kernel/copy/fill operation should be describable lazily and enqueue
native work when its operation state is started. If current alpaka APIs force some
eager preparation, keep externally visible queue submission on `start()`.

Preserve:

```text
same queue dependency           -> queue FIFO/no host wait
different queue, same device    -> native device wait/event where supported
cross backend unsupported       -> host-visible completion then start consumer
host/MPI completion             -> start after host-visible completion unless
                                   measured native interop exists
```

Do not require accelerator sender completion to execute arbitrary application
continuations on a device completion/progress authority; transfer continuation
execution to the scheduler/environment chosen by composition.

### Completion milestones and execution domains

An accelerator operation may have at least two useful milestones:

```text
submitted/backend-native dependency available
                |
                +---- same backend may consume directly
                |
          host-visible terminal completion
```

Do not collapse these if it forces same-device work through the host. Keep native
milestones backend-local initially.

The P2300 spike should test whether an alpaka execution domain/scheduler or a
smaller sender transformation can preserve native queue/event dependencies across
sender composition without putting backend types in `caravan::core`.

### Future SYCL and Kokkos support

Do not add speculative backends. A SYCL adapter should be able to retain
`sycl::event`-style dependencies where appropriate; a Kokkos adapter may expose
coarser capabilities. Capability differences are acceptable.

---

## Task-runtime interoperability

Task runtimes are independent execution choices. Caravan must permit integration
with Taskflow, HPX, oneTBB, OpenMP tasks, or a `std::execution` implementation
without requiring them to execute a Caravan-specific task object.

Prefer direct sender/scheduler interoperability when a runtime provides it.
Otherwise keep adapters local:

```text
external runtime sender/task
        |
        +---- completion bridge if needed
        |
Caravan MPI/alpaka sender
        |
        +---- continues_on(external scheduler)
```

Do not treat MPI/device progress authorities as public schedulers merely because
they execute backend code.

---

## PMacc run-loop and async-scope model

PMacc may use two explicit application-level control objects during migration:

```text
PmaccRunLoop / caravan::RunLoop
    manually driven single-thread host/control queue
    owns queued continuation storage and blocking drive operations

RunLoopScheduler
    cheap copyable scheduling handle obtained from the loop
    consumed by continuesOn and other placement algorithms

PmaccAsyncScope
    owns dynamically spawned application operations
    prevents stage/shutdown completion until spawned work is quiescent
```

If a supported P2300/standard implementation provides usable `run_loop` and
counting-scope facilities, prefer them directly. Otherwise implement only the
small semantic subset required for migration and keep the replacement boundary
clear.

Neither the loop, its scheduler, nor the async scope is a global Caravan singleton.
The run loop does not own dependency state or every native operation. The async
scope does not decide where work runs. Backends remain independently capable of
native progress.

Blocking PMacc boundaries should prefer a scope/run-loop-aware wait that continues
to run eligible host continuations/progress instead of recreating
`Manager::waitForFinished()` task scanning. `AsyncScope::join()` explicitly closes
the scope, and its owner must provide progress until the join Event is terminal.
Raw scope destruction performs no hidden progress and terminates if the scope is
unjoined or non-quiescent. `pmacc::async::Context` performs that progress-aware
join before destroying its scope; its `wait(Event)` installs a control-loop wakeup
so externally completed Events cannot race with entry into a blocking `runOne()`.

---

## Optional resource-access dependency layer

Resource-aware dependency inference is explicitly optional.

A future target may provide a model such as:

```cpp
auto e = resources.submit(
    kernel_sender(),
    read(fieldE),
    write(fieldB));
```

The resource tracker maintains logical resource identities and access leases,
infers conflicting predecessor operations, and composes/starts the supplied sender
only after those predecessors permit it. It does not execute kernels/MPI itself
and does not own a general worker pool.

The minimum initial access model should be read/write unless a real application
requires more. Access leases remain logically active until the associated
operation completes. The resource control state may need retained identity/state,
but the underlying application allocation is not thereby Caravan-owned.

Do not infer accesses from arbitrary pointers. Declarations are explicit. Debug
hazard diagnostics may be added independently of release-build dependency
inference.

The resource layer must be removable: explicit sender composition remains a fully
supported Caravan use mode.

---

## Standard execution / P2300 direction

P2300 alignment is now an architectural rule, while direct dependency on a
particular implementation remains optional during migration.

### Sender-first backend APIs

MPI and accelerator primitive operations should be sender-like and lazy. Eager
`Event`/`Future` APIs are wrappers/bridges for imperative and runtime-dynamic code,
not the primary backend abstraction.

### Async scopes and dynamic spawning

Use standard-style async scopes for dynamic work such as communication branches or
particle chunk loops when toolchain support permits. Migration scope semantics
should map mechanically to `counting_scope`/`simple_counting_scope`-style
facilities or their eventual standard equivalents.

Use spawn/spawn-future/ensure-started-style boundaries when eager operation is
required. Caravan `Event` may serve as the type-erased eager result of such a
spawn during migration.

### Run loops and execution transfer

Use the scheduler obtained from a manually driven `RunLoop` for the PMacc
single-thread control plane. `continuesOn` accepts the scheduler, not the loop.
Model execution-resource transitions explicitly (`starts_on`/`continues_on`
semantics) so native completion on MPI/device authorities does not accidentally
execute application code there.

### Runtime dynamic boundaries

Retain `Event`, dynamic `joinAll`, or other deliberate type-erasure firebreaks
where fully typed sender expressions would cause awkward runtime storage or
unacceptable compile-time/type complexity. Such boundaries are compatible with a
sender-first architecture.

### Long-term outcomes

1. Current custom core implements P2300-shaped migration semantics and a tested
   thin stdexec adapter; its senders are not direct standard sender models.
2. Standard/P2300 composition replaces most `Flow`/continuation code while keeping
   Caravan MPI/alpaka implementations.
3. If standard facilities satisfy eager spawning, scopes, waits, value/error/stopped
   channels, and dynamic composition needs, shrink Caravan core to backend and
   interoperability utilities plus optional resource-dependency support.

No migration step may require rewriting native MPI progress or accelerator
integration merely because the composition syntax changes.

---

## Buffer, resource, and ownership model

PMacc owns high-level buffer/allocation lifetime policy. Caravan core does not
provide a mandatory generic `KeepAlive` ownership system.

An async PMacc view may be borrowed:

```cpp
struct BufferView
{
    void* data;
    Extents extents;
    // lifetime guaranteed by enclosing PMacc scope/object
};
```

or explicitly owning/retaining at the PMacc/backend boundary:

```cpp
struct OwnedBufferView
{
    AllocationHandle allocation;   // PMacc/backend-specific ownership handle
    void* data;
    Extents extents;
};
```

A particular sender operation captures whatever ownership handle it requires in
its operation state. If the storage is borrowed, the API documents the required
lifetime. Backend-affine allocations/resources are finally released through the
backend authority required by that resource.

Consequences:

- MPI sends/receives cannot access storage after the operation reports terminal;
- callers may choose borrowed storage when an enclosing scope already guarantees
  lifetime;
- operations may explicitly capture allocations when they need independent
  lifetime;
- destruction does not wait on a hidden global frontier;
- resource dependency tracking, if enabled, tracks access hazards separately from
  ownership;
- GPU-aware MPI dependencies remain explicit;
- buffer accessors no longer mutate a hidden global transaction.

---

## Communication composition

Communication remains ordinary explicit composition; it is not represented by
hand-written polling task state machines.

### Host-staged send

```text
data ready
    -> pack/copy into contiguous device buffer
    -> device-to-host copy
    -> native nonblocking MPI send
    -> send completion
```

### GPU-aware send

```text
data ready
    -> pack if required
    -> satisfy GPU->MPI dependency using best available interop
       (host completion fallback initially)
    -> native nonblocking MPI send of device pointer
    -> send completion
```

### Receive

```text
destination lifetime contract satisfied + explicit dependency ready
    -> post native MPI receive as early as correctness permits
    -> immutable receive metadata/count
    -> host-to-device transfer if required
    -> unpack/insert
    -> completion
```

### Field exchange

```text
one branch per direction
    send:    pack -> send
    receive: receive -> unpack/insert
joinAll(direction tails)
```

Core computation can proceed in parallel and joins communication before border
work that depends on it.

### Particle exchange

Keep the dynamic chunk loop, expressed with explicit continuation/coroutine/
standard-execution composition rather than enum-driven polling task objects:

```text
pack chunk
 -> obtain count
 -> send
 -> repeat if full
```

and:

```text
receive chunk
 -> insert
 -> repeat if full
 -> fill border gaps after all receive directions are terminal
```

The initial migration may use explicit continuations. A coroutine or P2300
composition layer must use the same backend operations rather than introducing
another progress engine.

---

## Error handling and shutdown

Operations become `ready`, `failed`, or `stopped`.

Default sender-composition failure/stopped propagation prevents dependent operations
from starting unless an operation explicitly handles that terminal state.

A quiescent `joinAll()` waits for every branch to become terminal before
publishing the retained failure/stopped result.

Rules:

- no exception escapes a backend progress/owner thread;
- blocking result access reports stored errors;
- blocking on work that requires the calling backend's own progress authority is
  forbidden/diagnosed;
- shutting down a backend context rejects new submissions to that context;
- each backend context resolves its queued-but-not-started operations as
  failed/stopped according to its documented shutdown policy;
- active native operations are allowed to reach a safe terminal state unless the
  backend has a supported cancellation mechanism;
- backend-native resources are destroyed only after represented operations are
  resource-safe terminal, and destruction occurs through the required backend
  authority;
- MPI lifecycle cleanup occurs in the context required by the selected MPI
  policy;
- arbitrary blocking native calls already entered are not promised cancellable;
- fatal MPI/process failures have a documented process-termination path.

Prompt fatal-error notification, if required by PIConGPU, is separate from the
quiescent completion object.

---

## PMacc architectural boundary

PMacc should become the first major consumer of Caravan, not part of Caravan's
implementation.

PMacc owns:

- process/application policy selecting the dedicated MPI configuration;
- an explicit application async scope for dynamically spawned PMacc work;
- a selected single-thread `run_loop`-style control scheduler when useful;
- topology representation exposed to simulation code;
- communication direction and tag mapping;
- buffer/allocation ownership and borrowed-view policy;
- the decision whether to enable/use an optional resource dependency tracker;
- field and particle operation composition;
- signals and simulation-stage composition;
- decisions about which CPU/task runtime, if any, to use.

PMacc must not:

- reimplement generic MPI request progress;
- expose a global transaction stack;
- require Caravan-specific general tasks;
- hide dependency creation in buffer accessors;
- make the PMacc run loop a Caravan-global manager;
- rely on arbitrary application callbacks executing on MPI/device progress
  authorities;
- assume Caravan owns application resources merely because an async operation uses
  them;
- directly mutate active backend-native state.

PIConGPU is ported only after the PMacc migration gate passes.

---

# Migration plan

## Hard migration boundary

Until the PMacc gate passes:

- modify Caravan, PMacc, PMacc tests, and the selected PMacc examples;
- do not migrate `include/picongpu` or `share/picongpu` source;
- PIConGPU is allowed to stop building as legacy PMacc interfaces are removed;
- do not retain old interfaces solely for untouched PIConGPU code;
- keep one implementation of each runtime/progress mechanism; compatibility
  adapters are temporary and deleted after their last PMacc user is migrated.

PIConGPU migration begins only after PMacc and its examples run without the
legacy event system.

---

## Phase 0: Inventory and baselines

**Current state:** implementation baseline largely complete; target-GPU
measurements still required before the PMacc exit gate.

1. Inventory every direct MPI call in PMacc, PMacc examples, and PMacc tests.
2. Classify calls as bootstrap/lifecycle, topology/resource, request-based,
   immediate invocation, collective, signal, shutdown, error, or third-party
   MPI use.
3. Inventory every transaction, `EventTask`, manager wait, observer, task ID, and
   custom polling state machine in PMacc and target examples.
4. Record CPU-serial and CUDA compile baselines.
5. Record an untouched PIConGPU build/behavior/performance reference without
   modifying PIConGPU source.
6. Perform a read-only PIConGPU requirements inventory now: identify direct MPI,
   legacy event-system use, custom async state machines, MPI-enabled third-party
   libraries/plugins, lifecycle assumptions, and unusual communicator/collective
   patterns. Do not modify or migrate PIConGPU source; use the inventory only to
   prevent Caravan/PMacc design choices that would make Phase 8 impossible.
7. Add focused regression tests for device ordering, fork/join halo exchange,
   host-staged and GPU-aware MPI, field exchange, particle multi-chunk exchange,
   signals, and shutdown with outstanding work.
8. Record reproducible output for `gameOfLife2D` and `heatEquation2D`, including
   multi-rank paths.
9. Benchmark submission cost, manager CPU cost, MPI ping-pong, halo overlap, and
   example runtimes.
10. Before the Phase 7 PMacc gate, collect deferred target-GPU and GPU-aware MPI
   baselines from the recorded baseline revision.

**Exit criterion:** hardware-independent behavior baseline and inventories are
recorded; deferred target measurements are explicitly tracked.

---

## Phase 1: Completion core

**Current state:** implemented and tested; preserve behavior while adjusting the
architecture in Phase 2.

1. Maintain exactly-once completion state.
2. Maintain thread-safe continuation registration/completion races.
3. Maintain flat runtime-sized join implementation.
4. Maintain failure propagation and blocking-wait guards.
5. Maintain deterministic inline/fake execution support for tests.
6. Keep allocation/state representation simple until profiling justifies custom
   ownership or a slab.

**Exit criterion:** current completion-core tests and ThreadSanitizer coverage
continue to pass.

---

## Phase 2: Migrate the existing Caravan implementation to the sender-oriented library architecture

This phase occurs before adding more PMacc functionality. Reuse the working
completion/MPI code, but change the conceptual/API boundaries so new work aligns
with P2300 semantics and the clarified Caravan scope.

**Current state:** the hardware-independent architecture work is implemented and
covered by core, MPI, alpaka, and alpaka-to-MPI composition tests. The remaining
exit-gate work requires an accelerator environment: run the representative chain
on a target accelerator and validate HIP translation. The optional resource tracker
is deliberately not implemented; the boundary specified in 2.10 is sufficient
until a measured PMacc use case justifies Phase 11.

### 2.1 Complete the minimum typed sender vocabulary

1. Define one coherent completion-signature representation for value, error, and
   stopped channels, and use it to define the sender contract checked by generic
   algorithms and backend senders.
2. Implement typed `then`: predecessor values are passed to the callable and the
   callable's void or non-void result determines the successor value signature.
3. Generalize `letValue` to pass predecessor values to a sender-returning factory;
   retain the successor operation state without virtual/type-erased storage.
4. Implement fixed-arity typed sender `whenAll`; keep runtime-sized Event
   `whenAll(span<Event>)` as the separate eager quiescence operation.
5. Test value, void, error, stopped, laziness, and operation-state lifetime for
   each algorithm.
6. Add no broader P2300 query/environment/customization machinery until
   `continuesOn` or a real backend demonstrates the need.
7. Ensure backend APIs do not consume `Flow` internals; keep `Flow` optional and
   layered above sender/Event bridges.

### 2.2 Make the run loop scheduler-shaped

1. Keep `RunLoop` manually driven and single-threaded.
2. Split scheduling from driving: expose a cheap copyable `RunLoopScheduler`
   referring to the loop's queue/state.
3. Make `continuesOn` consume the scheduler and schedule through it; do not pass
   `RunLoop` as an executor.
4. Test scheduler copies, finish/post races, continuation placement, and manual
   `run`/`runReady` driving.
5. Use standard `run_loop` naming and semantics where practical; do not introduce a
   generic Caravan executor hierarchy.

### 2.3 Reposition `Event`/`Future` as eager bridges

1. Preserve exactly-once terminal state and thread-safe registration.
2. Preserve flat runtime-sized quiescent joins.
3. Keep `Event` useful for already-started work, runtime containers, and imperative
   PMacc migration boundaries.
4. Do not make new backend primitives require predecessor `Event` parameters.
5. Provide/prototype `Event -> sender` and sender -> spawned `Event` bridges.
6. Keep `Future<T>` as an eager shared-result migration boundary, not a competing
   general asynchronous value model.

### 2.4 Narrow lifetime responsibility to operation state and explicit captures

1. Caravan core owns/retains only its own operation/completion state.
2. Remove the architectural requirement for generic core `KeepAlive`/`LifetimeSet`
   ownership of arbitrary application resources. These can live in the optional caravan::resource layer.
3. Define borrowed-resource preconditions for primitive operations.
4. Allow particular operation state to explicitly capture PMacc/backend ownership
   handles when independent storage lifetime is required.
5. Keep MPI communicators/requests, accelerator events/queues, and other
   backend-affine native resources owned and destroyed by their backend authority.
6. Ensure terminal completion means native work can no longer access operation-owned
   or explicitly captured state.

### 2.5 Reframe and layer MPI around lazy sender operations

1. **Implemented:** rename `MpiExecutor` to `MpiContext` and present it as the MPI
   backend/runtime authority, not a scheduler; retain the working dedicated
   worker/progress implementation.
2. Preserve the current dedicated worker policy behavior and make the native
   nonblocking request engine the central implementation.
3. **Implemented:** put normal typed sender factories (`mpi::send`, `receive`,
   reductions, gathers, barrier, and peers) in the normal public MPI header.
4. **Implemented:** keep generic `request`, `invoke`, `invokeBlocking`,
   `NativeMpiContext`, and raw request/lifetime transfer in the native extension
   header.
5. Ensure every primitive sender's `start()` performs MPI initiation in the valid
   authority and convenience operations remain thin factories over that path.
6. **Implemented:** remove predecessor `Event` parameters from the MPI engine and
   its internal submission queue. Sender composition decides when an MPI operation
   is started.
7. **Implemented:** let temporary compatibility wrappers subscribe to/adapt
   predecessor Events above the backend; `submitAfter(Event, ...)` is not retained
   as the engine model.
8. Preserve per-communicator collective initiation ordering.
9. Avoid Caravan replicas of MPI types unless they express a Caravan-specific
   safety property.

### 2.6 Separate MPI completion from continuation execution

1. Isolate request initiation/storage/completion decoding from worker-loop policy.
2. Isolate MPI init/finalize ownership into the dedicated policy.
3. Preserve FUNNELED production behavior.
4. Ensure the MPI worker never becomes a public scheduler for arbitrary
   continuations.
5. Prototype an explicit execution-transfer path from MPI completion to a PMacc
   run-loop/external scheduler (`continues_on`-style semantics).

### 2.7 Build the alpaka/device sender prototype

**Current state:** the lazy borrowed-queue batch sender, kernel/copy/fill CPU test,
explicit capture-lifetime test, same-queue FIFO path, run-loop completion transfer,
and CUDA translation-unit compile checks for both the backend and cross-backend
chain are implemented. Target accelerator runtime and HIP translation validation
remain open.

Implement this prototype before deepening generic abstractions or migrating PMacc
polling chains. MPI validates native progress; alpaka must validate the genuinely
different accelerator half of the architecture.

1. Implement one real lazy sender over a caller-supplied alpaka queue, plus only the
   minimum copy/fill/kernel shape needed to compose a representative chain.
2. Validate operation-state and borrowed/captured resource lifetime through native
   completion.
3. Preserve same-queue ordering and backend-native dependency information without a
   host wait; use host-visible completion at the MPI boundary.
4. Validate completion placement so arbitrary continuations do not execute on a
   backend completion authority.
5. Compile the representative chain with the supported CPU and CUDA/HIP translation
   paths and record type/compile-time constraints.
6. Use the prototype to decide the smallest sender environment/domain support
   actually required; do not design it from MPI alone.

### 2.8 Preserve accelerator-native dependency information

1. Ensure `caravan::core` does not erase backend-local dependency information.
2. Do not add a generic cross-backend `NativeDependency` protocol yet.
3. Use host-visible completion at independent backend boundaries initially.
4. Revisit generic native interop only with a real second path.

### 2.9 P2300 feasibility and semantic-alignment spike

This is now a design gate, not merely a future adapter experiment.

**Current state:** a test composes a lazy alpaka batch into MPI send/receive and an
explicit `RunLoopScheduler` host continuation, spawned through `AsyncScope`. It
checks lazy start, the host-visible accelerator/MPI boundary, continuation thread
placement, eager Event bridging, CPU runtime behavior, and CUDA translation. HIP
translation and target-accelerator runtime validation remain open.

1. Compose the Phase 2 alpaka prototype into one sender chain:
   accelerator operation -> MPI request -> host continuation.
2. Verify lazy construction/start semantics.
3. Verify MPI completion can feed a receiver without exposing the MPI thread as an
   application scheduler.
4. Prototype explicit `continues_on` transfer to a `RunLoopScheduler` owned by a
   manually driven PMacc-style `RunLoop`.
5. Prototype a small async scope with spawn/join semantics comparable to
   `counting_scope`/`simple_counting_scope` or the available implementation.
6. Test an eager/type-erased bridge from a spawned sender to `Event`.
7. Check CUDA/HIP translation-unit constraints, compiler support, compile-time/type
   complexity, failure/stopped mapping, and runtime-dynamic join gaps.
8. Record concrete blockers if the ecosystem is unsuitable; preserve the same
   semantics in the custom migration layer instead of inventing incompatible
   concepts.

### 2.10 Define the optional resource-layer boundary

1. Specify a minimal pluggable interface around stable resource identity and
   explicit `read`/`write` access declarations.
2. Specify access-lease lifetime until operation completion.
3. Specify that inferred dependencies are lowered to ordinary sender/Event
   composition.
4. Specify that underlying application storage ownership is outside the resource
   tracker.
5. Do not require or fully implement the resource tracker for the PMacc migration
   unless a representative use case shows clear value.

### 2.11 Preserve current PMacc integration during refactor

1. Update temporary PMacc attachment and legacy MPI adapters to the revised MPI
   context/API.
2. Preserve currently migrated signal, reduction, gather, barrier, communicator,
   topology, and point-to-point functionality.
3. Do not expand the migrated PMacc feature set until this architecture passes the
   existing tests.

**Exit criterion:** existing Caravan/PMacc functionality still works; the typed
sender vocabulary and coherent completion signatures are implemented; `RunLoop`
and `RunLoopScheduler` have distinct driving/scheduling roles; normal MPI sender
operations are the obvious public API; the renamed MPI context retains its worker
as a progress/lifecycle authority with no backend-level Event predecessor model;
`Event` is explicitly an eager bridge; core no longer promises generic
application-resource ownership; MPI completion is separable from continuation
placement; one real alpaka sender validates lifetime, native dependency, completion
placement, and supported device compilation; an async-scope prototype is
documented; resource dependency inference has a pluggable non-core boundary; no
global Caravan supervisor or general task/scheduler hierarchy has been introduced.

---

## Phase 3: Complete the dedicated-thread MPI backend and PMacc MPI migration

**Current state:** implemented for the PMacc migration scope. PMacc startup,
topology, point-to-point operations, signals, barriers, reductions, and gathers
use the managed Caravan MPI context. Native MPI calls are confined to
`MPIReduce`'s generic request initiation hook, and a PMacc CI check rejects calls
outside that integration boundary. No PMacc-scoped MPI-enabled third-party call
requires `invokeBlocking`; PIConGPU plugin/library use remains deferred to Phase 8.

Begin this phase only after the Phase 2 alpaka sender prototype has exercised the
shared sender model across both backend shapes.

1. **Implemented:** finish the dedicated-thread submission queue and batched
   active-request progress path without reintroducing predecessor storage in the
   MPI engine.
2. **Implemented:** complete point-to-point operations, receive status/count
   metadata, required collectives, barriers, communicator creation/destruction,
   and topology setup through the generic MPI mechanisms.
3. **Implemented:** route all remaining PMacc direct MPI operations and target PMacc
   examples through `caravan::mpi` or narrowly scoped generic native invocation.
4. **Implemented:** route PMacc signal operations through the same MPI context.
5. **Implemented for the PMacc scope:** inventory PMacc-relevant MPI-enabled
   third-party calls; none require `invokeBlocking()`. PIConGPU plugin/library use
   remains a Phase 8 concern.
6. **Implemented:** preserve and test per-communicator collective initiation
   ordering for managed collective helpers; dependency-ready later collectives
   must not pass earlier submitted collectives on the same communicator.
7. **Implemented:** preserve early receive posting where explicit dependencies and
   the destination lifetime contract permit.
8. **Implemented:** add a PMacc-scoped CI rule/allowlist rejecting direct MPI calls
   outside the approved MPI integration layer during this migration stage.
9. **Implemented:** verify progress continues while the process application thread
   sleeps or does CPU work.
10. **Implemented:** verify startup/shutdown, failure propagation, communicator
   lifetime, and queued/in-flight request handling.

**Exit criterion:** no PMacc example, task, helper, or simulation thread directly
calls MPI; the selected production configuration uses the dedicated FUNNELED
worker and progresses independently of application polling.

---

## Phase 4: Alpaka accelerator backend

**Current state:** implemented and CPU-runtime tested. CUDA translation of the
backend, primitive chain, and alpaka-to-MPI chain is validated. Target-accelerator
runtime and HIP translation remain the hardware-dependent Phase 2 validation gate.

The implementation expands the Phase 2 prototype rather than adding a second
abstraction.

1. **Implemented:** `caravan::alpaka` is the first accelerator backend.
2. **Implemented:** kernel/copy/fill/size primitives are sender-oriented and lazy;
   native queue submission occurs on operation start.
3. **Implemented:** caller-supplied queues are borrowed and must outlive the
   operation; submit callables and primitive arguments are retained by value, while
   storage referenced by non-owning alpaka views remains borrowed.
4. **Implemented:** Caravan adds no submission thread or queue.
5. **Implemented:** same-queue stages lower directly to queue FIFO.
6. **Implemented for supported same-device queues:** queue changes record an alpaka
   event and insert a native queue wait.
7. **Implemented:** only the final queue host callback publishes host-visible
   completion; native dependency availability precedes it.
8. **Implemented:** native events remain private to `caravan::alpaka`; MPI and
   generic sender edges continue to consume host-visible completion.
9. **Implemented:** the small `caravan::alpaka::then` domain transformation merges
   typed alpaka senders into one native FIFO/event chain without adding a scheduler
   hierarchy.
10. **Implemented:** CPU and GPU use the same explicit alpaka queue-host-callback
    completion policy; no GPU-specific polling is assumed.
11. **Implemented and tested:** explicit `continuesOn` transfer places application
    continuations on the selected run-loop scheduler.
12. **Implemented and tested:** supported caller-supplied queues accept concurrent
    starts without Caravan serialization.

**Exit criterion met for the hardware-independent Phase 4 scope:** PMacc can start
accelerator operations through sender-like alpaka primitives; native accelerator-
only chains do not host-wait between stages; no legacy Manager is required for
these completion paths.

---

## Phase 5: Explicit PMacc dependencies, async scope, run loop, and ownership migration

**Current state:** implemented for the representative PMacc path. The
`gameOfLife2D` core/border step uses explicit lazy kernel senders, a PMacc-owned
async scope/run loop, and explicit allocation retention. Its communication call
remains the deliberate legacy `EventTask` boundary for Phase 6.

1. **Implemented:** PMacc's explicit `async::Context` owns dynamically spawned
   migration work in a Caravan `AsyncScope`.
2. **Implemented:** the context owns and drives a `RunLoop` and transfers sender
   completion through its `RunLoopScheduler`.
3. **Implemented with the replaceable custom subset:** the supported toolchain
   still uses Caravan's P2300-shaped scope/run-loop semantics.
4. **Deliberately not introduced:** `Flow` is unnecessary on the representative
   path; direct sender composition is smaller and remains the baseline.
5. **Implemented:** the representative `gameOfLife2D` local core/border step no
   longer reads or mutates global transaction state.
6. **Implemented:** buffer/view/pointer/size accessors no longer call
   `eventSystem::startOperation`; mutating legacy operations and destruction
   safety remain until their later call-site ports.
7. **Implemented:** PMacc exposes borrowed alpaka views and explicit `OwnedView` /
   `Retained` allocation capture for asynchronous operation state.
8. **Implemented and tested:** explicit kernel, copy, byte-fill, and size-transfer
   sender call sites use the alpaka backend; CPU runtime and CUDA translation pass.
9. **Implemented on migrated paths:** `async::Context::wait` drives the local run
   loop; the remaining Manager wait in `gameOfLife2D` is confined to its unmigrated
   Phase 6 communication boundary.
10. **Implemented where practical:** pending waits from an executor continuation
    are diagnosed; owned-view lifetime is tested by destroying PMacc buffer
    wrappers before native completion.
11. **Preserved:** resource-access dependency inference remains disabled; all new
    dependencies are explicit composition.
12. **Not needed:** no recurring forgotten-join defect was observed, so no
    speculative fork/join helper was added.

**Exit criterion met:** a representative PMacc example uses explicit local sender
composition, an explicit async scope, and selected run-loop execution; application
resource ownership is explicit rather than a generic Caravan `KeepAlive` rule; no
buffer accessor mutates global scheduling state and no Caravan-global Manager has
been introduced.

---

## Phase 6: PMacc communication and target examples

**Current state:** M1 and M2 are implemented. `Exchange` and `GridBuffer` expose
explicit send/receive branches with flat Event joins, and both target examples
compose halo communication and compute work without an `EventTask`/Manager
boundary. `heatEquation2D` also composes residual copy/reset and typed MPI
reduction, while its gather path uses an explicit completed-input boundary. The
four-rank CPU residual regression and CUDA translation pass; target-GPU runtime
validation and the broader M3 communication migration remain.

1. Port `Exchange` send and receive to explicit operation chains.
2. Preserve host staging, double buffering, and GPU-aware MPI.
3. Use the generic MPI request API beneath all send/receive convenience code.
4. Return immutable receive metadata/counts.
5. Port `GridBuffer::asyncCommunication` to explicit per-direction branches and
   one flat runtime-sized join.
6. Validate buffer-reuse dependencies across time steps and directions.
7. Port `gameOfLife2D` to explicit communication/compute composition.
8. Port `heatEquation2D`, including gather and reduction paths.
9. Run CPU paths, multi-rank configurations, and available CUDA compile/runtime
   validation.
10. Compare behavior against Phase 0 outputs.

**Exit criterion:** both target PMacc examples use Caravan backends and explicit sender/Event
composition and pass recorded behavior checks. Legacy send/receive task classes
remain only if a still-unmigrated Phase 7 PMacc path requires them.

---

## Phase 7: Complete PMacc and pass the PIConGPU entry gate

1. Replace remaining field parent send/receive polling tasks with explicit
   direction composition.
2. Replace particle send/receive enum/polling tasks with continuation-based,
   coroutine-based, or standard-execution-compatible chunk loops using the same
   Caravan backend operations.
3. Join all receive directions before field insertion or particle gap filling.
4. Add exact-capacity, empty, partial, and multi-chunk stress tests.
5. Port remaining reductions, gathers, signals, examples, tests, and helper
   operations.
6. Remove `FieldFactory`, `ParticleFactory`, and their task classes as soon as
   migrated replacements pass.
7. Delete Manager, transactions, legacy task IDs, observers, logical-and tasks,
   event pumping, legacy MPI/device task classes, and PMacc migration adapters
   after their last use.
8. Run complete PMacc unit/integration tests and target examples.
9. Collect the deferred target GPU/GPU-aware MPI baseline if not already done.
10. Compare behavior and performance with Phase 0.
11. Verify Caravan core and backends still contain no PMacc/PIConGPU headers.

**Exit criterion / PIConGPU entry gate:** PMacc and target examples use the new
Caravan architecture without global transaction/manager task scanning; PMacc
uses explicit scope/run-loop policy where required; direct MPI is
restricted to the integration layer; the production MPI policy progresses on
its dedicated worker; accelerator work uses the alpaka backend; legacy event
system code required by PMacc has been deleted; and the agreed performance gates
are met or deviations are explicitly understood and accepted.

Only then may PIConGPU source migration begin.

---

## Phase 8: PIConGPU inventory and migration

1. Refresh the read-only PIConGPU requirements inventory from Phase 0 and resolve
   any changes since it was recorded.
2. Classify any newly discovered MPI use into generic request-based, immediate
   MPI-context, blocking third-party, lifecycle/error, managed collective ordering,
   or application composition.
3. Use the untouched Phase 0 PIConGPU baseline as the reference; do not restore
   removed PMacc compatibility APIs merely to reproduce it.
4. Port field and particle code to the already-tested PMacc asynchronous
   operation APIs.
5. Port reductions, gathers, signals, checkpoints, plugins, diagnostics,
   examples, and tests.
6. Route MPI-enabled external-library calls through the documented Caravan MPI
   context or replace them with native nonblocking integration when available.
7. Keep PIConGPU task-runtime choices independent of Caravan. Do not introduce a
   Caravan task hierarchy during the port.
8. Remove raw MPI calling capability from ordinary PIConGPU simulation/plugin
   code; native handles may exist only where needed to submit through the MPI
   integration boundary.
9. Enable the final CI rule forbidding MPI calls outside the approved integration
   layer.
10. Validate CPU, threaded CPU, CUDA, multi-rank, checkpoint, plugin, field, and
    particle configurations according to available CI/hardware.

**Exit criterion:** all supported PMacc/PIConGPU paths use explicit asynchronous
composition and Caravan backend integration; PIConGPU contains no dependency on
the removed event runtime.

---

## Phase 9: Cleanup, documentation, and library boundary validation

1. Remove all migration-only wrappers and compatibility aliases.
2. Remove stale legacy includes, tests, documentation, build rules, and dead
   state-machine code.
3. Document Caravan independently from PMacc:
   - completion semantics;
   - operation-state lifetime, borrowed/owned application-resource contracts, and backend-affine resources;
   - backend contract;
   - backend-local native dependency handling and the criteria for introducing
     future cross-backend interop;
   - MPI generic request API;
   - dedicated-thread MPI policy;
   - accelerator backend rules;
   - sender/P2300 semantics, async scopes, run-loop integration, and standard-execution direction;
   - optional resource-access dependency inference boundary.
4. Document PMacc's selected production policies separately from Caravan's
   generic capabilities.
5. Verify shutdown does not depend on singleton destruction order.
6. Verify one can unit-test `caravan::core` without MPI/alpaka and test
   `caravan::mpi` without PMacc.
7. Review whether target boundaries justify splitting packages/CMake targets or
   external extraction. Do not extract solely for appearance.

**Exit criterion:** no legacy event-system or migration scaffold remains and the
Caravan/PMacc boundary matches the library vision.

---

## Phase 10: Standard execution and task-runtime interoperability

This phase is optional for the PMacc migration, but the API constraints were
already established in Phase 2.

1. Revisit the Phase 2 sender/run-loop/scope spike against the then-current C++
   standard execution implementation ecosystem and supported compiler/toolchain
   matrix.
2. Replace custom sender-like primitives with direct standard/P2300 models where
   this is mechanical and does not harm supported device compilation.
3. Implement/retain the smallest useful `Event`/`Future` <-> sender bridges.
4. Express one representative chain:

   ```text
   accelerator pack -> MPI send -> continues_on(PMacc scheduler) -> CPU continuation
   ```

   with both migration composition and standard-execution composition.
5. Replace the migration PMacc run loop with standard/P2300 `run_loop` if practical.
6. Replace the migration async scope with standard/P2300 counting-scope semantics
   if practical.
7. Compare allocations, compile-time/type complexity, CUDA/HIP/SYCL toolchain
   impact, error/stopped behavior, dynamic-join behavior, and runtime overhead.
8. Add task-runtime adapters only for real application needs; prefer existing
   sender/scheduler integration.
9. Keep MPI progress/device native integration unchanged regardless of composition
   syntax.
10. Do not expose MPI/device progress authorities as general schedulers.

**Exit criterion:** Caravan either directly models the useful standard execution
concepts or has documented toolchain/performance reasons for retaining a thin
compatible implementation; no competing Caravan scheduler/task framework has
appeared.

---

## Phase 11: Optional resource-access dependency layer

Implement this phase only if PMacc/PIConGPU or another consumer benefits from
inferred resource dependencies beyond explicit composition.

1. Implement stable logical resource identity/control state independent of the
   underlying application allocation ownership.
2. Implement the minimal `read`/`write` access model and access leases lasting
   until operation completion.
3. Infer writer->reader, writer->writer, and readers->writer dependencies.
4. Lower inferred predecessors to ordinary sender/completion composition; do not
   introduce a second executor or global task scheduler.
5. Support externally supplied/borrowed resources without transferring allocation
   ownership to Caravan.
6. Start with compound logical resources; split subresources such as buffer data
   and device-side size only where semantics and measurements justify it.
7. Keep explicit composition fully supported and test that all backends work
   without the resource layer.
8. Add optional debug hazard diagnostics that can be enabled independently of
   release-build dependency scheduling.
9. Benchmark graph bookkeeping and concurrency gains against explicit PMacc flows.

**Exit criterion:** if implemented, the resource layer is a removable dependency
planner over sender-style operations, not a prerequisite of `caravan::core`, MPI,
or accelerator backends. If no measured use case justifies it, document that and
leave the phase unimplemented.

---

## Phase 12: Additional accelerator backends when required

Add SYCL, Kokkos, or another backend only for a real consumer.

For each backend:

1. implement sender/P2300-compatible operation start/completion semantics;
2. define native resource ownership/borrowing policy;
3. define synchronization/progress authority;
4. define backend-local native dependency capabilities; add cross-backend
   import/export only if a real supported pair requires it;
5. provide host-completion fallback at independent backend boundaries;
6. define continuation execution transfer away from backend progress authorities;
7. do not emulate unsupported fine-grained semantics at disproportionate cost;
8. test cross-domain composition with MPI and at least one CPU composition path;
9. keep backend-specific types out of `caravan::core`.

---

## Phase 13: Profile-driven optimization

Only after correctness and migration:

1. batch producer queue drains and native submissions where measured useful;
2. coalesce accelerator completion events at dependency boundaries;
3. replace shared-state allocation with a slab/intrusive scheme only if profiles
   show material cost;
4. replace mutex-protected queues only if contention is measured;
5. tune MPI polling/yield/affinity policy on target systems;
6. evaluate alternate MPI progress policies only with a concrete use case;
7. tune PMacc run-loop batching/wakeup behavior only from measurements;
8. evaluate multiple accelerator submission authorities only if a single authority
   is a measured bottleneck;
9. evaluate/optimize cross-domain native dependency paths only after a real
   supported interop pair demonstrates the needed abstraction;
10. profile optional resource-dependency bookkeeping separately from backend
    execution overhead;
11. preserve backend independence while optimizing implementation internals.

**Exit criterion:** no unacceptable full-step regression, measurable host/runtime
improvements over the legacy baseline, and no optimization has expanded Caravan
into a general scheduler/runtime or mandatory resource manager.

---

# Testing strategy

## Core tests

- completion before and after continuation registration;
- simultaneous completion/registration races from many threads;
- exactly-once terminal transition;
- exactly-once continuation dispatch;
- zero-, one-, and many-input flat joins;
- runtime-sized join from dynamically built vectors;
- join waits for all inputs after early failure/stopped completion;
- join precedence failed > stopped > ready and no promise of race-defined first
  failure;
- predecessor failure/stopped propagation;
- terminal completion implies represented native work no longer accesses retained
  operation state;
- ready-event fast path/allocation behavior;
- blocking-wait guards;
- operation-state lifetime through start -> terminal completion;
- borrowed-resource lifetime precondition tests and explicit operation capture tests;
- sender construction is lazy and native side effects begin on start;
- completion-signature representation and sender concept checks;
- typed `then` for void and value predecessors/results;
- typed value-forwarding `letValue` without successor type erasure;
- fixed-arity sender `whenAll` value, error, and stopped behavior;
- Event <-> sender eager bridge behavior;
- async-scope spawn/join/quiescence and shutdown behavior;
- `RunLoopScheduler` copies schedule onto their owning manually driven `RunLoop`;
- `continuesOn` consumes a scheduler and places completion on the run-loop thread;
- PMacc run-loop continuation placement and progress-aware waits;
- no global outstanding-operation registry is required by `caravan::core`;
- `Flow` move-only/local semantics if retained;
- forgotten-fork tests/documentation or structured-helper tests;
- adapter tests proving core does not require a backend owner thread or global run loop;
- no arbitrary continuation execution on MPI/device progress authorities;
- ThreadSanitizer coverage where supported.

## Optional resource-layer tests

If the resource layer is implemented:

- read/read overlap is permitted;
- writer->reader, writer->writer, and readers->writer dependencies are inferred;
- access leases remain active until asynchronous completion;
- resource control-state lifetime is independent of application allocation
  ownership;
- explicit composition and resource-derived composition produce equivalent
  correctness;
- compound vs split resource identities behave as documented;
- disabling the resource layer leaves MPI/alpaka/core functionality unchanged.

## MPI tests

Run with at least one, two, and four ranks where applicable:

- dedicated-policy thread ownership assertion for every MPI entry point;
- startup/finalization on the dedicated worker;
- generic one-request submission;
- generic multi-request submission;
- request-start exception cleanup after partial initiation;
- eager and rendezvous-sized send/receive;
- wildcard receive copied status/count;
- bidirectional neighbor exchange;
- nonblocking collectives/barriers;
- repeated communicator/resource creation/destruction;
- progress while application thread sleeps or computes;
- new submissions while requests are active;
- `invokeBlocking()` starts only after dependencies explicitly composed before it;
  it does not wait for unrelated active requests;
- a regression case where an earlier receive requires a later send, proving
  blocking invocation does not manufacture an implicit quiescence deadlock;
- per-communicator managed collective initiation order despite out-of-order
  dependency readiness;
- point-to-point progress is not unnecessarily serialized by the collective lane;
- recursive/invalid MPI-context invocation rejection;
- blocking third-party invocation path with explicit quiescence dependency;
- shutdown with queued and active requests;
- propagated MPI errors where practical;
- no arbitrary application callback execution on the progress thread;
- thin convenience wrappers use the same generic request engine;
- normal typed MPI sender operations are available from the normal public header;
- the MPI engine stores no Event predecessor; compatibility adaptation is above it.

## Accelerator tests

- same-queue ordering;
- cross-queue native waits where supported;
- backend-native dependency availability before host completion;
- alpaka-to-alpaka native chaining remains backend-local;
- accelerator -> MPI and MPI -> accelerator use correct host-completion fallback
  until a concrete native interop path is implemented;
- host-staged and GPU-aware communication paths;
- concurrent submissions using supported caller-supplied queues;
- no extra submission thread/queue is required in the default alpaka adapter;
- owned and borrowed native-resource policy where supported;
- backend-affine native resources are destroyed on the required authority;
- explicitly captured application allocations survive until operation completion;
- borrowed allocation lifetime violations are diagnosed where practical;
- sender creation does not enqueue native work before start;
- sender operation/resource lifetime through native completion;
- representative typed chains compile on supported CPU and CUDA/HIP paths;
- CPU backend behavior without GPU-only polling assumptions.

## PMacc regression tests

These gate PIConGPU source changes:

- `gameOfLife2D` fork/join and multi-rank halo exchange;
- `heatEquation2D` communication, gather, reduction, and output;
- field halo exchange;
- particle exchange with empty, partial, exact-capacity, and multi-chunk data;
- signal/checkpoint barriers;
- buffer reuse across steps/directions;
- shutdown with outstanding PMacc work.

## PIConGPU regression tests

After the PMacc gate:

- field and particle communication;
- reductions/gathers/signals;
- checkpoints and plugin synchronization;
- diagnostics using MPI-enabled third-party libraries;
- CPU serial backend;
- threaded CPU backend/runtime configurations;
- CUDA compile validation on no-GPU machines;
- CUDA runtime tests on suitable hardware/CI;
- supported GPU-aware MPI configurations;
- full-step behavior/performance comparison with baseline.

---

# Performance acceptance criteria

1. The PMacc production MPI policy makes progress while application threads do
   not call Caravan.
2. MPI ping-pong latency and bandwidth remain within 5% of equivalent direct
   nonblocking MPI for representative message sizes unless a measured,
   understood platform-specific reason is accepted.
3. No full PIConGPU simulation-step regression greater than 2% on the primary GPU
   configuration without an understood and accepted cause.
4. Host scheduling/progress overhead of the PMacc run-loop/scope configuration is lower than the legacy PMacc manager for
   representative kernel and exchange counts.
5. No central scan proportional to all outstanding application operations.
6. Runtime-sized join uses one flat completion node rather than a heap-allocated
   binary task tree.
7. Accelerator operations depending only on work in the same backend can be
   enqueued using backend-native dependencies without host-observed completion
   where supported.
8. Independent backend boundaries use correct host completion until a concrete
   native interop implementation is justified; `caravan::core` contains no
   speculative cross-backend event protocol.
9. MPI convenience operations do not create separate request/progress state
   machines.
10. Final PMacc/PIConGPU runtime code is smaller and conceptually simpler than the
    removed event/task hierarchy.
11. Optional standard-execution/task-runtime adapters do not materially regress
    native paths merely by being enabled in the build.
12. Sender-oriented primitive APIs do not introduce a host synchronization boundary
    between same-backend accelerator operations.
13. Optional resource dependency inference, if enabled, has measured value relative
    to its bookkeeping cost and remains absent from backends when disabled.

---

# Main risks and mitigations

## Scope creep into another general task runtime

**Risk:** Caravan accumulates task types, schedulers, worker pools, structured
execution primitives, allocator propagation, and composition algorithms already
provided by `std::execution` or task runtimes.

**Mitigation:** require a concrete heterogeneous-backend/interoperability need
before adding generic execution abstractions. Treat standard execution as the
preferred future generic composition model.

## The current MPI implementation is too coupled to its worker thread

**Risk:** request objects, public API, lifecycle, and completion logic assume one
specific worker implementation, making future integration policies expensive.

**Mitigation:** Phase 2 explicitly separates request semantics from lifecycle and
progress while retaining identical FUNNELED behavior for PMacc.

## Accidentally reimplementing MPI

**Risk:** Caravan grows one wrapper/type/state machine per MPI API.

**Mitigation:** make generic request submission and generic MPI-context invocation
the central primitives. Keep native MPI types in the MPI-specific layer. Add
convenience functions only as thin forwarding/composition helpers.

## Dedicated MPI progress consumes a CPU core

**Risk:** one reserved core per rank can be expensive on some systems.

**Mitigation:** retain the dedicated policy for correctness and overlap first;
provide configurable spin/yield behavior; benchmark affinity/reservation; add an
alternative policy only when a real deployment needs it.

## MPI lifecycle constraints conflict with external runtimes

**Risk:** an externally initialized MPI environment may be incompatible with a
worker making MPI calls under the provided thread-support level.

**Mitigation:** treat lifecycle/thread-support as an explicit policy contract.
The current PMacc policy owns initialization/finalization on the FUNNELED worker.
Future attach modes must validate their MPI threading contract rather than
silently reusing the worker.

## Implicit MPI quiescence creates dependency cycles

**Risk:** a blocking invocation waits for every active request even when
completion of one of those requests requires a later MPI operation to be
initiated, manufacturing a deadlock that was not present in the application.

**Mitigation:** dependencies before `invokeBlocking()` are composed by the caller
outside the primitive. Application-required quiescence is built with explicit
joins; the MPI backend never stores Event predecessors or silently adds a
dependency on all active requests.

## Dependency readiness reorders MPI collectives

**Risk:** a later collective becomes ready before an earlier submitted collective
on the same communicator and is initiated first, violating the application's MPI
collective ordering.

**Mitigation:** managed collective helpers use a per-communicator initiation
sequence/lane. Point-to-point work remains independent. Expert invocation that
executes collectives has an explicit caller-managed or integrated ordering
contract.

## Cross-domain dependencies cause hidden host synchronization

**Risk:** reducing every same-backend accelerator dependency to host-ready
completion destroys queue concurrency and overlap.

**Mitigation:** preserve backend-native dependency information inside each backend.
Use host-visible completion at independent backend boundaries until a measured
second interop path justifies a generic protocol.

## Premature native dependency abstraction becomes a backend-type registry

**Risk:** core gains a speculative type-erased protocol or direct knowledge of
alpaka/SYCL/CUDA/HIP/Kokkos event semantics before their common requirements are
known.

**Mitigation:** define no generic cross-backend native-event abstraction during the
initial migration. Learn from a real second interop pair first; keep concrete
native types in backend targets.

## `Flow::fork()` allows forgotten joins

**Risk:** imperative graph construction can accidentally let a branch escape a
required synchronization point.

**Mitigation:** keep backend-context shutdown independent of `Flow`; let PMacc or
an explicit application scope own whole-application structured lifetime; add a
sealed `parallel()`/`forkJoin()` helper if migration shows recurring mistakes;
plan for eventual standard-execution/scoped composition.

## Operation lifetime and application-resource ownership are conflated

**Risk:** Caravan starts retaining arbitrary application objects by default, adding
reference-counting overhead and obscuring ownership, or borrowed resources die
while native work is still using them.

**Mitigation:** core owns only operation state. Borrowed storage has explicit
preconditions; operations capture application/backend ownership handles only when
needed. Resource dependency tracking does not imply storage ownership. Native
backend resources are destroyed through their required authority.

## PMacc run loop or async scope recreates the PMacc Manager

**Risk:** the migration run loop/scope accumulates global dependency state, task-ID
lookup, native-operation ownership, and backend polling until it becomes a renamed
Manager.

**Mitigation:** the run loop schedules ready host/control continuations only and may
invoke narrow progress hooks; the async scope tracks structured application work
only. Dependencies live in sender composition or the optional resource layer;
backends own native progress/state. Neither object is a Caravan singleton.

## Sender/P2300 template complexity harms PIConGPU toolchains

**Risk:** deeply typed sender expressions increase compile time, diagnostics, or
CUDA/HIP compiler fragility enough to outweigh architectural benefits.

**Mitigation:** verify representative chains in Phase 2; use deliberate eager/type-
erased `Event` boundaries as compile-time firebreaks; keep native backend code
independent of composition syntax; require measured benefit before increasing
sender-expression depth.

## Optional resource layer grows into a mandatory scheduler

**Risk:** resource identities, access tables, and ready queues become coupled to
MPI/alpaka execution so all Caravan users are forced through a RedGrapes-like
runtime.

**Mitigation:** the resource layer only infers predecessor relationships and emits
ordinary async composition. Backends and explicit sender composition must remain
usable with the layer completely absent.

## Task-runtime adapters execute work on the wrong authority

**Risk:** a generic adapter accidentally schedules arbitrary callbacks on MPI
progress or device-owner threads.

**Mitigation:** distinguish native progress/context authority from public
schedulers. Cross a boundary by posting completion to the chosen external
runtime, not by treating every backend authority as a scheduler.

## Backend capability differences complicate abstraction

**Risk:** SYCL, alpaka, Kokkos, MPI, and CPU task runtimes offer different
fine-grained synchronization semantics.

**Mitigation:** define a minimal portable completion contract plus optional
capabilities. Do not require unsupported native features to be emulated.

---

# Definition of done

The PMacc/PIConGPU migration is complete when:

- no global PMacc event manager, transaction stack, observer system, task-ID lookup,
  or polling task hierarchy remains;
- the custom sender layer has one coherent sender concept/completion-signature
  representation plus typed `then`, value-forwarding `letValue`, and fixed-arity
  sender `whenAll`;
- new primitive Caravan MPI/alpaka operations have sender/P2300-compatible lazy
  start/completion semantics, even if the implementation is still custom;
- dependencies are explicit composition relationships, or are produced by an
  optional pluggable resource layer rather than hidden buffer-access side effects;
- buffer accessors no longer mutate scheduling state;
- Caravan core owns async operation-state lifetime only; application resource
  ownership is explicit/borrowed at PMacc/backend boundaries;
- terminal completion means represented native work no longer accesses
  operation-owned or explicitly retained state;
- PMacc has an explicit async scope for dynamic structured work and may use a
  manually driven `RunLoop` through its cheap `RunLoopScheduler` without either
  becoming a Caravan-global Manager;
- native completion on MPI/device authorities does not accidentally execute
  arbitrary application continuations there; execution transfer is explicit;
- the MPI backend authority is exposed as `MpiContext`, not as a generic executor;
  PMacc selects its dedicated FUNNELED policy, but Caravan architecture does not
  require that policy universally;
- normal typed MPI sender operations live in the normal public MPI layer, while
  request/invoke/native-context facilities remain an explicit extension layer;
- MPI request progress uses one generic native request engine rather than one
  implementation per MPI operation, and that engine stores no Event predecessor;
- blocking MPI-context operations have only explicitly composed dependencies and
  never manufacture implicit global quiescence;
- managed collectives preserve per-communicator initiation order independent of
  dependency-ready timing;
- native MPI types are not duplicated by an unnecessary full Caravan type system;
- no ordinary PMacc/PIConGPU thread calls MPI outside the approved integration
  boundary;
- MPI progresses independently of simulation-thread polling in the production
  configuration;
- accelerator support is provided through sender-oriented alpaka adapters, not a
  universal Caravan device-owner model;
- same-backend accelerator dependency chains avoid host waits where native support
  exists; independent backend boundaries have a correct host-completion path;
- `Event`/`Future` are eager/type-erased bridges rather than mandatory primitive
  backend APIs, and `Flow`, if retained, is migration convenience only;
- `caravan::core` contains no speculative generic cross-backend native-event
  protocol, global outstanding-operation supervisor, required task type, worker
  pool, work-stealing scheduler, or competing scheduler hierarchy;
- optional resource dependency inference, if implemented, is a removable layer
  over async composition and does not own application storage or backend execution;
- the Phase 2 P2300 spike constrains the implementation, including async scopes,
  run-loop semantics, continuation placement, and sender lifecycle;
- migration to a production standard-execution implementation remains possible
  without rewriting MPI progress or accelerator-native integration;
- PMacc and PIConGPU CPU/GPU/multi-rank test matrices pass;
- agreed performance criteria are met or deviations are explicitly accepted;
- legacy event/runtime code and temporary migration adapters are deleted;
- Caravan targets contain no PMacc or PIConGPU headers;
- the resulting Caravan library is independently useful as a heterogeneous async
  interoperability/backend layer, while PMacc remains free to choose its tasking,
  resource-dependency, and parallel execution systems.
