# Caravan Event-System Rework: Hardening and Simplification Plan

## Purpose

This file supplements [`PLAN.md`](PLAN.md) and
[`PLAN_REVIEW_ACTIONS.md`](PLAN_REVIEW_ACTIONS.md). It records follow-up work from
the review of commit `e34b53574d83667ce18b63df5caef6b8ecd397e4`.

The architecture selected in `PLAN.md` remains the target:

- backend primitives are lazy senders;
- dependency structure is expressed by sender composition;
- `Event` and `Future<T>` are eager, type-erased migration and interoperability
  boundaries;
- PMacc owns its async scope and optional manually driven control run loop;
- `MpiContext` owns MPI progress and MPI-affine native resources, but is not a
  general application scheduler;
- accelerator-local dependencies remain native to the accelerator backend;
- operation state owns only state and resources explicitly retained by the
  operation;
- resource-access dependency inference remains optional and above the core; and
- no global Caravan Manager, task hierarchy, or mandatory resource scheduler is
  introduced.

This plan does not redesign those decisions. It addresses correctness defects,
removes accidental complexity found in the current implementation, reduces eager
path allocation and dispatch cost, and defines the gates for completing the PMacc
migration.

## Current assessment

The branch contains a sender-oriented core, MPI request engine, alpaka submission
backend, PMacc scope/run-loop integration, and representative field, particle, and
example paths. The C1-C6 correctness implementation is complete and its focused
CPU and multi-rank tests pass. The implementation is not yet a replacement for the
legacy event system because the simplification, migration, performance, and full
validation gates remain open.

The current implementation has two main kinds of work remaining:

1. **Transitional complexity:** the legacy Manager/task system, explicit
   runtime-sized eager adapters, and new sender infrastructure coexist; remaining
   PIConGPU callers still keep the old PMacc task paths alive.
2. **Unverified performance and platforms:** allocation counts, MPI progress cost,
   sanitizer runs, GPU-aware MPI, and the performance gates in `PLAN.md` have not
   been demonstrated.

The C1-C6 gate for expanding migration is satisfied. The PMacc API-stability and
PIConGPU entry gates remain blocked by S5-S6, P1-P3, M1-M2, and V1.

---

# Design and correctness invariants

Every change in this plan must preserve the following invariants.

1. Starting an operation eventually produces exactly one value, error, or stopped
   completion unless the process is terminated because the backend cannot recover.
2. An operation state remains alive until its completion operation begins.
3. Terminal completion is not published while native work may still access
   operation-owned or explicitly retained storage.
4. Destroying an unstarted ordinary sender leaves no unrecoverable global or
   context state behind.
5. No user callback executes while an internal mutex is held or from a destructor.
6. Completion publication does not recursively traverse an unbounded continuation
   chain.
7. Application continuations execute only on their explicitly selected execution
   resource.
8. MPI calls execute only on the MPI authority allowed by the selected thread
   support policy.
9. Collective initiation order is identical across ranks for a managed
   communicator, independently of dependency-ready timing.
10. Typed sender composition remains allocation-free unless a backend operation or
    execution transfer intrinsically requires dynamic storage.
11. Eager type erasure is paid for only at a deliberate eager or dynamic boundary.

---

# Work ordering

| ID | Priority | Status | Work | Blocks |
|---|---|---|---|---|
| C1 | P0 | Implemented | Retain continuation schedulers safely | Any eager continuation use |
| C2 | P0 | Implemented | Make blocking consumers operation-lifetime safe | Public `syncWait` use |
| C3 | P0 | Implemented | Make managed collective abandonment safe | Multi-rank production use |
| C4 | P0 | Implemented | Preserve MPI requests and buffers on progress errors | MPI production use |
| C5 | P0 | Implemented | Propagate particle callback failures | Particle migration |
| C6 | P0 | Implemented | Give MPI submission strong exception safety | Reliable shutdown/error handling |
| S1 | P1 | Implemented | Establish sender-first PMacc APIs and name eager adapters | Stable PMacc API |
| S2 | P1 | Implemented | Replace particle callback state machines | Particle migration exit |
| S3 | P1 | Implemented | Simplify `AsyncScope` to counting-scope semantics | Eager allocation work |
| S4 | P1 | Implemented | Reduce communicator and MPI type-erasure layers | Stable MPI hot path |
| S5 | P1 | Partly implemented | Simplify collective ordering state | MPI maintainability |
| S6 | P1 | Open | Remove smaller accidental complexity | Stable internal implementation |
| P1 | P1 | Open | Add allocation and dispatch measurement harness | Allocation decisions |
| P2 | P1 | Open | Remove unconditional eager-path allocations | Performance gates |
| P3 | P1 | Open | Bound and tune progress-loop work | MPI latency and CPU-cost gates |
| M1-M2 | P1 | Open | Complete PMacc migration and delete legacy paths | PIConGPU entry gate |
| V1 | P1 | In progress | Run sanitizer, multi-rank, GPU, and performance validation | Production acceptance |

C1-C6 and S1-S2 are implemented. P1 measurements may proceed concurrently with
S3-S6, but structural allocation optimizations in P2 must use those measurements
and must not weaken the invariants above.

---

# Phase C: correctness blockers

**Implementation status: complete.** The changes now:

- retain eager continuation scheduler handles by value and reject `syncWait` on an
  executor thread before connect/start;
- use move-only managed-collective reservation tokens whose destruction skips an
  unused slot without executing user code;
- retain mixed-error MPI request groups and leases until every request is inactive,
  while treating unrecoverable `MPI_Testsome` errors as fatal;
- forward particle progression callback exceptions to the result and retain every
  eager watcher completion; and
- construct and commit MPI queue commands before updating outstanding counts, with
  exceptions from native `noexcept start()` boundaries delivered as errors.

Focused Caravan tests pass with one, two, and four MPI ranks. The PMacc 2D/3D
context, communicator, and particle tests pass, including all particle failure
injection points and a mixed failed/pending native MPI request group. ThreadSanitizer,
allocator-failure injection, fatal MPI-error injection, and target GPU validation
remain V1 work rather than C-phase implementation blockers.

## C1: Retain continuation schedulers by value

### Problem

`Event::then` and `Event::continueWith` accept an executor by lvalue reference and
capture its address. A continuation can outlive the scheduler handle object even
when the underlying run loop remains alive. `pmacc::async::Context::wait` exposes a
specific race by registering a continuation against a local scheduler handle.

### Change

- Accept cheap scheduler/dispatcher handles by value and move them into the
  continuation state.
- Keep the owning execution resource lifetime as an explicit precondition. A copied
  `RunLoopScheduler` may refer to its owning `RunLoop`, but it must never refer to a
  destroyed scheduler handle object.
- Audit every continuation and backend callback for captured references or raw
  pointers whose lifetime is shorter than the returned completion.
- Prefer APIs shaped like `continuesOn(sender, scheduler)` over generic executor
  references with unclear ownership.

### Tests

- Complete an Event concurrently while the waiting thread executes unrelated
  run-loop work.
- Destroy the original scheduler handle after registering a continuation while the
  owning run loop remains alive.
- Exercise pending, already-ready, failed, and stopped predecessors.
- Run the race tests under ThreadSanitizer.

### Done when

- no continuation stores a pointer to a caller-owned scheduler handle;
- the scheduler-lifetime race is reproducibly covered; and
- application continuation placement remains explicit.

## C2: Make `syncWait` and other blocking consumers lifetime-safe

### Problem

`syncWait` starts a stack-owned operation before the pending-wait guard runs. When
called on an executor/progress thread, the guard can throw and destroy a still-live
operation state.

### Change

- Diagnose an invalid blocking context before connecting or starting the sender.
- Keep the operation state alive through every exception path after start.
- Distinguish a truly blocking consumer from a PMacc consumer that drives a run
  loop while waiting.
- Document which threads may call each consumer and which progress mechanism the
  consumer supplies.
- Audit `Event::wait`, `Future::result`, `Context::wait`, shutdown, and destructors
  for the same start-then-throw pattern.

### Tests

- Call `syncWait` on an executor thread with an already-ready sender; it may
  complete synchronously.
- Call it with a pending sender; it must reject the call before start or retain the
  operation until completion.
- Verify that rejection starts no backend work.
- Verify normal success, error, and stopped propagation from non-executor threads.

## C3: Make managed collective planning abandonment-safe

### Problem

`CollectiveLane::submit` reserves a sequence ticket when constructing a sender. An
unstarted or destroyed sender never retires that ticket, so all later entries can
remain permanently blocked.

### Change

Choose and implement one explicit contract:

1. **Abandonment-safe planned sender:** a reservation is represented by an internal
   move-only token. Destroying an unused token retires the slot through an internal,
   non-user-callback path. `MpiContext` must outlive all tokens.
2. **Explicit finalized collective plan:** constructing entries is a planning
   operation rather than ordinary sender construction. A plan is finalized before
   execution and guarantees that every reserved slot is either started or skipped.

The first option is preferable if the type can still satisfy ordinary sender
expectations without hidden unbounded work in destructors. The second option is
preferable if planning lifetime cannot be made obvious and safe.

In either design:

- failed and stopped predecessors retire their entries;
- construction, connection, and start exceptions cannot leave sequence gaps;
- shutdown accounts for unreleased and queued managed entries; and
- point-to-point work is not serialized behind a collective lane.

### Tests

- Construct and discard the first entry, then start the second.
- Connect but do not start an entry.
- Throw while constructing the sender factory, operation state, queue command, and
  native collective.
- Shut down with a skipped entry followed by a queued entry.
- Repeat the inversion tests on two and four ranks.

## C4: Preserve native MPI lifetime after `MPI_Testsome` errors

### Problem

The current error path fails all logical operations and clears all request and
retained-lifetime state. `MPI_Testsome` can report an error while other requests
remain active. Clearing their leases can allow MPI to access freed storage.

### Change

- Decode completion indices and per-request statuses returned with
  `MPI_ERR_IN_STATUS`.
- Complete or fail only requests known to be inactive.
- Retain pending requests, their `NativeGroup`, and all buffer/resource leases until
  they become inactive.
- Define which request kinds may be cancelled and drained safely.
- If the MPI implementation cannot provide a safe recovery path, abort instead of
  falsely publishing terminal completion and continuing with invalid lifetimes.
- Ensure a multi-request operation reports exactly one terminal completion after
  every request in the group is inactive.

### Tests

- Add a controllable fake/native-progress seam that can return an error with a mix
  of completed, failed, and pending requests.
- Verify that pending request leases remain alive.
- Verify exactly one logical completion for multi-request groups.
- Verify shutdown after a recoverable error and the selected fatal-error behavior.

## C5: Propagate all failures from particle chunk progression

### Problem

The particle `watch` helpers discard the successor Event returned by
`continueWith`. If an `after*` callback throws, only the discarded Event fails and
the state machine's result remains pending forever.

### Immediate correction

- Wrap every callback invocation in a terminal forwarding boundary.
- Convert callback exceptions into `result.setFailed(current_exception())`.
- Ensure every state transition either starts a retained successor or completes the
  result exactly once.
- Store or otherwise account for every eagerly created successor completion.

### Tests

Inject exceptions from:

- packing completion;
- size extraction;
- send initiation and completion;
- receive initiation and completion;
- particle insertion; and
- retry-loop setup.

Every test must observe a failed terminal Event and a quiescent scope rather than a
hang.

## C6: Give MPI submission strong exception safety

### Problem

Native operation `start() noexcept` functions call potentially throwing queue and
type-erasure code. MPI submission also advances `m_outstanding` and collective
sequence state before queue insertion is known to have succeeded.

### Change

- Catch all exceptions at every `noexcept start()` boundary and send them through
  `set_error`.
- Construct queue commands before mutating counters and sequence state where
  possible.
- Otherwise add rollback guards for `m_outstanding`, sequence allocation, and
  managed tickets.
- Notify the progress worker only after a command is committed.
- Ensure a rejected submission completes exactly once and cannot block context
  shutdown.

### Tests

- Use allocation-failure injection for command and callback construction.
- Reject submissions during shutdown through both normal and native APIs.
- Verify `m_outstanding` returns to zero and collective sequences contain no gaps.

---

# Phase S: simplify the model and migration code

## S1: Make PMacc APIs sender-first

**Implementation status: complete for the migrated PMacc API.** Exchange,
GridBuffer, field-direction, and particle-chunk primitives return lazy typed
senders without `Context` or predecessor `Event` parameters. Runtime-sized
field, particle, and GridBuffer entry points are explicitly named
`spawnCommunication` and retain the deliberate eager boundary.

Normal composable PMacc operations should return lazy senders:

```cpp
auto packed = pmacc::fields::pack(queue, field, direction);
auto sent = caravan::letValue(
    std::move(packed),
    [&] { return buffer.send(mpi, direction); });

auto received = buffer.receive(mpi, direction);
auto inserted = caravan::letValue(
    std::move(received),
    [&](auto metadata) { return pmacc::fields::insert(queue, field, direction, metadata); });

auto exchange = caravan::whenAll(std::move(sent), std::move(inserted));
auto completion = context.spawn(std::move(exchange));
```

Rules:

- normal backend and PMacc primitives do not accept an `Event` predecessor;
- fixed graphs remain typed and lazy until one deliberate `spawn` boundary;
- runtime-sized joins and imperative legacy entry points may use `Event`;
- eager compatibility functions are named as such, live in an adapter header, or
  are otherwise unmistakably transitional; and
- avoid repeated `Event -> asSender -> spawn -> Event` conversions within a fixed
  branch.

## S2: Replace particle callback state machines

**Implementation status: complete.** Particle send and receive chunk loops are
lazy senders with dedicated recursive operation states. Value, error, and stopped
completion are forwarded structurally; the former `enable_shared_from_this`,
Event watcher, and control-scheduler callback chain is removed. Completed typed
stage states are retained inside the operation to make synchronous completion
safe; P1 will determine whether their list-node allocations need reusable slots.

The dedicated operation-state option was selected instead of adding a general
coroutine runtime solely for this loop. The implementation:

- keeps the chunk loop lazy until start;
- retains all per-direction state in one operation state;
- propagates value/error/stopped structurally;
- avoids one eager Event and scheduler hop per state transition;
- supports empty, partial, exact-capacity, and multi-chunk transfers; and
- exposes no state enum, callback chain, or Manager protocol to callers.

## S3: Replace the scope registry with counting-scope semantics

**Implementation status: complete.** `AsyncScope` now tracks only its open/joining/joined
state and live-operation count. Each spawned operation has one concrete allocation,
retains the scope association in its receiver, and destroys itself at terminal
receiver completion before decrementing the count. Generated IDs, the hash registry,
and virtual operation ownership are removed. Construction failure rolls back the
count, and focused tests cover synchronous self-destruction and failed connection.

`AsyncScope` needs to prevent destruction while spawned operations are live; it
does not otherwise use operation IDs.

- Remove generated IDs and the `unordered_map` registry.
- Use a counting-scope-style association retained by each spawned operation.
- Keep close/join behavior explicit and retain the terminate-on-invalid-destruction
  contract.
- Own each dynamically spawned operation with one allocation or one allocator
  request, not a virtual shared object plus a separate hash node.
- Allow operation-state destruction during receiver completion without accessing
  the invalid state afterward.
- Keep this component mechanically replaceable by standard
  `simple_counting_scope`/`spawn` facilities.

## S4: Reduce communicator and MPI type-erasure layers

**Implementation status: complete.** The sole communicator implementation is now
used directly: `ICommunicator` and its dimension-erasing `EnvironmentController`
registry are removed. Grid topology remains owned by `GridController` and
`CommunicatorMPI`, while asynchronous execution delegates directly to the attached
Caravan `MpiContext`. The eager communicator methods and compatibility-owned async
context are removed; the remaining legacy MPI tasks own their temporary eager
bridge locally until M2 deletes those tasks.

Ordinary `OperationSender<T>` now stores concrete, MPI-free operation descriptors
instead of an allocating `std::function` start closure. Native types and the
remaining queue callback erasure stay behind the normal/native header boundary. An
executable-local allocation check with GCC/libstdc++ records zero allocations for
borrowed-send construction and connect; P1 remains responsible for the full,
portable start/completion and callback-size baseline.

- Audit whether `ICommunicator` still needs runtime polymorphism after legacy task
  removal.
- Separate topology/domain configuration from async send/receive execution.
- Remove eager `startSendAsync`, `startReceiveAsync`, and `progressAsync` from the
  normal interface after their callers migrate.
- Avoid an additional PMacc async context owned only for compatibility methods.
- Measure whether `OperationSender<T>`'s `std::function` start and three callback
  objects allocate for representative operations.
- If material, replace callback type erasure with a concrete operation descriptor,
  small-buffer move-only erasure, or direct templated native sender while preserving
  the normal/native header split.

Do not expose native MPI types to ordinary PMacc or PIConGPU code merely to remove
one abstraction layer.

## S5: Simplify collective ordering state

- Verify whether raw collective submissions can ever reach the MPI worker out of
  their reservation order when ticket creation and FIFO queue insertion are one
  atomic transaction under the queue mutex.
- Remove the raw pending map if it cannot observe an inversion.
- Keep the managed ordering layer only where dependency readiness can invert
  initiation.
- Consolidate ticket allocation, queue commit, skip, and shutdown accounting into
  one state machine with explicit invariants.

## S6: Remove smaller accidental complexity

- Remove the duplicate `sendCompletion` dependency currently added by both field
  packing and `GridBuffer::asyncSend`.
- Bound `RunLoop::runReady` or document its drain-until-empty semantics so a
  self-reposting continuation cannot starve its caller indefinitely.
- Add a non-empty-stage constraint to alpaka `SubmitSender` construction.
- Separate const send buffers from mutable receive buffers in the MPI buffer API;
  do not require PMacc to `const_cast` send storage.
- Either implement valid root in-place gather semantics with `MPI_IN_PLACE` or
  reject overlapping gather input/output buffers.
- Remove obsolete factories, state enums, observers, and task stringification with
  their final callers rather than retaining compatibility shells.

---

# Phase P: allocation and dispatch performance

## P1: Establish a reproducible allocation baseline

Add a dedicated microbenchmark/test executable that records:

- allocation count;
- allocated bytes;
- peak live allocations;
- sender construction time;
- connect/start time;
- synchronous completion time;
- cross-thread completion-to-continuation latency; and
- contention with multiple producers.

Cover at least these cases:

| Case | Variants |
|---|---|
| `readyEvent`, Event copy, state observation | ready/error/stopped |
| `Event::then` and `continueWith` | already terminal and pending |
| typed `then` and `letValue` | void, 8-byte, 64-byte, move-only value |
| typed `whenAll` | 1, 2, 8 senders |
| runtime Event `whenAll` | 0, 1, 8, 26, 52 Events |
| `continuesOn` | void, 8-byte, 64-byte, move-only value |
| `AsyncScope::spawn` | synchronous and pending sender |
| MPI sender | construct, connect, submit, complete with fake/minimal backend |
| PMacc field direction | receive/insert and pack/send branch |
| Particle chunk loop | one, two, and many chunks |

Record Release results for supported GCC and Clang configurations. Keep a debug
configuration to make allocation-count regressions easy to diagnose. Report
standard-library version and whether `std::function` uses its small-object path.

Prefer executable-local allocation instrumentation or explicit test allocators.
Do not add production-global `operator new` hooks.

## P2: Allocation budgets and targeted changes

The first optimization pass should meet these structural budgets:

1. `readyEvent`, Event copies, terminal observation, and typed sender expression
   construction perform zero allocations.
2. Typed `then`, `letValue`, and fixed-arity `whenAll` add zero framework
   allocations through connect/start when their backend sender is also
   allocation-free.
3. `continuesOn` performs no unconditional `make_shared` allocation for transferred
   values. Store the value tuple in operation state and post a small handle to that
   state. For void and small values, the only permitted allocation is one required
   by the selected run-loop queue representation.
4. One eager continuation uses at most one combined successor/work allocation,
   excluding a queue node required by the dispatcher. Do not separately allocate
   `EventSource` state and callable state when they have the same lifetime.
5. `AsyncScope::spawn` uses one owned operation allocation, plus at most one shared
   completion allocation required by the returned Event/Future. It uses no
   per-operation map node.
6. Runtime-sized `whenAll` remains a flat aggregate. Its allocation count is
   constant with respect to input count apart from bulk storage; it does not create
   one independently allocated logical-and node per input.
7. The ordinary MPI path contains no avoidable allocation per callback channel.

These are budgets, not permission to add a broad custom memory subsystem. Apply
changes in this order:

1. remove unconditional allocations whose lifetime already fits in operation
   state;
2. combine allocations for objects with identical lifetime;
3. remove registries and type erasure that are no longer needed;
4. use bounded inline callable storage only after measuring remaining
   `std::function` allocations; and
5. consider pools, arenas, or allocator propagation only if the simpler changes do
   not meet the PMacc performance gates.

### Eager-path design constraints

- An asynchronous eager result intrinsically needs shared lifetime somewhere; the
  goal is not to claim zero allocations where ownership requires one.
- The already-ready fast path must be defined precisely. At minimum, constructing,
  copying, observing, waiting on, and converting a default ready Event to a sender
  must allocate nothing. Registering asynchronous work that returns a new shared
  Event may consume its single combined state allocation.
- Inline completion must continue to use a trampoline or equivalent bounded-stack
  dispatch.
- Allocation removal must not reintroduce a dangling operation-state pointer.
- Queue-node ownership and operation-state ownership must remain distinguishable in
  measurements.

## P3: Bound and tune progress work

### MPI worker

- Process a bounded command batch before each `MPI_Testsome` call.
- Guarantee that sustained producers cannot starve active-request progress.
- Make active polling policy configurable between busy spin, yield, and bounded
  backoff.
- Record progress-loop CPU usage, request completion latency, and sensitivity to
  reserving or not reserving a core per rank.
- Add optional affinity integration at the PMacc policy layer; do not hard-code
  machine placement in Caravan core.

### PMacc run loop

- Define a fairness policy for callbacks that continuously post more callbacks.
- Measure wake-up, queueing, and batching cost with one and many producer threads.
- Keep control-loop dispatch out of accelerator-native chains that do not require
  host-visible completion.

### Performance gates

Retain the gates from `PLAN.md`:

- representative MPI ping-pong latency and bandwidth within 5% of direct
  nonblocking MPI unless an understood platform-specific deviation is accepted;
- no unexplained full-step regression above 2% on the primary GPU configuration;
- PMacc run-loop/scope overhead below the legacy Manager for representative task
  and exchange counts;
- no scan proportional to all outstanding application operations; and
- native accelerator dependency chains do not insert intermediate host waits.

Add allocation-specific gates:

- no regression against the budgets in P2;
- no allocation count proportional to the number of typed adaptor layers; and
- no allocation per MPI completion channel in the ordinary send/receive path unless
  the measured callable exceeds documented inline storage.

---

# Phase M: complete migration and remove parallel infrastructure

## M1: PMacc migration

1. Convert field, particle, reduction, gather, signal, and helper operations to the
   sender-first API.
2. Keep one final eager spawn boundary per independent dynamic graph.
3. Remove `Event` predecessor parameters from normal primitives.
4. Move unavoidable legacy bridges into named adapter headers.
5. Port PMacc tests and examples before migrating PIConGPU call sites.
6. Verify CPU serial, threaded CPU, multi-rank, CUDA, HIP where available, and
   GPU-aware MPI behavior.

## M2: Legacy deletion

After the final PMacc caller has migrated, delete:

- the global Manager and transaction stack;
- task IDs, task maps, observers, and destructor notifications;
- logical-and task trees;
- device and MPI polling task classes;
- field and particle factories and state-machine tasks;
- global event pumping and Manager waits; and
- eager adapters whose only purpose was compatibility with those components.

Do not count the migration as simpler while both complete systems remain enabled.
Track deleted concepts and call sites, not only added sender equivalents.

---

# Phase V: validation matrix

## Correctness and sanitizers

- completion/registration races;
- scheduler-handle destruction races;
- executor-thread wait rejection;
- operation-state lifetime on all exceptions;
- collective construction/start inversion and abandonment;
- MPI progress errors with pending requests;
- shutdown with outstanding native and managed-lane work;
- particle empty, partial, exact-capacity, multi-chunk, and injected-failure cases;
- ThreadSanitizer on host concurrency tests;
- AddressSanitizer and UndefinedBehaviorSanitizer on CPU paths; and
- debug assertions for exactly-once start and completion.

## Backend and application matrix

- one, two, and four MPI ranks;
- MPI FUNNELED policy with dedicated worker;
- CPU serial and threaded alpaka queues;
- CUDA translation and runtime;
- HIP translation and runtime where supported;
- host-staged and GPU-aware MPI;
- `gameOfLife2D` and `heatEquation2D` output regression;
- field and particle exchange regression;
- signal, checkpoint, reduction, and gather paths; and
- complete PMacc shutdown and reinitialization where supported.

## Standard-execution compatibility

- Keep the existing thin stdexec adapter test compiling and running.
- Verify that backend fixes do not require rewriting MPI or alpaka integration when
  standard composition is used.
- Re-evaluate standard `run_loop`, `simple_counting_scope`, `spawn`, and coroutine
  facilities as supported compilers advance.
- Do not add Caravan environment, domain, or stop-token machinery without a tested
  backend or composition requirement.

---

# Completion checklist

## Before expanding PMacc migration

- [x] C1 scheduler handles retained safely.
- [x] C2 blocking consumers preserve operation-state lifetime.
- [x] C3 collective abandonment and shutdown are safe.
- [x] C4 MPI errors retain active requests and buffers.
- [x] C5 particle callback failures cannot leave pending results.
- [x] C6 MPI submission has strong exception safety.

## Before declaring the PMacc API stable

- [x] S1 normal PMacc operations are sender-first.
- [x] S2 particle chunking no longer uses the callback state machine.
- [x] S3 scope uses counting-scope rather than registry semantics.
- [x] S4 communicator and MPI type erasure have been reduced or justified by
      measurements.
- [ ] S5 collective ordering state has one documented model.
- [ ] S6 smaller redundant dependencies and unsafe edge cases are removed.
- [ ] Eager adapters are isolated and named as migration/interop boundaries.

## Before the PIConGPU entry gate

- [ ] P1 allocation and latency baseline is recorded.
- [ ] P2 eager and scope allocation budgets pass.
- [ ] P3 progress fairness, CPU use, and MPI latency gates pass.
- [ ] Remaining PMacc callers and tests are migrated.
- [ ] Legacy Manager, transactions, tasks, observers, and adapters are deleted.
- [ ] Sanitizer and backend validation matrices pass.
- [ ] Target GPU and GPU-aware MPI performance gates pass or deviations are
      explicitly understood and accepted.

---

# Explicit non-actions

Do not respond to this plan by:

- adding Event predecessors to primitive MPI or alpaka operations;
- making Event/Future the normal typed composition model;
- introducing a global Caravan scheduler or task manager;
- adding operation IDs to the simplified scope;
- building a general resource-ownership system into Caravan core;
- implementing the complete P2300 environment/domain model speculatively;
- adding cross-backend native event interoperability without a concrete second
  backend and measured requirement;
- hiding collective ordering inside readiness-dependent primitive MPI start; or
- introducing a custom pool/arena before the allocation baseline and simpler
  lifetime/layout changes are complete.
