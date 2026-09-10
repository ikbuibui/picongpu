# Caravan schedulers, domains, and scoped placement

## Goal

Make short native submissions cheap and natural to compose. An alpaka -> MPI -> alpaka chain should submit the second alpaka operation directly from MPI completion, without returning through an application or alpaka completion-thread queue.

Explicit user placement remains authoritative. If the user requests a return to an application scheduler or introduces a host-completion boundary, preserve it rather than silently removing it for performance.

This is an implementation plan; not all APIs below are implemented. Examples use Caravan's existing camelCase naming convention.

Current subset: core placement (`startsOn`, `on`, `continuesOn`) is implemented. Alpaka now has a borrowed queue-bound `alpaka::Scheduler`, an explicit `SubmissionDomain<Queue>`, native generic `whenAll`, and explicit `alpaka::sequence` for native chaining. The sender-taking `then` experiment was reverted: `then` accepts callables, while `letValue` starts a sender returned by a factory. Both remain host-completion boundaries. See [implemented contracts and syntax](CARAVAN_SCHEDULING.md#alpaka-submission-domain).

This subset uses eager customization of explicit submissions, not ambient/late domain rewriting. MPI scheduler factoring, stdexec metadata translation, and the broader migration/benchmark work below remain planned.

## 1. Fix the contracts before refactoring

Keep four responsibilities separate:

| Concept | Responsibility |
| --- | --- |
| Context/resource | Native resource ownership, authority, progress, and shutdown |
| Scheduler | Resource selection and placement of scheduling completion |
| Domain | Backend-aware lowering of supported sender expressions |
| Operation sender | Native initiation, request/fence tracking, and terminal completion |

Schedulers are lightweight handles, not replacements for contexts. Borrowed contexts and queues must outlive connected operations. Keep MPI communicator identifiers and request ownership explicit; execution on the owner thread alone is not sufficient resource binding.

### Placement algorithms

- `startsOn(scheduler, sender)`: start through the scheduler and expose that scheduler as the current scheduling context to the child. Do not restore completion placement.
- `on(scheduler, sender)`: capture the ambient current scheduler from the receiver environment, start in the selected context, then restore terminal completion through the captured scheduler.
- `continuesOn(sender, scheduler)`: explicitly transfer downstream completion. Do not treat the physical completion thread as an implicit environment update.
- `then(f)` and `letValue(f)`: ordinary host callbacks run inline on upstream value completion. `letValue` starts the returned sender there; that sender determines its own completion placement.

Restoration uses a scheduler, not a saved thread identity. Inline restoration executes wherever completion arrives. Missing ambient scheduling context for `on` is a diagnostic, not an implicit fallback to an application thread or inline scheduler.

Specify value/error/stopped handling and scheduling-failure precedence. Preserve existing `continuesOn` behavior where scheduling failure replaces the pending completion. Do not claim universal affinity for errors occurring before placement succeeds.

## 2. Establish the preferred API and execution path

Keep existing resource-explicit convenience factories:

```cpp
auto work = caravan::alpaka::kernel<Acc>(queue, /* ... */)
    | caravan::letValue([&] {
          return caravan::mpi::send(mpi, buffer, peer, tag);
      })
    | caravan::letValue([&] {
          return caravan::alpaka::kernel<Acc>(queue, /* ... */);
      });
```

Their conceptual definitions are:

```text
mpi::send(context, ...) = startsOn(MpiScheduler{context}, bound MpiSend{...})
alpaka::kernel(queue, ...) = startsOn(AlpakaScheduler{queue}, bound Kernel{...})
```

These are semantic decompositions, not requirements to allocate or execute extra wrapper operations. Lower them into the existing backend paths where possible. Convenience factories use non-restoring placement: they must not acquire a hidden return hop from the caller's ambient scheduler.

Expected successful execution:

```text
Application thread:       submit A; return without waiting
Alpaka completion thread: observe A complete; enqueue MPI initiation
MPI owner thread:         initiate and progress M
                          observe M complete; invoke factory; submit B directly
Alpaka completion thread: observe B complete; run downstream inline callbacks
```

`then` callbacks between M and B occupy the MPI progress thread. Document that short submission work is intended there; expensive host work should use explicit placement.

Expose scoped `on` for users who want restoration. An inline outer context reproduces the no-return-hop behavior:

```cpp
caravan::startsOn(
    caravan::InlineScheduler{},
    makeA()
        | caravan::letValue([&] {
              return caravan::on(mpiScheduler, makeBoundMpiPrimitive());
          })
        | caravan::letValue([&] { return makeB(); }));
```

Replacing the outer inline scheduler with an application run-loop scheduler intentionally returns MPI completion to that loop before submitting B. Examples must distinguish bound primitives from already-scheduled convenience senders to avoid teaching double placement.

## 3. Add the minimum core environment and placement support

Primary files: `include/caravan/core/sender/common.hpp`, `include/caravan/core/sender/`, `include/caravan/core/inline_scheduler.hpp`, and `include/caravan/core/run_loop.hpp`.

- Add current-scheduler and domain queries plus a small environment overlay that preserves unrelated receiver queries.
- Implement `startsOn` and restoring `on` using existing scheduler senders and operation-state patterns.
- Forward environments consistently through `then`, `letValue`, `whenAll`, and placement algorithms; test nested scopes and restoration to the outer scheduler.
- Keep runtime queue/resource identity separate from the domain type. A common domain does not make two queues interchangeable.
- Avoid type erasure, dynamic scheduler registries, and heap allocation introduced solely by domain dispatch or inline placement.
- Keep submission scheduler metadata distinct from completion scheduler metadata. An inline alpaka scheduler must not be advertised as the kernel's completion scheduler merely because it supplied the queue.

Do not build a full standard-execution implementation. Match the intended placement semantics and verify terminology against the pinned stdexec version before publishing interoperability claims.

## 4. Factor MPI scheduling from native initiation

Primary files: `include/caravan/mpi/context.hpp`, `context.cpp`, `operations.hpp`, `operations.cpp`, and `native.hpp`.

- Add `MpiScheduler` as a handle to a specific context's owner-thread execution queue.
- Separate queued submission from owner-only primitive initiation. A scheduled primitive starts/tracks native requests directly instead of enqueuing a second command.
- Fuse the convenience `startsOn(MpiScheduler, primitive)` path into one queue entry when starting outside the owner thread.
- Keep request completion asynchronous and driven by the existing request engine. Submission acceptance is not send completion.
- Preserve request recovery after partial submission, retained buffer lifetimes, shutdown rejection/draining, and scoped native authority.
- Preserve FIFO initiation and `CollectiveLane` behavior. Scheduling does not solve cross-rank collective agreement.
- Do not add an automatic owner-thread inline fast path if it would overtake queued work or break native-callback/reentrancy restrictions. The initial performance target is no double queueing, not changed FIFO semantics.
- Preserve both dedicated `MpiRuntime` and externally pumped `MpiExternalRuntime` operation. A scheduler handle does not create a progress thread.

## 5. Add the inline alpaka submission scheduler and domain

Primary file: `include/caravan/alpaka/operations.hpp`.

- Add a queue-bound `AlpakaScheduler` whose scheduling operation completes inline on its caller, with an alpaka domain for lowering.
- Keep native kernel/copy/fill initiation on that caller. Do not add a submission worker or a public scheduler for the shared completion thread.
- Reuse existing submission operation state, fences, retained captures, and polling engine.
- Continue to publish generic terminal completion only after native work is quiescent, including error cleanup.
- Preserve the distinction between inline scheduling, asynchronous native execution, and host-observed completion.
- Verify that supported queue/device backends permit submission on the initiating thread; preserve any required native device activation and queue concurrency rules.

## 6. Preserve explicit alpaka sequencing alongside domain composition

Target: combine native `whenAll` with explicit `alpaka::sequence`, without changing host callback semantics.

- Introduce a minimal domain customization point for supported compositions.
- Reuse existing native submission senders for known kernel/copy/fill work. Implemented spelling: `previous | caravan::alpaka::sequence(caravan::alpaka::kernel<Acc>(queue, ...))`; `scheduler.submit(f)` binds a host-side submission callable to its queue. This is not nvexec-style execution of `f` inside a device kernel.
- Lower recognized adjacent operations on the same queue to FIFO submission with the existing shared terminal fence.
- Lower recognized queue changes to native event dependencies and the existing per-run error/lifetime boundaries.
- Retain operation captures until all relevant work is quiescent, even if terminal receiver completion destroys the operation.
- Keep generic `then`/`letValue` callbacks as host-completion boundaries. Do not invoke an arbitrary factory early just because its return type is an alpaka sender: it may read results or have observable side effects.
- Do not fuse across explicit placement boundaries unless the requested execution and completion behavior is provably preserved.
- Keep `alpaka::sequence` as the native sequencing API. Do not overload `then` to accept senders or replace native sequences with unfused generic `letValue` chains and call that a performance-preserving migration.

Initial scope is host submission composition, not transforming arbitrary host lambdas into device kernels. Dynamic host-dependent selection retains its necessary completion boundary.

## 7. Verify correctness and performance before migrating PMacc

Extend existing tests in `include/caravan/test/`; use deterministic thread markers and queue/fence counters rather than timing assertions.

Required checks:

1. A -> M -> B: B is submitted on the MPI owner thread, but its terminal completion and following `then` execute on the alpaka completion thread.
2. `on` with inline restoration: same path, with no restoration queue entry.
3. `on` with application restoration: B is submitted only after the application loop runs; the explicit hop is preserved.
4. Nested scheduling environments, missing-context diagnostics, and value/error/stopped restoration including scheduling failure.
5. MPI convenience submission adds one owner-queue command, not two; collective ordering and external progress tests remain valid.
6. Recognized same-queue alpaka composition has no intermediate host completion; cross-queue composition retains native dependencies.
7. A generic callback that reads a completed result is not executed early.
8. Existing partial-failure, retained-owner, quiescence, shutdown, and receiver-destroys-operation checks still pass.

Benchmark against the current implementation using a persistent initialized runtime, excluding MPI startup. Measure submission latency, A -> M -> B latency, allocations, MPI queue entries, native fence counts, and long-chain behavior. Compare the convenience and explicit-placement forms to expose hidden overhead. Run CPU checks first, then available CUDA/HIP device gates; do not claim untested backend results.

Update `include/caravan/stdexec.hpp` and the optional spike to deliberately translate supported scheduler/domain metadata or document the adaptation boundary. The current adapter returns an empty sender environment and does not preserve domains; interoperability must not silently claim fusion that is lost there.

## Delivery order and completion criteria

1. Placement contracts, environment support, and core tests.
2. MPI scheduler factoring with one-queue-entry submission.
3. Inline alpaka scheduler using the existing completion engine.
4. Recognized domain composition with native fusion parity.
5. Mixed-chain tests, measurements, documentation, then representative PMacc migrations.

Done means the convenient A -> M -> B spelling has no avoidable return hop; restoring `on` has predictable ambient-context semantics; domain composition preserves existing native fusion where supported; and explicit user placement is never discarded for speed.

No new completion thread, scheduler hierarchy, mandatory stdexec dependency, cancellation model, or general device-lambda execution framework is required for this work.
