# Caravan scheduling

Include `<caravan/core.hpp>` or the individual headers under
`caravan/core/sender/`.

| Algorithm | Effect |
| --- | --- |
| `startsOn(scheduler, sender)` | Schedule child initiation; do not restore completion placement. |
| `on(scheduler, sender)` | Schedule child initiation; restore terminal completion through the ambient scheduler. |
| `continuesOn(sender, scheduler)` | Transfer terminal completion to an explicit scheduler. |

All three also support `sender | algorithm(scheduler)`.

## Choosing restoration

`on` queries `getScheduler(receiver.get_env())` at connection. An enclosing
`startsOn` supplies that logical current scheduler to its child:

```cpp
auto work = caravan::startsOn(
    application.scheduler(),
    caravan::on(backend.scheduler(), makeWork())
        | caravan::then(updateApplicationState));
```

The application loop initiates the outer chain. The backend loop starts
`makeWork()`'s connected operation. Its terminal completion is posted back to
the application loop before `updateApplicationState` runs. Both loops must be
driven; a scheduler handle does not create a worker or progress engine.

For restoration without a return hop:

```cpp
auto work = caravan::startsOn(
    caravan::InlineScheduler{},
    caravan::on(backend.scheduler(), makeWork())
        | caravan::letValue(makeNextSubmission));
```

Inline restoration invokes `makeNextSubmission` on the thread delivering the
backend operation's completion. It does not return to the thread that started
`work`. The returned sender can later complete on a different thread.

If restoration is not wanted at all, prefer `startsOn(backend.scheduler(),
makeWork())`. Unlike `on`, its completion placement does not depend on the ambient
scheduler. Neither algorithm changes the execution semantics of ordinary callable `then` or
`letValue`: short callbacks run inline on upstream value completion. Transfer
expensive or thread-affine host work explicitly with `continuesOn`.

## Environment and lifetime contract

- `startsOn` overlays the current scheduler for the whole child expression.
  Nested `on` restores the nearest enclosing context, not the original physical
  thread. Asynchronous completion and `continuesOn` do not rewrite that environment.
- Custom receiver environments provide `query(caravan::GetScheduler)` returning
  a scheduler handle. Other query CPOs are forwarded through the overlay by
  invoking them on the original environment; arbitrary environment member names
  are not automatically forwarded.
- `on` without a queryable ambient scheduler fails to connect. `AsyncScope`,
  `syncWait`, and `ControlContext::spawn` do not implicitly supply one; use an
  enclosing `startsOn` when necessary.
- Construction/connection does not start work. Child and scheduling operations
  connect before start, and connection failures throw to the caller.
- Failed or stopped initiation scheduling skips the child. `on` restores those
  outcomes as well as the child's value/error/stopped completions. Failure while
  storing a completion value is also transferred. Failed/stopped restoration
  replaces the pending outcome and cannot guarantee the requested affinity.
- Connected operations must stay at a stable address until terminal completion.
  Borrowed scheduler resources must outlive them. Placement adds no worker,
  queue, or heap allocation of its own; the selected scheduler may allocate.

These are Caravan APIs, not direct standard-execution models. stdexec
scheduler/domain metadata translation is not implemented.

## Alpaka submission domain

Include `<caravan/alpaka.hpp>`. Existing `submit`, `kernel`, `copy`, `fill`, and
`size` factories return typed native submissions. Chain them with
`alpaka::sequence` (or its pipe adaptor); `then` accepts callables, not senders:

```cpp
caravan::alpaka::Scheduler scheduler{queueA}; // borrows queueA

auto work = caravan::whenAll(
    scheduler.submit([=](auto& queue) {
        alpaka::exec<Acc>(queue, workDiv, kernelA, argsA);
    }),
    caravan::alpaka::kernel<Acc>(queueB, workDiv, kernelB, argsB))
    | caravan::alpaka::sequence(caravan::alpaka::kernel<Acc>(queueA, workDiv, kernelC, argsC))
    | caravan::then([] { /* Host callback: all preceding native work is complete. */ });

caravan::syncWait(std::move(work));
```

The scheduler's `submit(f)` is shorthand for `alpaka::submit(queueA, f)`.
`schedule()` completes inline on its caller, using the existing inline scheduler;
that scheduling completion is **not** device completion. `startsOn(scheduler, work)`
can expose it as the current scheduler, but is not needed for native composition.
There is no submission worker or public completion-thread scheduler.

- `getDomain` queries explicitly typed descriptions and schedulers. Alpaka uses
  `SubmissionDomain<Queue>`; runtime queue identity remains separate.
- `whenAll` customizes only when all children are native submissions of the same
  queue type. It preserves independent branches, including nested joins. Distinct
  queues may overlap; work sharing a queue remains FIFO.
- `alpaka::sequence(nativeSubmission)` connects predecessor tails to continuation roots using
  FIFO or native event waits. All reachable submissions are issued during `start`,
  without an intermediate host-observed device completion. Unbranched adjacent
  same-queue stages retain the existing shared-fence fast path.
- `then(f)` and `letValue(f)` keep host-completion semantics. After a mixed-backend
  join, host callback, different queue type, or placement wrapper, use `letValue`
  with a factory returning the next submission. It starts after successful host
  completion and can use predecessor values.
- No fusion crosses explicit `on`, `startsOn`, or `continuesOn` wrappers. Compose
  native work inside the placement scope when native batching is desired.

There is no regrouping across a host boundary: return
`a | caravan::alpaka::sequence(b)` from a `letValue` factory to submit A and B as
one native continuation. Separate `letValue` stages retain separate
host-completion boundaries.

Submission callables execute on the **host**, not inside a device kernel. They
must enqueue tracked work on the supplied queue, must not read unfinished results,
and must not wait for completion. Explicitly retained captures live until all
submitted work is quiescent; queues and unowned pointees must outlive that work.

A synchronous submission failure skips dependent stages, but independent branches
are still submitted. An asynchronous execution error cannot retract an already
queued continuation. Submission errors take precedence over observed execution
errors; terminal completion waits for all recorded cleanup fences. Error detection
remains backend-specific. This is native dependency ordering, not device-side
conditional execution or cancellation.

The implementation uses a fixed-size dependency matrix for explicit expressions
(O(N²) storage/scans), with no dynamic graph engine, device-value storage model, or
ambient/late domain rewriting. These are deliberately narrower contracts than
nvexec's device-callable `then` and stdexec's domain machinery.
