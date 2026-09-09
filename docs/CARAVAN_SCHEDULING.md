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
scheduler. Neither algorithm changes the execution semantics of `then` or
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

These are Caravan APIs, not direct standard-execution models. Backend schedulers,
domains, and stdexec environment-query translation are not part of this change.
