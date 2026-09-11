# Caravan scheduling contracts

This document describes the current Caravan placement, Alpaka, and MPI contracts.

## Core placement

Include `<caravan/core.hpp>` or the individual sender headers.

| Algorithm | Effect |
| --- | --- |
| `startsOn(scheduler, sender)` | Schedule child initiation and expose that scheduler in the child environment. Completion is not restored. |
| `on(scheduler, sender)` | Schedule child initiation and restore terminal completion through the scheduler from the receiver environment. |
| `continuesOn(sender, scheduler)` | Transfer terminal completion to an explicit scheduler. |

All three support pipe syntax. `then` runs a callback at upstream completion;
`letValue` creates and starts a returned sender at upstream value completion.
Neither algorithm supplies thread affinity by itself. Use `continuesOn` before
blocking or thread-affine application callbacks.

`on` requires an ambient scheduler in the receiver environment. An enclosing
`startsOn` can provide it:

```cpp
auto work = caravan::startsOn(
    application.scheduler(),
    caravan::on(backend.scheduler(), makeWork())
        | caravan::then(updateApplicationState));
```

Placement wrappers add no worker or progress thread. Scheduler resources and
borrowed objects must outlive connected operations. A receiver may destroy its
operation during terminal completion, so delivery code must not access operation
state afterward.

## Alpaka submission

Include `<caravan/alpaka.hpp>`. Queues are explicit arguments to `submit`,
`kernel`, `copy`, and `fill`. There is no Alpaka submission scheduler: native
`alpaka::sequence` and same-domain `whenAll` express queue dependencies directly.

```cpp
auto work = caravan::whenAll(
    caravan::alpaka::kernel<Acc>(queueA, workDiv, kernelA),
    caravan::alpaka::kernel<Acc>(queueB, workDiv, kernelB))
    | caravan::alpaka::sequence(
          caravan::alpaka::kernel<Acc>(queueA, workDiv, kernelC));
```

`then`, `letValue`, and placement wrappers are host-completion boundaries. Native
fusion does not cross them. Submission callables run on the host and must enqueue
tracked work rather than inspect unfinished device results.

`retain(value, owner)` keeps the owner alive through native completion. Retention
does not synchronize access and does not extend the lifetime of borrowed queues.
Synchronous submission failure skips dependent stages but does not retract
independent or already-enqueued work. Retained state is released only after
recorded cleanup fences become terminal.

## MPI submission and progress

Include `<caravan/mpi.hpp>` for ordinary and native MPI operations. This umbrella
now exposes `mpi.h`; core and Alpaka headers remain MPI-independent.

Ordinary operations (`send`, `receive`, reductions, gathers, barriers, and
communicator operations) are thin wrappers over `mpi::request` or `mpi::invoke`.
Construction and connection remain allocation-free where previously guaranteed.
Starting every operation submits through the context's existing FIFO owner queue.
There is no MPI scheduler, bound primitive, scheduled-initiation branch, owner
thread inline bypass, or additional progress thread.

The context owns communicators, validates operations on its owner, retains buffer
owners, recovers partially started request batches, progresses requests with
`MPI_Testsome`, and drains accepted work during shutdown. `MpiRuntime` supplies a
dedicated owner thread. `MpiExternalRuntime` instead requires the caller to invoke
`progress()` on the thread that owns MPI.

`mpi::request<T>` accepts a start callback returning `NativeRequestBatch` and a
completion callback. `mpi::invoke` accepts a callback for immediate owner-thread
MPI work:

```cpp
auto rank = caravan::mpi::invoke(
    mpi,
    [](caravan::NativeMpiContext& native) {
        int value = -1;
        MPI_Comm_rank(native.communicator(caravan::worldCommunicator), &value);
        return value;
    });
```

All native callbacks run on the MPI owner. A blocking `invoke` callback blocks
request progress, so keep callbacks short unless that serialization is intended.
Recursive native submission from a native callback is rejected.

Both `request` and `invoke` accept an optional collective communicator. Ordinary
collective wrappers supply it, allowing `CollectiveLane` to validate context and
communicator compatibility. Every rank must still reserve, abandon, and start the
same collective sequence; local scheduling cannot establish cross-rank agreement.

### Mixed Alpaka/MPI chains

Ordinary MPI senders self-queue, and successful completion is delivered on the MPI
owner. A following `letValue` therefore starts its sender on that owner unless an
explicit transfer intervenes:

```cpp
auto work = makeAlpakaA()
    | caravan::letValue([&] {
          return caravan::mpi::send(mpi, buffer, peer, tag);
      })
    | caravan::letValue([&](caravan::SendResult const&) {
          return makeAlpakaB();
      });
```

To place application completion explicitly, transfer it:

```cpp
auto placed = std::move(work)
    | caravan::continuesOn(applicationLoop.scheduler())
    | caravan::then(updateApplicationState);
```

Queue acceptance is not request completion. Acceptance/rejection errors need not
have owner-thread affinity; successful request and invocation callbacks do.

## Validation

The registered Caravan tests cover:

- core placement, failure propagation, shutdown, and receiver-driven destruction;
- allocation-free ordinary MPI sender construction/connection;
- ordinary and native MPI operations, FIFO initiation, collective ordering,
  retained lifetimes, partial-start recovery, and failure cleanup;
- dedicated MPI runs with 1, 2, and 4 ranks and external progress with 1 and 2;
- mixed Alpaka/MPI data flow and explicit `continuesOn` placement.

Run available CPU, CUDA, and HIP configurations through the existing CMake/CTest
targets and `share/ci/run_caravan_device_tests.sh`. Record unavailable device
backends as compile-only or not tested; do not infer GPU behavior from CPU queues.

No submission framework, scheduler hierarchy, cancellation model, or progress
thread is planned without a measured requirement.
