# Caravan standard-execution spike

## Result

Tested with NVIDIA/stdexec `nvhpc-25.09` (commit
`1f6379682dd1598c9b48313fa6dfdae620bc8535`). A thin adapter is preferable to
making the migration senders direct stdexec models:

```cpp
auto chain
    = caravan::stdexecInterop::adapt(caravan::alpaka::kernel(...))
    | stdexec::let_value([&] {
          return caravan::stdexecInterop::adapt(caravan::mpi::send(...));
      })
    | stdexec::continues_on(controlLoop.get_scheduler())
    | stdexec::then(userContinuation);
```

`include/caravan/test/stdexec.cpp` runs this chain on CPU using
`exec::async_scope::spawn_future`, `stdexec::sync_wait`, and
`exec::async_scope::on_empty`. It also verifies that the final continuation runs
on the stdexec run-loop thread. The same chain translates with nvcc 13.3.

The adapter only translates completion signatures and receiver completion CPOs.
It does not change MPI progress or alpaka submission, and it allocates no state of
its own.

## Decisions and blockers

- **Adapter, not direct modeling:** current operations call receiver members and
  expose Caravan completion-signature types. The adapter isolates both
  differences in one header and leaves every backend unchanged.
- **Environment:** current backends need no environment data. The adapter forwards
  receiver environments, but Caravan operations do not query them, so stdexec
  stop tokens do not yet cancel or suppress backend start.
- **Alpaka fusion:** existing alpaka FIFO/event fusion remains available before
  adaptation. A stdexec domain transformation could preserve fusion across
  standard algorithms, but was not implemented: it would add machinery without a
  current mixed-chain requirement.
- **Run loop:** stdexec's run loop works with a dedicated driver thread. It cannot
  directly replace PMacc's current manually pumped loop because it has no
  `runOne`/`runReady` API.
- **Scope:** `exec::async_scope` replaces Caravan `AsyncScope` for the CPU spike
  without backend changes. Its `on_empty()` is the tested equivalent of join;
  this release has no close/join spelling. Instantiating its `spawn_future` path
  under nvcc 13.3 fails stdexec's `sender_in` constraints, so the CUDA gate covers
  the representative chain only, not stdexec scope lifetime.
- **CUDA/HIP:** nvcc 13.3 translates the chain but emits four host/device-call
  diagnostics from stdexec adaptor closures. HIP was unavailable on the test
  host and remains an open hardware/toolchain gate.
- **Type erasure:** `Event` remains justified for runtime-sized joins, imperative
  PMacc boundaries, and toolchain/type-complexity firebreaks.

The custom Caravan types are P2300-shaped migration types, not source-compatible
P2300 senders. Standard composition requires `stdexecInterop::adapt`.

## Measurements

Debug CPU build, GCC 16.2.1, Open MPI, one rank:

| Metric | Custom representative | stdexec spike |
|---|---:|---:|
| clean TU build wall time | 2.83 s | 3.40 s |
| executable file sections (`size`, decimal) | 471,673 B | 465,396 B |
| median full-process runtime, 5 runs | 1.378 s | 1.378 s |
| Massif peak heap, full stdexec process | n/a | 4.264 MiB |

MPI initialization dominates runtime and heap. Structurally, the stdexec sender
expression and Caravan adapter allocate nothing; each tested
`exec::async_scope::spawn_future` owns one heap future state. Isolated operation
latency/allocation benchmarking belongs to A8 rather than this compatibility
spike.

The nvcc 13.3 translation took 5.65 s including the initial Caravan MPI build;
a subsequent chain-only rebuild succeeded with the diagnostics noted above.
These figures are directional, not performance acceptance criteria.

## Reproduce

```bash
git clone --branch nvhpc-25.09 --depth 1 \
  https://github.com/NVIDIA/stdexec.git /tmp/stdexec
cmake -S <caravan-wrapper> -B build/caravan-stdexec \
  -DCARAVAN_BUILD_ALPAKA=ON \
  -DCARAVAN_BUILD_TESTING=ON \
  -DCARAVAN_BUILD_STDEXEC_SPIKE=ON \
  -DCARAVAN_STDEXEC_INCLUDE_DIR=/tmp/stdexec/include
cmake --build build/caravan-stdexec --target caravan-stdexec-spike
ctest --test-dir build/caravan-stdexec/caravan -R caravan-stdexec-spike
```

The spike is opt-in; normal Caravan and PMacc builds do not depend on stdexec.
