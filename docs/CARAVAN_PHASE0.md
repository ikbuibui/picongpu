# Caravan Phase 0 inventory and local baseline

Recorded on 2026-08-27 from `7964ed0823` before changing PMacc or PIConGPU.
This is the reproducible local baseline; GPU runtime and target-system
performance measurements remain CI/cluster work because this host has no GPU.

## Environment

- Linux 7.1.8, x86-64, Intel Core i5-1235U, 12 logical CPUs
- GCC 16.2.1, CMake 4.4.2
- Open MPI 5.0.10
- CUDA toolkit 13.3, no local GPU
- CPU backend: alpaka serial, release build

The CPU configuration used for PMacc, both examples, and PIConGPU was:

```sh
cmake -S <source> -B <build> \
  -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=OFF \
  -DCMAKE_CXX_COMPILER=g++
cmake --build <build> -j2
```

The examples additionally used `-DGOL_RELEASE=ON` and
`-DHEATEQ_RELEASE=ON`, respectively.

## Direct PMacc MPI inventory

The inventory scope is `include/pmacc`, `include/pmacc/test`, and
`share/pmacc/examples`. MPI constants used only for datatype/operation mapping
are not calls. Each direct-call owner and migration path is listed below.

| Classification | Current owner | Migration path |
|---|---|---|
| Bootstrap/finalize | `Environment.tpp`, `gameOfLife2D/main.cpp` | `MpiRuntime::run()` on process main; remove example finalization |
| Topology/communicators | `communication/CommunicatorMPI.cpp` | MPI-thread commands plus immutable topology snapshot |
| Point-to-point | `CommunicatorMPI.cpp`, `TaskSendMPI.hpp`, `TaskReceiveMPI.hpp` | `MpiExecutor::send/receive`; status/count copied into results |
| Barrier/progress | `eventSystem/mpiBarrier.cpp`, `simulationControl/SimulationHelper.cpp`, `simulationControl/Checkpointing.hpp` | nonblocking executor barriers; delete manager pumping |
| Signals | `eventSystem/tasks/TaskSignal.hpp` | signal-communicator all-reduce futures |
| Diagnostics barriers | `device/MemoryInfo.hpp` | executor barriers |
| Reduction | `mpi/reduceMethods/{Reduce,AllReduce}.hpp`, `mpi/MPIReduce.hpp` | executor collective futures and MPI-owned communicators |
| Gather | `mpi/GatherSlice.hpp` | executor gather/gatherv operations and MPI-owned communicator |
| Error checking | `communication/manager_common.hpp` | keep only inside Caravan MPI implementation |

The exact direct calls are reproducible with:

```sh
rg -n '\bMPI_[A-Za-z0-9_]+\s*\(' \
  include/pmacc share/pmacc test/pmaccHeaderCheck \
  --glob '*.{hpp,tpp,cpp,cu}'
```

No PMacc example has an unassigned MPI migration path. MPI-related datatype and
operation traits (`GetMPI_Op`, `GetMPI_StructAsArray`, math operation headers)
move behind Caravan descriptors when their collective callers are migrated.
The PIConGPU and third-party MPI inventory remains deferred to Phase 7.

## Legacy event-system inventory

There are 124 direct transaction/wait call sites. The exact list is
reproducible with:

```sh
rg -n '\b(getTransactionEvent|setTransactionEvent|startTransaction|endTransaction|waitForAllTasks|waitForFinished|mpiBlocking)\s*\(' \
  include/pmacc share/pmacc/examples test/pmaccHeaderCheck \
  --glob '*.{hpp,tpp,cpp,cu}'
```

Migration ownership is grouped by the component that removes each call:

| Current component | Migration path |
|---|---|
| `eventSystem/{Manager,eventSystem,transactions,events}` | Caravan Events/Flows; delete after last PMacc caller |
| Basic device tasks (`TaskKernel`, copy, fill, size transfer) | `DeviceExecutor` operations in Phases 3-4 |
| `TaskSend`, `TaskReceive`, `TaskSendMPI`, `TaskReceiveMPI` | temporary Phase 2 MPI adapters, deleted in Phase 5 |
| `TaskSignal`, `mpiBarrier` | `MpiExecutor` in Phase 2 |
| Buffer accessors and `GridBuffer` | explicit dependencies and allocation leases in Phases 4-5 |
| Field task classes and `FieldFactory` | direction Flows in Phase 6 |
| Particle task classes and `ParticleFactory` | continuation chunk loops in Phase 6 |
| `GatherSlice`, `MPIReduce`, reduce methods | MPI futures, then explicit Flows |
| `gameOfLife2D`, `heatEquation2D` | explicit fork/join Flows in Phase 5 |
| PMacc tests | Caravan operation APIs by the phase owning the tested component |

The custom state-machine classes are the `Task*` classes under
`eventSystem/tasks`, `fields/tasks`, and `particles/tasks`. Their migration is
covered by the rows above; none is retained for PIConGPU compatibility.

## Local compile and behavior baseline

### PMacc examples

Both CPU release builds passed. With a warm compiler/filesystem cache and two
build jobs, clean builds took 28.2 s for `gameOfLife2D` and 27.1 s for
`heatEquation2D`.

`gameOfLife2D` behavior commands:

```sh
mpirun -n 1 gameOfLife -d 1 1 -g 64 64 -s 5 -p 1 1
mpirun -n 4 gameOfLife -d 2 2 -g 128 128 -s 5 -p 1 1
```

| Ranks | Elapsed | Decoded final-image SHA-256 |
|---:|---:|---|
| 1 | 2.08 s | `5f2112abef1856c6483191dfe566a5757838a2fed2dd88b503ec23a61fc52301` |
| 4 | 1.67 s | `483909c57a62c29c331d399521eabba4a94851dc340375e41b90d184386b57b0` |

PNG files contain unstable metadata, so the hashes cover decoded pixels:

```sh
pngtopnm gol_000004.png 2>/dev/null | sha256sum
```

`heatEquation2D` passed its fixed four-rank, 1000-step run in 4.34 s. Its final
reported value was:

```text
Residual at time 999 = 4.58358
```

The example produced no PNG files with this local configuration, so the final
residual is the local reproducibility check. CTest smoke regressions now run
both Game of Life rank configurations and the four-rank heat equation, checking
the rule mask and final residual respectively.

### CUDA compile

CUDA 13.3 configuration succeeded, but the current untouched PMacc baseline
failed while compiling `SimulationHelper.cpp`:

```text
IdProvider.hpp(92): error: __host__ __device__ extended lambdas cannot be generic lambdas
```

This pre-existing toolchain incompatibility blocks local CUDA compile baselines
for both examples. CUDA compile and runtime baselines must be recorded by CI
with a supported compiler/toolkit before the Phase 6 gate.

### Untouched PIConGPU

The untouched PIConGPU CPU-serial release build passed in 145.3 s. This minimal
one-rank behavior run also passed:

```sh
mpirun -n 1 picongpu -d 1 1 1 -g 32 32 32 -s 1
```

Reported timings were 1.085 s full runtime and 6 ms average simulation-step
time. The final progress line was `100 % = 1`.

## Remaining target-system baseline work

Before Phase 2 changes PMacc startup, CI or a target cluster must record:

- CUDA runtime behavior and GPU-aware MPI exchange;
- current host submission cost and manager CPU time;
- MPI ping-pong latency/bandwidth for representative message sizes;
- halo overlap and full-step performance on the primary GPU configuration;
- CUDA compile baselines with a supported toolkit/compiler pair.

These measurements cannot be replaced by the local CPU numbers above and are
the remaining Phase 0 exit-gate work.
