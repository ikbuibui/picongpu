# Caravan and PMacc device testing on `hal8999`

This records the reproducible CUDA/HIP build procedure for the hardware-dependent
work in `PLAN.md`. It was last updated on 2026-09-02 at commit `07f0c2ff5`.

## Hardware and toolchains

- NVIDIA A30 (`sm_80`), CUDA 12.9.1
- AMD Radeon RX 7900 XTX (`gfx1100`), HIP/ROCm 7.0.2
- GCC 13.4.0, CMake 3.31.9, Open MPI 4.1.5
- Spack setup: `/home/spack/share/spack/setup-env.sh`
- PIConGPU dependency profile: `~/pixi-envs/pic/hal_picongpu_profile.sh`

The scripts default to these versions and architectures. Their environment
variables allow overriding them when the installed toolchains or devices change.
Build trees and logs default to `~/build/picongpu-device-tests`, outside the
source tree.

## Commands

From the PIConGPU checkout, run the complete matrix with:

```bash
./share/ci/run_device_test_matrix.sh
```

The component commands are:

```bash
# Caravan core, accelerator, accelerator-to-MPI, and MPI tests
./share/ci/run_caravan_device_tests.sh cuda
./share/ci/run_caravan_device_tests.sh hip

# gameOfLife2D and heatEquation2D compile/runtime regressions
./share/ci/run_pmacc_device_examples.sh cuda
./share/ci/run_pmacc_device_examples.sh hip
```

An individual example can be selected:

```bash
./share/ci/run_pmacc_device_examples.sh cuda gameOfLife2D
./share/ci/run_pmacc_device_examples.sh hip heatEquation2D
```

Useful overrides include:

```bash
BUILD_ROOT=/scratch/$USER/device-tests PARALLEL=8 \
    CUDA_ARCH=80 ./share/ci/run_caravan_device_tests.sh cuda

HIP_VERSION=7.0.2 HIP_ARCH=gfx1100 \
    ./share/ci/run_pmacc_device_examples.sh hip
```

Both scripts remove their selected old build trees by default. Set `CLEAN=0` for
an incremental rebuild. The scripts use `set -e`, so a
compile or test failure stops the run immediately. Consult `configure.log`,
`build.log`, and `test.log` in the corresponding build directory.

## Why the scripts contain explicit runtime-library paths

The local Spack CUDA module exposes `nvcc` but does not put the matching CUDA
runtime directory in `LD_LIBRARY_PATH` for executables without another dependency
that contributes an RPATH. Similarly, HIP-linked MPI tests require the GCC 13
`libstdc++` path when launched through `mpiexec`. The scripts add both paths as
needed. Omitting this produced misleading `libcudart.so.12` and
`GLIBCXX_3.4.31` loader failures even though compilation was successful.

The HIP PMacc examples load `rocrand` and `hiprand`. Do not use
`-Dalpaka_DISABLE_VENDOR_RNG=ON` for `gameOfLife2D`: its initialization kernel uses
`pmacc::random::methods::XorMin`. The standalone Caravan tests do not need vendor
RNG and disable it to keep that test environment minimal.

## Results recorded on 2026-09-02

### Passed

- Caravan CUDA translation and runtime:
  - accelerator kernel/copy/fill and same-/cross-queue chaining on the A30;
  - accelerator -> MPI -> run-loop continuation chain;
  - core/header tests and MPI tests with 1, 2, and 4 ranks.
- Caravan HIP translation and runtime:
  - accelerator kernel/copy/fill and same-/cross-queue chaining on the RX 7900 XTX;
  - accelerator -> MPI -> run-loop continuation chain;
  - core/header tests and MPI tests with 1, 2, and 4 ranks.
- CUDA `gameOfLife2D` release build and runtime regression with 1 and 4 ranks.
- CUDA `heatEquation2D` release build and four-rank runtime completed. Its final
  residual was `4.58355`, versus the CPU baseline `4.58358`.
- HIP `gameOfLife2D` release build and runtime regression with 1 and 4 ranks.
- HIP `heatEquation2D` release build and four-rank runtime regression. Its final
  residual was `4.58365`.
- The heat-equation CTest expression was relaxed to `4.583[0-9][0-9]` to permit
  the observed CPU/CUDA/HIP floating-point variation while retaining a narrow
  behavior check. CUDA and HIP CTest both pass with this expression.

### Still to finish

- GPU-aware MPI with device pointers was not exercised. The current Caravan
  representative chain intentionally uses the documented host-visible boundary
  and sends host memory.
- Performance baselines (submission cost, MPI ping-pong, overlap, and full-step
  regression) remain separate Phase 0/Phase 7 work.

ROCm 7.0 emits an alpaka warning because this vendored alpaka revision officially
lists HIP 5.1 through 6.2. The Caravan HIP translation and runtime tests nevertheless
passed on HIP 7.0.2; keep the warning visible when upgrading either dependency.

## Test fixes made while validating devices

- `include/caravan/test/alpaka.cpp` now selects CUDA or HIP when that backend is
  enabled instead of always instantiating `AccCpuSerial`. This makes the primitive
  accelerator test a real target-device runtime test.
- `include/caravan/test/core.cpp` no longer places required state transitions or
  `waitpid()` inside `assert()`. Release builds previously removed
  `source.setReady()` and then blocked forever in `wait()`.
- The heat-equation CTest accepts the narrow CPU/CUDA/HIP floating-point range
  described above.
