# mallocMC alpaka3 port

## Plan

- [x] Inspect `mallocMC` for legacy alpaka integration points and identify likely alpaka3 replacements.
- [x] Update dependency fetching and CMake integration to pull `alpaka-group/alpaka3` and match alpaka3's build requirements.
- [x] Replace legacy accelerator/tag/device/queue usage in library headers with alpaka3 APIs.
- [x] Replace legacy accelerator/tag/device/queue usage in tests and examples with alpaka3 APIs.
- [x] Update warp-size handling to use alpaka3 `onAcc::Acc` compile-time information where required.
- [x] Build and run the CPU test suite.
- [x] Build and run the CPU examples across the enabled host backends.
- [ ] Configure and compile the project with `nvcc` without running GPU tests.
- [ ] Do a final regression pass and record remaining risks.

## Notes

- Keep changes local to `mallocMC`.
- Prefer API adapters or small helper traits over wide rewrites where possible.
- The temporary `alpaka3_compat.hpp` and `alpaka3_host.hpp` helpers were removed again; the port now uses alpaka3 APIs directly.
- CPU tests and CPU examples pass with direct alpaka3 usage.
- Remaining CUDA compile failures are isolated to the native-CUDA wrapper path in `mallocMC.cuh` / `examples/native-cuda` and not to the CPU-tested allocator code paths.
