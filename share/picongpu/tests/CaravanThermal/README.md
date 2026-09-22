# Caravan Thermal smoke scaffolding

This directory is the work-package-0 test scaffold for the first PIConGPU
Caravan vertical slice. It does **not** build PIConGPU or validate physics.
It registers only lifecycle/step smoke tests for the future dedicated minimal
Thermal executable:

| Test | MPI ranks | Arguments |
| --- | ---: | --- |
| `thermal-caravan-initialized-zero-step` | 1 | `-d 1 1 1 -g 32 32 32 -s 0 --periodic 1 1 1` |
| `thermal-caravan-smoke-two-steps` | 1 | `-d 1 1 1 -g 32 32 32 -s 2 --periodic 1 1 1` |

Neither test supplies a plugin/diagnostic option or `--no-start-simulation`.
Each test has a distinct working directory and a 120-second timeout. The
`32 32 32` domain gives each rank four-by-four-by-eight Thermal supercells,
which exceeds the `DomainAdjuster` minimum of three supercells per dimension.

## Configure the future minimal target

`thermal-caravan-cache.cmake` selects the existing Thermal extension, its
`cmakeFlags` preset 0 values (`EmZ`, `CIC`, and 25 PPC), and the opt-in
`PICONGPU_MINIMAL_CARAVAN_THERMAL` target. Configure it from `include/picongpu`:

```bash
cmake -S /path/to/picongpu/include/picongpu -B /tmp/thermal-build \
  -C /path/to/picongpu/share/picongpu/tests/CaravanThermal/thermal-caravan-cache.cmake \
  <supported-backend-options>
cmake --build /tmp/thermal-build --target picongpu-minimal-thermal
```

## Register and run the smoke tests

```bash
cmake -S /path/to/picongpu/share/picongpu/tests/CaravanThermal \
  -B /tmp/thermal-smoke-driver \
  -DPICONGPU_THERMAL_EXECUTABLE=/tmp/thermal-build/picongpu-minimal-thermal
ctest --test-dir /tmp/thermal-smoke-driver --output-on-failure
```

A mock executable may be used only to check CTest registration and argument
construction; it is not a simulation result.

## Evidence and blockers

- 2026-09-22: HAL profile loaded successfully: CMake 3.31.9, GCC 13.4.0, Open
  MPI 4.1.5, CUDA backend `cuda:80`.
- `7964ed0823` is a genuine pre-migration application baseline for this slice:
  its `Simulation.hpp` and `ParticlePush.x.cpp` retain `EventTask` and
  `eventSystem` transactions. A separate CPU worktree/build completed the two
  smoke invocations successfully. The logs report seed 42 and 819200 expected
  macro-particles on the 32³ domain, but contain no field/particle probe or
  checksum; they are lifecycle evidence, not a physics reference.
- The plugin-free `picongpu-minimal-thermal` target excludes every source under
  `include/picongpu/plugins/`; its generated compile commands contain no plugin
  translation unit. It retains the normal `SimulationStarter`/`PluginConnector`
  lifecycle, whose registry is empty because registration translation units are
  absent. Checkpoint/restart CLI support is compile-time disabled for this target.
- The CPU minimal build reaches core PIConGPU migration errors in `EMFieldBase`,
  FDTD, particle, atomic-physics, and runtime-density headers after the isolated
  source set is accepted. No PIConGPU run, legacy probe capture, or
  physics/reference acceptance has been performed. The current driver was
  checked only with a mock executable (CTest registration), not with PIConGPU.
