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

## Focused field-buffer test

The `field-buffer/` subdirectory is a standalone CMake project that exercises
`include/picongpu/fields/detail/FieldBufferOperations.hpp` with a real
`pmacc::GridBuffer<pmacc::math::Vector<float, 3>, 3>`. It links only PMacc, the
repository Catch2, and the test, so it does not depend on the still-unmigrated
application translation units:

```bash
cmake -S /path/to/picongpu/share/picongpu/tests/CaravanThermal/field-buffer \
  -B /tmp/picongpu-field-buffer-test \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPMacc_DIR=/path/to/picongpu/include/pmacc \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -Dalpaka_ACC_GPU_CUDA_ENABLE=OFF \
  -Dalpaka_ACC_GPU_HIP_ENABLE=OFF
cmake --build /tmp/picongpu-field-buffer-test --target picongpu-field-buffer-test -j4
ctest --test-dir /tmp/picongpu-field-buffer-test --output-on-failure
```

Recorded 2026-09-22 on the HAL CPU Debug profile: `ctest` passed 2/2 (a
single-rank case and a two-rank periodic case); the direct run reported 313
assertions in 8 test cases on each of 1 and 2 ranks. The storage cases cover
pending-predecessor non-mutation, capacity-restoring zero reset with a chained
download, and nonzero upload/download round-trip. The communication cases cover
scatter and gather each depending independently on producer, scatter tail, and
gather tail (also with no exchanges configured); and an alternating
scatter→gather→scatter sequence on aliased device storage that is ordered by the
stored tails alone and compared against serialized execution, with a separate
value oracle for the intermediate `17`/`17`/`136` results. A failed-predecessor
case is not included because this revision only distinguishes pending from ready
completion and treats submission/connection failures as fatal.

## Compile-time isolation checks

Two standalone, build-independent drivers validate minimal-mode exclusions with
sanitized compile commands (`-c`/`-o`/`-M*` stripped; only `-E -H` or
`-fsyntax-only`). They require an already-configured minimal build directory
with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`; a successful application build is not
needed:

- `atomic-negative/check_atomic_isolation.py` — atomic-physics/IPD isolation and
  four-tag rejection.
- `runtime-density/check_runtime_density_isolation.py` — runtime-density
  openPMD exclusion, instantiation rejection, and unused-alias tolerance.
- `incident-field/check_incident_field_isolation.py` — incident-field
  `FromOpenPMDPulse` openPMD exclusion and selection rejection.
- `ionization/check_ionization_isolation.py` — ThomasFermi by-collision
  ionization exclusion and single/list ionizer rejection.
- `absorber-policy/` — standalone policy test (minimal + ordinary) and
  `check_pml_isolation.py` for PML exclusion/retention.

Each directory has its own README with usage and the exact checks.

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
- The CPU minimal application build is still blocked, but the collision/derived-field
  and particle-to-grid errors are now compile-time excluded in minimal mode:
  `simulation/stage/Collision.x.cpp` guards those includes and rejects nonempty
  `CollisionPipeline`/`CollisionScreeningSpecies` at compile time. Remaining failures
  are the FDTD transaction graph, `Particles.tpp` synchronization overrides, PML
  `Field`, atomic-physics `SuperCellField`/`StewartPyattIPD`, incident-field
  `FromOpenPMDPulse`, runtime-density `FromOpenPMDImpl`, and the deferred
  `FieldJ::assign` blocker. Because the link never completes, this is not evidence
  that the migrated `EMFieldBase`/`FieldJ`/`FieldTmp` translation units build in
  context. No PIConGPU run, legacy probe capture, or physics/reference acceptance
  has been performed. The smoke driver was checked only with a mock executable
  (CTest registration), not with PIConGPU.
