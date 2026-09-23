# Thermal correctness probe (WP6) and legacy oracle (WP0)

Test-only observability for the `picongpu-minimal-thermal` Caravan migration
slice, plus a reproducible way to instrument the pre-migration checkout
(`7964ed0823`) so the two can be compared.

The probe is **not** a plugin and does not touch the ordinary `picongpu` target.
It is compiled only when `PICONGPU_THERMAL_PROBE=ON` is passed to a
`PICONGPU_MINIMAL_CARAVAN_THERMAL=ON` configure.

## What it measures

Per MPI rank, after completed initialization and after each completed step:

* owned macro-particle count over `CORE + BORDER` (guards excluded),
* weight sum, stored (macro-weighted) momentum, kinetic and total energy,
* the number of particles with a non-finite position/momentum/weight,
* owned E/B/J finite checks, sum of squares, sum of `|value|`, min and max,
* a full owned-field snapshot with global cell coordinates.

The measured unit is the PIC-internal unit (`float_X`); legacy and Caravan use
the same internal units, so no conversion is applied. E/B and B are at their
native Yee time levels at the observation point; the probe records the state
after the field post-update, exactly where the legacy plugin notification reads
the state.

## Files

| File | Purpose |
| --- | --- |
| `include/picongpu/thermalProbe/ThermalProbe.hpp` | the shared instrumentation (Caravan and legacy branches) |
| `probe/validate_probe.py` | schema/consistency validation and Caravan-vs-legacy comparison |
| `probe/legacy/` | copy of the shared header plus a patch that hooks the legacy `Simulation.hpp` |

Output layout in `$PICONGPU_THERMAL_PROBE_DIR`:

```
thermal_probe_rank<rank>.csv              step,stage,species,count,... (scalars)
thermal_snapshot_rank<rank>_step<step>.csv   gx,gy,gz,Ex,...,Jz (owned cells)
```

## Runtime configuration

| Environment variable | Effect |
| --- | --- |
| `PICONGPU_THERMAL_PROBE_DIR` | output directory; unset disables the probe |
| `PICONGPU_THERMAL_PROBE_STEPS` | comma-separated step list or `final`; default all |
| `PICONGPU_THERMAL_PROBE_SNAPSHOTS` | `0` disables full field snapshots |

`final` observes only the last configured step, which allows a run that is
quiescent until teardown (diagnostic synchronization cannot hide a race there).

## Build and run (Caravan CPU)

```bash
cmake -S include/picongpu -B /tmp/picongpu-thermal-probe-cpu \
  -C share/picongpu/tests/CaravanThermal/thermal-caravan-cache.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON \
  -DPICONGPU_THERMAL_PROBE=ON
cmake --build /tmp/picongpu-thermal-probe-cpu --target picongpu-minimal-thermal -j

PICONGPU_THERMAL_PROBE_DIR=/tmp/probe-caravan mpirun -n 1 \
  /tmp/picongpu-thermal-probe-cpu/picongpu-minimal-thermal \
  -d 1 1 1 -g 32 32 32 -s 5 --periodic 1 1 1
```

## Instrument and build the legacy oracle

```bash
share/picongpu/tests/CaravanThermal/probe/legacy/apply_legacy_probe.sh /tmp/picongpu-legacy-7964ed0823
# then configure/build with the -DPICONGPU_THERMAL_PROBE -DPICONGPU_THERMAL_PROBE_LEGACY
# flags printed by the script, and run the same command line as above.
```

Match backend, compiler, precision, parameter overwrites, grid, rank layout,
seed and initialized state between the two runs. The legacy `7964ed0823`
checkout aborts during teardown (`Manager.cpp` cooperative-task assertion) after
the probe has flushed every record; that is a pre-existing legacy teardown
defect, not a probe result. Use the written CSV files.

## Validate and compare

```bash
python3 share/picongpu/tests/CaravanThermal/probe/validate_probe.py --selftest
python3 share/picongpu/tests/CaravanThermal/probe/validate_probe.py \
  --compare /tmp/probe-caravan /tmp/probe-legacy
```

The validator rejects missing steps/ranks, malformed or non-finite values,
duplicate snapshot cells, count changes, unavailable or zero evolved J, and
scalar/field values outside tolerance. `--selftest` exercises positive and
negative fixtures so the validator itself is covered.

Tolerances:

* generated owned-particle counts are compared **exactly**;
* scalar sums use `--rel-tol` (default `1e-4`). Signed momentum sums are
  cancellation-limited in `float_X`; the observed Caravan-vs-legacy maximum is
  `3e-5`. Positive energy sums agree to `<=2e-8`;
* field cells use `--snapshot-rel-tol` / `--snapshot-abs-tol`. Independent
  stages are ordered differently by the legacy event system and by Caravan, so
  per-cell `float_X` values are not bit-identical even when totals match.
  The validator reports the largest absolute and scale-normalized deviation per
  step/component.

## Recorded results (HAL CPU Debug, 2026-09-22)

1 rank, `32x32x32`, 5 steps: counts `819200` at every observation on both sides;
the step-0 record is bit-identical; the largest field deviation is `4.7e-10`
absolute and `~6e-7` scale-normalized; total energies agree to `<=2e-8`.

2 ranks, `64x32x32`, 5 steps: global owned count exactly `1638400` every step;
the same tolerance checks pass. Per-rank counts move as particles cross the
decomposition boundary, which is expected.

## Correctness findings from this probe

The probe exposed two defects that the lifecycle smoke tests could not:

1. **The particle halo exchange never ran.** `GridBuffer::getSendMask()` and
   `Exchange` looked up the communication mask/communicator through
   `Environment<DIM>`, but the particle frame exchange buffers are
   one-dimensional (`DIM1`) while the simulation is three-dimensional, so they
   read the never-initialized `GridController<DIM1>`. `hasSendExchange()` was
   false for all 26 directions and every pushed particle that crossed a
   supercell boundary was stranded in a guard cell. The `GridBuffer`/`Exchange`
   templates now carry an explicit communicator dimension (`T_CommDim`) so the
   exchange uses the simulation topology. Legacy avoided this because its
   communicator was a process-global `EnvironmentController`.
2. **The insertion kernel was launched with the wrong count.**
   `receiveChunks()` passed the received particle-payload size to
   `KernelInsertParticles`, which is indexed by per-supercell exchange-index
   entries. This over-launched the kernel and duplicated particles
   deterministically (`819200 -> 819238` over five steps before the fix). It now
   uses `getHostCurrentSize()` and the shared `InsertNonEmptySender` field and
   `ParticlesBase::insertParticles` parameter are named `numIndexEntries`.

The first defect is why the pre-existing "GPU smoke" runs could pass without
exercising any halo communication.
