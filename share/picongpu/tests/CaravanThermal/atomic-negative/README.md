# Minimal-mode atomic-physics / IPD isolation check

This directory holds the reproducible validation for the compile-time exclusion
of the unmigrated atomic-physics / ionization-potential-depression (IPD)
implementation from the `PICONGPU_MINIMAL_CARAVAN_THERMAL` target.

## Contents

- `include-neg/picongpu/param/speciesDefinition.param` - a deliberately
  unsupported species definition. It attaches an `atomicPhysicsParticle<>`
  flag to a species; the four `<tag>` variants are selected with the
  `NEG_EXPECT_{ION,ELECTRON,ONLYIPDION,ONLYIPDEELECTRON}` preprocessor macros.
  For the `ION` case an additional Electron-tagged species is present so the
  pre-existing `AtomicPhysics.hpp` "at least one electron" assertion does not
  mask the rejection under test.
- `check_atomic_isolation.py` - the driver.

## Usage

The build directory must already be configured (this script does not run
CMake) with `-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON` and
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` so that `compile_commands.json` exists. A
fully built application is **not** required; the checks are preprocessing and
syntax only.

```bash
# Example: configure the CPU minimal target (does not have to build).
cmake -S <picongpu-source>/include/picongpu -B <build> \
  -DPIC_EXTENSION_PATH=<picongpu-source>/share/picongpu/benchmarks/Thermal \
  -DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=ON

./check_atomic_isolation.py <build> [<log-dir>]
```

The driver strips `-c`/`-o` (and any `-M*` dependency options) from the
recorded compile command. It runs `-E -H` for the include-isolation check and
`-fsyntax-only` for the rejection check, so it never rewrites target object
files. Full diagnostics are written under `<log-dir>` (default
`<build>/atomic-negative-logs`).

## What is checked

1. **Positive include isolation.** Preprocessing the minimal
   `AtomicPhysics.x.cpp` translation unit (`-E -H`, stdout discarded) must not
   open `SuperCellField.hpp` or `StewartPyattIPD.hpp`/`ionizationPotentialDepression`;
   the ordinary translation unit must open them. The check parses actual
   include-trace lines and deduplicates paths.
2. **Negative rejection.** Each `NEG_EXPECT_*` case must fail to compile and
   emit an actual `error: static assertion failed:` diagnostic containing the
   expected text, for example
   `PICONGPU_MINIMAL_CARAVAN_THERMAL does not support only-IPD ion species`.
   A nonzero exit or the bare substring is not accepted, because the tree
   still has unrelated transitive blockers.

The driver exits `0` only if every positive and negative check passes.
