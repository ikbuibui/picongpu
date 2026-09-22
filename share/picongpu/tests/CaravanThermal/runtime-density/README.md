# Minimal-mode runtime-density openPMD isolation check

Reproducible validation for excluding the unmigrated runtime-density-from-openPMD
implementation from the `PICONGPU_MINIMAL_CARAVAN_THERMAL` target. Incident-field
openPMD profiles are intentionally left in place by this slice.

## Contents

- `check_runtime_density_isolation.py` — driver. It generates four small probe
  translation units (in the log directory) and runs them with the sanitized
  compile command of the minimal/ordinary `AtomicPhysics.x.cpp` entry from
  `compile_commands.json`.

## Usage

The build directory must already be configured with
`-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON` and
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. A successful application build is not
required — the checks are preprocessing and syntax only.

```bash
./check_runtime_density_isolation.py <build> [<log-dir>]
```

The driver strips `-c`/`-o` (and any `-M*` options) and replaces the
translation-unit argument with the probe path, so it never rewrites target
object files. Full diagnostics are saved under `<log-dir>` (default
`<build>/runtime-density-logs`).

## What is checked

1. **Minimal exclusion.** Preprocessing a probe that includes
   `particles/densityProfiles/FromOpenPMDImpl.hpp` in minimal mode must not open
   `openPMD/openPMD.hpp`.
2. **Ordinary retention.** The same probe in the ordinary target must open
   `openPMD/openPMD.hpp`.
3. **Unused-alias tolerance.** A focused probe that includes
   `FromOpenPMDImpl.def` + `FromOpenPMDImpl.hpp`, defines `ProbeParam`, and
   names `FromOpenPMDImpl<ProbeParam>` without instantiating it must compile
   with **exit code 0** (it deliberately avoids `particles/param.hpp`, which
   still pulls unrelated blockers).
4. **Instantiation rejection.** The same focused probe plus
   `sizeof(FromOpenPMDImpl<ProbeParam>)` must **both** exit nonzero **and**
   emit an actual `error: static assertion failed:` diagnostic containing
   `PICONGPU_MINIMAL_CARAVAN_THERMAL does not support the FromOpenPMD density profile`.
5. **CLI guard.** Preprocessing a probe that includes
   `simulation/stage/RuntimeDensityFile.hpp` must succeed and the
   `_runtimeDensityFile` registration literal must be absent in minimal mode and
   present in the ordinary target. This is preprocessing evidence, not an
   application CLI run.

The driver exits `0` only if all checks pass.
