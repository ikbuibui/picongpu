# Minimal-mode incident-field `FromOpenPMDPulse` isolation check

Reproducible validation for excluding the unmigrated incident-field-from-openPMD
pulse implementation from the `PICONGPU_MINIMAL_CARAVAN_THERMAL` target, while
still rejecting any configuration that selects it.

Other incident-field profiles (Free, Gaussian, PlaneWave, ...) are intentionally
untouched by this slice.

## Contents

- `include-fromopenpmd/include/picongpu/param/incidentField.param` — negative
  fixture equivalent to the default parameter file but with
  `XMin = profiles::FromOpenPMDPulse<>`.
- `include-fromopenpmd-list/include/picongpu/param/incidentField.param` — negative
  fixture with `XMin = pmacc::MakeSeq_t<profiles::None, profiles::FromOpenPMDPulse<>>`,
  exercising selection inside a profile list.
- `check_incident_field_isolation.py` — driver.

## Usage

The build directory must already be configured with
`-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON` and
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. A successful application build is not
required — the checks are preprocessing and syntax only.

```bash
./check_incident_field_isolation.py <build> [<log-dir>]
```

The driver strips `-c`/`-o` (and any `-M*` options), replaces the
translation-unit argument with a probe, and injects the fixture include path
for the selection check. It never rewrites target object files. Full
diagnostics are saved under `<log-dir>` (default
`<build>/incident-field-logs`).

## What is checked

1. **Minimal exclusion.** Preprocessing a probe that includes
   `FromOpenPMDPulse.hpp` in minimal mode must not open
   `openPMD/openPMD.hpp`; the ordinary target must open it.
2. **Unused-alias tolerance.** Naming, but not selecting,
   `profiles::FromOpenPMDPulse<>` must compile with exit code 0 in minimal
   mode.
3. **Thermal configuration.** The default incident-field parameter file must
   pass the `EnabledProfiles` rejection policy both with `ENABLE_OPENPMD=1` and
   with `ENABLE_OPENPMD=0` (exit code 0 in both).
4. **Selection rejection (single profile).** With the fixture selecting
   `FromOpenPMDPulse<>` on the active `XMin` boundary, `EnabledProfiles.hpp`
   must fail to compile and emit an actual `error: static assertion failed:`
   diagnostic containing
   `PICONGPU_MINIMAL_CARAVAN_THERMAL does not support the FromOpenPMDPulse incident-field profile`.
5. **Selection rejection (profile list).** The same must hold when
   `FromOpenPMDPulse<>` is nested inside a `pmacc::MakeSeq_t` on `XMin`, i.e.
   the flattened selection is detected.

The driver exits `0` only if all checks pass.
