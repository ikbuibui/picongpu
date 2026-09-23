# Minimal-mode ionization (ThomasFermi) isolation check

Reproducible validation for excluding the unmigrated Thomas-Fermi by-collision
ionization implementation from the `PICONGPU_MINIMAL_CARAVAN_THERMAL` target,
while still rejecting any configuration that selects an ionizer.

The by-field models (ADK/BSI/Keldysh) are not modified, but in minimal mode their
implementation includes are equally excluded and any species selecting them is
rejected.

The rejection is based on the `ionizers<>` flag itself: any species carrying
that flag - **including one with an explicitly empty ionizer list** - fails to
compile. The policy does not inspect the resolved ionizer-list contents.

## Contents

- `include-thomasfermi/include/picongpu/param/speciesDefinition.param` - negative
  fixture with `ionizers<MakeSeq_t<ThomasFermi<NegDest>>>` on the electron
  species. The synthetic `NegDest` and the unrelated second ionizer in the list
  fixture exercise **flag-based rejection**, not a valid ThomasFermi physics
  configuration or its execution.
- `include-thomasfermi-list/include/picongpu/param/speciesDefinition.param` -
  negative fixture with ThomasFermi nested in a multi-element ionizer list.
- `check_ionization_isolation.py` - driver.

## Usage

The build directory must already be configured with
`-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON` and
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. A successful application build is not
required - the checks are preprocessing and syntax only.

```bash
./check_ionization_isolation.py <build> [<log-dir>]
```

The driver strips `-c`/`-o` (and any `-M*` options), replaces the
translation-unit argument with a probe, and injects a fixture include path for
the rejection checks. It never rewrites target object files. Diagnostics are
saved under `<log-dir>` (default `<build>/ionization-logs`).

## What is checked

1. **Minimal exclusion.** Preprocessing `ParticleIonization.x.cpp` in minimal
   mode must not include `ThomasFermi_Impl.hpp`; the ordinary target must.
2. **Thermal policy.** The Thermal configuration (no ionizers) must not fire the
   rejection assertion and must have no `ThomasFermi` diagnostics. This is
   partial isolation evidence only: the TU still exits **1** with eight
   unrelated PML/Particles errors, so it is **not** a passing compilation. The
   clean exit-0 test is the unused-alias probe.
3. **Single-ionizer rejection.** The single-ionizer fixture must fail to compile
   and emit an actual `error: static assertion failed:` diagnostic containing
   `PICONGPU_MINIMAL_CARAVAN_THERMAL does not support configured ionization`.
4. **List rejection.** The same must hold when ThomasFermi is nested inside a
   multi-element ionizer list.
5. **Unused-alias tolerance.** Naming, but not configuring,
   `particles::ionization::ThomasFermi<T>` must compile with exit code 0.

The driver exits `0` only if all checks pass.
