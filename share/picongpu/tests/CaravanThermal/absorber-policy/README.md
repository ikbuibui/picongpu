# Minimal-mode field-absorber / PML isolation check

Validates the periodic-only, no-absorber policy of the
`PICONGPU_MINIMAL_CARAVAN_THERMAL` target and proves that the unmigrated PML
implementation is excluded from its translation units.

## Contents

- `absorberPolicyTest.cpp` — helper test for the policy in
  `include/picongpu/fields/absorber/AbsorberPolicy.hpp`. Built as
  `absorber-policy-{minimal,ordinary}-test`; no PMacc/alpaka/MPI link is needed.
- `absorberFactoryTest.cpp` — integration test for the production
  `AbsorberFactory::setKind()`/`getKind()`. Built as
  `absorber-factory-{minimal,ordinary}-test` and linked against PMacc (alpaka
  defines), mirroring the field-buffer test setup.
- `CMakeLists.txt` — builds all four test executables.
- `check_pml_isolation.py` — preprocessing check against a configured minimal
  build.

## Usage

```bash
cmake -S <picongpu-source>/share/picongpu/tests/CaravanThermal/absorber-policy \
  -B /tmp/picongpu-absorber-policy-test -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/picongpu-absorber-policy-test -j4
ctest --test-dir /tmp/picongpu-absorber-policy-test --output-on-failure

python3 check_pml_isolation.py <configured-minimal-build> [<log-dir>]
```

The build directory for the Python check must already be configured with
`-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON` and
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.

## What is checked

Policy helper test, minimal build:

- all-periodic 3D, and all-active-periodic 2D with an inactive false z flag,
  resolve to `None` for requested `None`, `Pml`, and `Exponential`;
- each active non-periodic axis is rejected independently, exercised with
  requested `None`, and the rejection reports the periodicity policy message;
- invalid enum values are rejected.

Policy helper test, ordinary build (unchanged behavior):

- all-periodic (3D and 2D) resolves to `None`; a non-periodic active axis keeps
  the requested kind (`None`/`Pml`/`Exponential`); invalid enum values rejected.

Factory integration test (minimal and ordinary):

- minimal: `setKind(None)` succeeds; `setKind(Pml)`, `setKind(Exponential)`, and
  an invalid enum value all throw and leave the installed kind unchanged;
- ordinary: all three valid kinds install; an invalid enum value throws and
  leaves the installed kind unchanged.

PML preprocessing check (for both `AbsorberImpl.x.cpp` and `EMFieldBase.x.cpp`):

- minimal preprocessing must not open `fields/absorber/pml/Pml.hpp` or
  `fields/absorber/pml/Field.hpp`;
- ordinary preprocessing must open them;
- the `"fieldAbsorber"` CLI registration literal must be absent in minimal
  preprocessing and present in ordinary preprocessing.
