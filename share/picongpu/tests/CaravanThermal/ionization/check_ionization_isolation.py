#!/usr/bin/env python3
"""Reproducible validation of minimal-mode ionization (ThomasFermi) isolation.

Usage:
    check_ionization_isolation.py <configured-thermal-build-dir> [log-dir]

The build directory must already be configured with
-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON and CMAKE_EXPORT_COMPILE_COMMANDS=ON.
A successful application build is not required.

Checks
------
1. Minimal preprocessing of the ParticleIonization translation unit does not
   include the ThomasFermi implementation; ordinary preprocessing does.
2. The Thermal configuration (no ionizers) passes the rejection policy: the
   intended static-assert diagnostic is absent (the TU still fails on unrelated
   PML/Particles blockers).
3. A configured ThomasFermi ionizer fails to compile and emits the intended
   `error: static assertion failed:` diagnostic.
4. The same holds when ThomasFermi is nested inside a multi-element ionizer
   list.
5. Naming, but not configuring, ThomasFermi compiles cleanly.

Compile commands are sanitized: -c/-o and any -M* options are stripped; only
-E -H or -fsyntax-only are used. Full diagnostics are saved under <log-dir>
(default: <build>/ionization-logs).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "include-thomasfermi", "include")
FIXTURE_LIST = os.path.join(HERE, "include-thomasfermi-list", "include")

EXPECTED = "PICONGPU_MINIMAL_CARAVAN_THERMAL does not support configured ionization"
THOMAS_FERMI_IMPL = "ThomasFermi_Impl.hpp"
STATIC_ASSERT_RE = re.compile(r"error: static assertion failed:")

DEP_OPTIONS_WITH_ARG = ("-MF", "-MT", "-MQ")
DEP_OPTIONS_NO_ARG = ("-M", "-MM", "-MD", "-MMD", "-MG", "-MP")

UNUSED_ALIAS_PROBE = (
    '#include "picongpu/particles/ionization/byCollision/collisionalIonizationCalc.def"\n'
    '#include "picongpu/particles/ionization/byCollision/ionizers.def"\n'
    "struct DummyDest {};\n"
    "using UnusedAlias = picongpu::particles::ionization::ThomasFermi<DummyDest>;\n"
)


def load_command(build_dir: str, want_minimal: bool) -> list[str]:
    with open(os.path.join(build_dir, "compile_commands.json"), encoding="utf-8") as handle:
        database = json.load(handle)
    for entry in database:
        command = entry.get("command", "")
        if not entry["file"].endswith("ParticleIonization.x.cpp"):
            continue
        if ("MINIMAL" in command) == want_minimal:
            return clean_args(command)
    raise SystemExit(f"no {'minimal' if want_minimal else 'ordinary'} ParticleIonization.x.cpp command in {build_dir}")


def clean_args(command: str) -> list[str]:
    args = shlex.split(command)
    if args and args[0] == "cd":
        args = args[args.index("&&") + 1 :]
    cleaned: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-c", "-o"):
            index += 2 if arg == "-o" else 1
            continue
        if arg in DEP_OPTIONS_WITH_ARG:
            index += 2
            continue
        if arg in DEP_OPTIONS_NO_ARG:
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return cleaned


def replace_source(args: list[str], probe: str) -> list[str]:
    for index in range(len(args) - 1, -1, -1):
        if args[index].endswith(".cpp"):
            return args[:index] + [probe] + args[index + 1 :]
    raise SystemExit("could not find the translation-unit argument in the compile command")


def prepend_include(args: list[str], include_dir: str) -> list[str]:
    args = list(args)
    first_include = next(i for i, arg in enumerate(args) if arg.startswith("-I"))
    args.insert(first_include, f"-I{include_dir}")
    return args


def run(args: list[str], build_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=build_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def trace_has(args: list[str], build_dir: str, log_path: str, needle: str) -> bool:
    completed = run(args + ["-E", "-H"], build_dir)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"preprocessing failed (rc={completed.returncode}); see {log_path}")
    return any(re.match(r"^\.+ .*" + re.escape(needle), line) for line in completed.stderr.splitlines())


def has_static_assert(output: str, expected: str) -> bool:
    return any(STATIC_ASSERT_RE.search(line) and expected in line for line in output.splitlines())


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    build_dir = os.path.abspath(sys.argv[1])
    log_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else os.path.join(build_dir, "ionization-logs")
    os.makedirs(log_dir, exist_ok=True)

    unused_probe = os.path.join(log_dir, "probe-unused-alias.cpp")
    with open(unused_probe, "w", encoding="utf-8") as handle:
        handle.write(UNUSED_ALIAS_PROBE)

    minimal = load_command(build_dir, want_minimal=True)
    ordinary = load_command(build_dir, want_minimal=False)
    failures: list[str] = []

    minimal_has = trace_has(minimal, build_dir, os.path.join(log_dir, "positive-minimal-h.log"), THOMAS_FERMI_IMPL)
    ordinary_has = trace_has(ordinary, build_dir, os.path.join(log_dir, "positive-ordinary-h.log"), THOMAS_FERMI_IMPL)
    print(f"[positive] minimal TU includes {THOMAS_FERMI_IMPL}: {minimal_has}")
    print(f"[positive] ordinary TU includes {THOMAS_FERMI_IMPL}: {ordinary_has}")
    if minimal_has:
        failures.append(f"minimal ParticleIonization TU still includes {THOMAS_FERMI_IMPL}")
    if not ordinary_has:
        failures.append(f"ordinary ParticleIonization TU no longer includes {THOMAS_FERMI_IMPL} (guard regression)")

    thermal = run(minimal + ["-fsyntax-only"], build_dir)
    thermal_output = thermal.stdout + thermal.stderr
    with open(os.path.join(log_dir, "thermal-config.log"), "w", encoding="utf-8") as handle:
        handle.write(thermal_output)
    thermal_fired = has_static_assert(thermal_output, EXPECTED)
    thermal_thomas_fermi_errors = thermal_output.count("ThomasFermi")
    print(
        f"[positive] Thermal config: exit={thermal.returncode} rejection-fired={thermal_fired} "
        f"ThomasFermi-diagnostics={thermal_thomas_fermi_errors} (unrelated blockers expected)"
    )
    if thermal_fired:
        failures.append("Thermal configuration incorrectly fired the ionization rejection")
    if thermal_thomas_fermi_errors:
        failures.append("Thermal TU still has ThomasFermi diagnostics")

    for name, fixture in (("single", FIXTURE), ("list", FIXTURE_LIST)):
        args = prepend_include(minimal, fixture) + ["-fsyntax-only"]
        completed = run(args, build_dir)
        output = completed.stdout + completed.stderr
        with open(os.path.join(log_dir, f"configured-{name}.log"), "w", encoding="utf-8") as handle:
            handle.write(output)
        found = has_static_assert(output, EXPECTED)
        print(
            f"[negative] configured ThomasFermi ({name}): exit={completed.returncode} "
            f"static-assert-diagnostic={'yes' if found else 'NO'}"
        )
        if completed.returncode == 0:
            failures.append(f"configured ThomasFermi ({name}) compiled successfully in minimal mode")
        if not found:
            failures.append(f"configured ThomasFermi ({name}) did not emit the intended diagnostic")

    unused = run(replace_source(minimal, unused_probe) + ["-fsyntax-only"], build_dir)
    with open(os.path.join(log_dir, "unused-alias.log"), "w", encoding="utf-8") as handle:
        handle.write(unused.stdout + unused.stderr)
    print(f"[positive] minimal unused ThomasFermi alias compiles cleanly: exit={unused.returncode}")
    if unused.returncode != 0:
        failures.append("naming (not configuring) ThomasFermi did not compile cleanly in minimal mode")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("\nOK: ionization isolation, Thermal policy, single/list rejection, and unused-alias tolerance verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
