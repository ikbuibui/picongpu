#!/usr/bin/env python3
"""Reproducible validation of minimal-mode FromOpenPMDPulse isolation.

Usage:
    check_incident_field_isolation.py <configured-thermal-build-dir> [log-dir]

The build directory must already be configured with
-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON and CMAKE_EXPORT_COMPILE_COMMANDS=ON.
A successful application build is not required.

Checks
------
1. Minimal preprocessing of the pulse implementation header does not open
   openPMD; ordinary preprocessing does (ordinary target unchanged).
2. Naming (not selecting) FromOpenPMDPulse<> compiles cleanly in minimal mode.
3. The Thermal incident-field configuration passes the EnabledProfiles
   rejection policy (compiles cleanly).
4. Selecting FromOpenPMDPulse<> on an active boundary (XMin) fails to compile
   and emits the intended `error: static assertion failed:` diagnostic.

Compile commands are sanitized: -c/-o and any -M* options are stripped; only
-E -H or -fsyntax-only are used. Full diagnostics are saved under <log-dir>
(default: <build>/incident-field-logs).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "include-fromopenpmd", "include")
FIXTURE_LIST = os.path.join(HERE, "include-fromopenpmd-list", "include")

EXPECTED = "PICONGPU_MINIMAL_CARAVAN_THERMAL does not support the FromOpenPMDPulse incident-field profile"
STATIC_ASSERT_RE = re.compile(r"error: static assertion failed:")

DEP_OPTIONS_WITH_ARG = ("-MF", "-MT", "-MQ")
DEP_OPTIONS_NO_ARG = ("-M", "-MM", "-MD", "-MMD", "-MG", "-MP")

OPENPMD_PROBE = '#include "picongpu/fields/incidentField/profiles/FromOpenPMDPulse.hpp"\n'
UNUSED_ALIAS_PROBE = (
    '#include "picongpu/defines.hpp"\n'
    '#include "picongpu/fields/incidentField/profiles/profiles.def"\n'
    "using UnusedAlias = picongpu::fields::incidentField::profiles::FromOpenPMDPulse<>;\n"
)
THERMAL_PROBE = '#include "picongpu/fields/incidentField/EnabledProfiles.hpp"\n'
SELECTED_PROBE = '#include "picongpu/fields/incidentField/EnabledProfiles.hpp"\n'


def load_command(build_dir: str, want_minimal: bool) -> list[str]:
    with open(os.path.join(build_dir, "compile_commands.json"), encoding="utf-8") as handle:
        database = json.load(handle)
    for entry in database:
        command = entry.get("command", "")
        if not entry["file"].endswith("AtomicPhysics.x.cpp"):
            continue
        if ("MINIMAL" in command) == want_minimal:
            return clean_args(command)
    raise SystemExit(f"no {'minimal' if want_minimal else 'ordinary'} AtomicPhysics.x.cpp command in {build_dir}")


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


def override_define(args: list[str], name: str, value: str) -> list[str]:
    """Replace any existing -D<name>[=...] with -D<name>=<value>."""
    prefix = f"-D{name}"
    filtered = [arg for arg in args if not (arg == prefix or arg.startswith(prefix + "="))]
    filtered.append(f"-D{name}={value}")
    return filtered


def run(args: list[str], build_dir: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=build_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def opens_openpmd(args: list[str], build_dir: str, log_path: str) -> bool:
    completed = run(args + ["-E", "-H"], build_dir)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"preprocessing failed (rc={completed.returncode}); see {log_path}")
    return any(re.match(r"^\.+ .*openPMD/openPMD\.hpp", line) for line in completed.stderr.splitlines())


def has_static_assert(output: str, expected: str) -> bool:
    return any(STATIC_ASSERT_RE.search(line) and expected in line for line in output.splitlines())


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    build_dir = os.path.abspath(sys.argv[1])
    log_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else os.path.join(build_dir, "incident-field-logs")
    os.makedirs(log_dir, exist_ok=True)

    probes = {
        "openpmd": (os.path.join(log_dir, "probe-openpmd.cpp"), OPENPMD_PROBE),
        "unused": (os.path.join(log_dir, "probe-unused-alias.cpp"), UNUSED_ALIAS_PROBE),
        "thermal": (os.path.join(log_dir, "probe-thermal.cpp"), THERMAL_PROBE),
        "selected": (os.path.join(log_dir, "probe-selected.cpp"), SELECTED_PROBE),
    }
    for path, content in probes.values():
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    minimal = load_command(build_dir, want_minimal=True)
    ordinary = load_command(build_dir, want_minimal=False)
    failures: list[str] = []

    minimal_opens = opens_openpmd(
        replace_source(minimal, probes["openpmd"][0]), build_dir, os.path.join(log_dir, "positive-minimal-h.log")
    )
    ordinary_opens = opens_openpmd(
        replace_source(ordinary, probes["openpmd"][0]), build_dir, os.path.join(log_dir, "positive-ordinary-h.log")
    )
    print(f"[positive] minimal FromOpenPMDPulse.hpp opens openPMD: {minimal_opens}")
    print(f"[positive] ordinary FromOpenPMDPulse.hpp opens openPMD: {ordinary_opens}")
    if minimal_opens:
        failures.append("minimal FromOpenPMDPulse.hpp still opens openPMD/openPMD.hpp")
    if not ordinary_opens:
        failures.append("ordinary FromOpenPMDPulse.hpp no longer opens openPMD/openPMD.hpp (guard regression)")

    unused = run(replace_source(minimal, probes["unused"][0]) + ["-fsyntax-only"], build_dir)
    with open(os.path.join(log_dir, "unused-alias.log"), "w", encoding="utf-8") as handle:
        handle.write(unused.stdout + unused.stderr)
    print(f"[positive] minimal unused alias compiles cleanly: exit={unused.returncode}")
    if unused.returncode != 0:
        failures.append("naming (not selecting) FromOpenPMDPulse<> did not compile cleanly in minimal mode")

    thermal = run(replace_source(minimal, probes["thermal"][0]) + ["-fsyntax-only"], build_dir)
    with open(os.path.join(log_dir, "thermal-config.log"), "w", encoding="utf-8") as handle:
        handle.write(thermal.stdout + thermal.stderr)
    print(f"[positive] Thermal incident-field config passes rejection policy (openPMD on): exit={thermal.returncode}")
    if thermal.returncode != 0:
        failures.append("Thermal incident-field configuration failed the EnabledProfiles rejection policy")

    thermal_no_openpmd_args = override_define(replace_source(minimal, probes["thermal"][0]), "ENABLE_OPENPMD", "0")
    thermal_no_openpmd = run(thermal_no_openpmd_args + ["-fsyntax-only"], build_dir)
    with open(os.path.join(log_dir, "thermal-config-no-openpmd.log"), "w", encoding="utf-8") as handle:
        handle.write(thermal_no_openpmd.stdout + thermal_no_openpmd.stderr)
    print(
        "[positive] Thermal incident-field config passes rejection policy (openPMD off): "
        f"exit={thermal_no_openpmd.returncode}"
    )
    if thermal_no_openpmd.returncode != 0:
        failures.append("Thermal incident-field configuration failed with ENABLE_OPENPMD=0")

    selected_args = prepend_include(replace_source(minimal, probes["selected"][0]), FIXTURE)
    selected = run(selected_args + ["-fsyntax-only"], build_dir)
    selected_output = selected.stdout + selected.stderr
    with open(os.path.join(log_dir, "selected-fromopenpmd.log"), "w", encoding="utf-8") as handle:
        handle.write(selected_output)
    selected_found = has_static_assert(selected_output, EXPECTED)
    print(
        f"[negative] selected FromOpenPMDPulse<> on XMin: exit={selected.returncode} "
        f"static-assert-diagnostic={'yes' if selected_found else 'NO'}"
    )
    if selected.returncode == 0:
        failures.append("selected FromOpenPMDPulse<> compiled successfully in minimal mode")
    if not selected_found:
        failures.append(f"selection did not emit the intended diagnostic: {EXPECTED!r}")

    listed_args = prepend_include(replace_source(minimal, probes["selected"][0]), FIXTURE_LIST)
    listed = run(listed_args + ["-fsyntax-only"], build_dir)
    listed_output = listed.stdout + listed.stderr
    with open(os.path.join(log_dir, "selected-list-fromopenpmd.log"), "w", encoding="utf-8") as handle:
        handle.write(listed_output)
    listed_found = has_static_assert(listed_output, EXPECTED)
    print(
        f"[negative] FromOpenPMDPulse<> inside MakeSeq_t on XMin: exit={listed.returncode} "
        f"static-assert-diagnostic={'yes' if listed_found else 'NO'}"
    )
    if listed.returncode == 0:
        failures.append("FromOpenPMDPulse<> inside a profile list compiled successfully in minimal mode")
    if not listed_found:
        failures.append(f"profile-list selection did not emit the intended diagnostic: {EXPECTED!r}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("\nOK: FromOpenPMDPulse isolation, unused-alias tolerance, Thermal policy, and selection rejection verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
