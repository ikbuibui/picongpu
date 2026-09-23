#!/usr/bin/env python3
"""Reproducible validation of minimal-mode runtime-density openPMD isolation.

Usage:
    check_runtime_density_isolation.py <configured-thermal-build-dir> [log-dir]

The build directory must already be configured with
-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON and CMAKE_EXPORT_COMPILE_COMMANDS=ON.
A successful application build is not required.

Checks
------
1. Minimal preprocessing of the density implementation header does not open
   openPMD (the runtime-density loading path is excluded).
2. Ordinary preprocessing of the same header does open openPMD (the ordinary
   target is unchanged).
3. Naming (without instantiating) the profile specialization compiles cleanly
   in minimal mode, i.e. the alias declaration is still valid.
4. Instantiating the profile in minimal mode fails to compile and emits the
   intended `error: static assertion failed:` diagnostic.
5. The `<species>_runtimeDensityFile` CLI registration is preprocessed away in
   minimal mode and retained in the ordinary target.

Compile commands are sanitized: -c/-o and any -M* options are stripped; only
-E -H, -E, or -fsyntax-only are used. Full diagnostics are saved under
<log-dir> (default: <build>/runtime-density-logs).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

EXPECTED = "PICONGPU_MINIMAL_CARAVAN_THERMAL does not support the FromOpenPMD density profile"
CLI_OPTION_LITERAL = "_runtimeDensityFile"
STATIC_ASSERT_RE = re.compile(r"error: static assertion failed:")

DEP_OPTIONS_WITH_ARG = ("-MF", "-MT", "-MQ")
DEP_OPTIONS_NO_ARG = ("-M", "-MM", "-MD", "-MMD", "-MG", "-MP")

# Focused probes: include the forward declaration and the definition directly, so
# they do not depend on the Thermal parameter files.
_PROFILE_HEADERS = (
    '#include "picongpu/particles/densityProfiles/FromOpenPMDImpl.def"\n'
    '#include "picongpu/particles/densityProfiles/FromOpenPMDImpl.hpp"\n'
    "struct ProbeParam {};\n"
)
NEGATIVE_PROBE = (
    _PROFILE_HEADERS + "// Force instantiation/completeness of the profile wrapper.\n"
    'static_assert(sizeof(picongpu::densityProfiles::FromOpenPMDImpl<ProbeParam>) > 0, "");\n'
)
UNUSED_ALIAS_PROBE = (
    _PROFILE_HEADERS + "// Name the specialization without instantiating it.\n"
    "using UnusedAlias = picongpu::densityProfiles::FromOpenPMDImpl<ProbeParam>;\n"
)
OPENPMD_PROBE = '#include "picongpu/particles/densityProfiles/FromOpenPMDImpl.hpp"\n'
CLI_PROBE = '#include "picongpu/simulation/stage/RuntimeDensityFile.hpp"\n'


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


def preprocessed_contains_cli(args: list[str], build_dir: str, log_path: str) -> tuple[int, bool]:
    completed = run(args + ["-E"], build_dir)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(completed.stdout + completed.stderr)
    return completed.returncode, CLI_OPTION_LITERAL in completed.stdout


def has_static_assert(output: str, expected: str) -> bool:
    return any(STATIC_ASSERT_RE.search(line) and expected in line for line in output.splitlines())


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    build_dir = os.path.abspath(sys.argv[1])
    log_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else os.path.join(build_dir, "runtime-density-logs")
    os.makedirs(log_dir, exist_ok=True)

    probes = {
        "negative": (os.path.join(log_dir, "probe-negative.cpp"), NEGATIVE_PROBE),
        "unused": (os.path.join(log_dir, "probe-unused-alias.cpp"), UNUSED_ALIAS_PROBE),
        "openpmd": (os.path.join(log_dir, "probe-openpmd.cpp"), OPENPMD_PROBE),
        "cli": (os.path.join(log_dir, "probe-cli.cpp"), CLI_PROBE),
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
    print(f"[positive] minimal density header opens openPMD: {minimal_opens}")
    print(f"[positive] ordinary density header opens openPMD: {ordinary_opens}")
    if minimal_opens:
        failures.append("minimal FromOpenPMDImpl.hpp still opens openPMD/openPMD.hpp")
    if not ordinary_opens:
        failures.append("ordinary FromOpenPMDImpl.hpp no longer opens openPMD/openPMD.hpp (guard regression)")

    unused = run(replace_source(minimal, probes["unused"][0]) + ["-fsyntax-only"], build_dir)
    unused_output = unused.stdout + unused.stderr
    with open(os.path.join(log_dir, "unused-alias.log"), "w", encoding="utf-8") as handle:
        handle.write(unused_output)
    print(f"[positive] minimal unused alias compiles cleanly: exit={unused.returncode}")
    if unused.returncode != 0:
        failures.append("naming (not instantiating) FromOpenPMDImpl did not compile cleanly in minimal mode")

    negative = run(replace_source(minimal, probes["negative"][0]) + ["-fsyntax-only"], build_dir)
    negative_output = negative.stdout + negative.stderr
    with open(os.path.join(log_dir, "negative-instantiation.log"), "w", encoding="utf-8") as handle:
        handle.write(negative_output)
    negative_found = has_static_assert(negative_output, EXPECTED)
    print(
        f"[negative] minimal FromOpenPMD instantiation: exit={negative.returncode} "
        f"static-assert-diagnostic={'yes' if negative_found else 'NO'}"
    )
    if negative.returncode == 0:
        failures.append("instantiating FromOpenPMD compiled successfully in minimal mode")
    if not negative_found:
        failures.append(f"instantiation did not emit the intended diagnostic: {EXPECTED!r}")

    minimal_cli_rc, minimal_cli_present = preprocessed_contains_cli(
        replace_source(minimal, probes["cli"][0]), build_dir, os.path.join(log_dir, "cli-minimal.log")
    )
    ordinary_cli_rc, ordinary_cli_present = preprocessed_contains_cli(
        replace_source(ordinary, probes["cli"][0]), build_dir, os.path.join(log_dir, "cli-ordinary.log")
    )
    print(f"[positive] CLI literal in minimal preprocessing: rc={minimal_cli_rc} present={minimal_cli_present}")
    print(f"[positive] CLI literal in ordinary preprocessing: rc={ordinary_cli_rc} present={ordinary_cli_present}")
    if minimal_cli_rc != 0 or ordinary_cli_rc != 0:
        failures.append("CLI-guard preprocessing did not succeed")
    if minimal_cli_present:
        failures.append("minimal preprocessing still advertises the _runtimeDensityFile option")
    if not ordinary_cli_present:
        failures.append("ordinary preprocessing no longer advertises the _runtimeDensityFile option")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print(
        "\nOK: runtime-density openPMD isolation, instantiation rejection, "
        "unused-alias tolerance, and CLI guard verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
