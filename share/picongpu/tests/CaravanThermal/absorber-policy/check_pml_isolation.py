#!/usr/bin/env python3
"""Reproducible preprocessing check for minimal-mode PML exclusion.

Usage:
    check_pml_isolation.py <configured-thermal-build-dir> [log-dir]

The build directory must already be configured with
-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON and CMAKE_EXPORT_COMPILE_COMMANDS=ON.
A successful application build is not required.

Checks, for both the absorber implementation TU and a field TU:
1. Minimal preprocessing must not open `fields/absorber/pml/Pml.hpp` or
   `fields/absorber/pml/Field.hpp`.
2. Ordinary preprocessing must open them (ordinary target unchanged).
3. The `"fieldAbsorber"` CLI registration literal must be absent in minimal
   preprocessing and present in ordinary preprocessing.

Only `-E -H` (include traces) and `-E` (CLI literal) are used, with -c/-o and
any -M* options stripped, so no object files are touched. Traces are saved under
<log-dir>.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

DEP_OPTIONS_WITH_ARG = ("-MF", "-MT", "-MQ")
DEP_OPTIONS_NO_ARG = ("-M", "-MM", "-MD", "-MMD", "-MG", "-MP")
PML_HEADERS = re.compile(r"fields/absorber/pml/(Pml|Field)\.hpp")


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


def load_command(build_dir: str, tu_suffix: str, want_minimal: bool) -> list[str]:
    with open(os.path.join(build_dir, "compile_commands.json"), encoding="utf-8") as handle:
        database = json.load(handle)
    for entry in database:
        command = entry.get("command", "")
        if not entry["file"].endswith(tu_suffix):
            continue
        if ("MINIMAL" in command) == want_minimal:
            return clean_args(command)
    raise SystemExit(f"no {'minimal' if want_minimal else 'ordinary'} {tu_suffix} command in {build_dir}")


def replace_source(args: list[str], probe: str) -> list[str]:
    for index in range(len(args) - 1, -1, -1):
        if args[index].endswith(".cpp"):
            return args[:index] + [probe] + args[index + 1 :]
    raise SystemExit("could not find the translation-unit argument in the compile command")


def cli_literal_present(args: list[str], build_dir: str, log_path: str) -> tuple[int, bool]:
    completed = subprocess.run(
        args + ["-E"],
        cwd=build_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(completed.stdout + completed.stderr)
    return completed.returncode, '"fieldAbsorber"' in completed.stdout


def pml_headers(args: list[str], build_dir: str, log_path: str) -> set[str]:
    completed = subprocess.run(
        args + ["-E", "-H"],
        cwd=build_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"preprocessing failed (rc={completed.returncode}); see {log_path}")
    headers: set[str] = set()
    for line in completed.stderr.splitlines():
        match = re.match(r"^\.+ (.*)$", line)
        if match and PML_HEADERS.search(match.group(1)):
            headers.add(os.path.basename(match.group(1)))
    return headers


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    build_dir = os.path.abspath(sys.argv[1])
    log_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else os.path.join(build_dir, "pml-isolation-logs")
    os.makedirs(log_dir, exist_ok=True)

    failures: list[str] = []
    for label, suffix in (("absorber-impl", "AbsorberImpl.x.cpp"), ("field", "EMFieldBase.x.cpp")):
        minimal = load_command(build_dir, suffix, want_minimal=True)
        ordinary = load_command(build_dir, suffix, want_minimal=False)
        minimal_hits = pml_headers(minimal, build_dir, os.path.join(log_dir, f"{label}-minimal-h.log"))
        ordinary_hits = pml_headers(ordinary, build_dir, os.path.join(log_dir, f"{label}-ordinary-h.log"))
        print(f"[{label}] minimal PML headers: {sorted(minimal_hits)}")
        print(f"[{label}] ordinary PML headers: {sorted(ordinary_hits)}")
        if minimal_hits:
            failures.append(f"{label}: minimal TU still opens PML headers {sorted(minimal_hits)}")
        if not ordinary_hits:
            failures.append(f"{label}: ordinary TU no longer opens PML headers (guard regression)")

    # CLI registration: minimal must not advertise --fieldAbsorber; ordinary must.
    probe = os.path.join(log_dir, "probe-fieldabsorber.cpp")
    with open(probe, "w", encoding="utf-8") as handle:
        handle.write('#include "picongpu/simulation/stage/FieldAbsorber.hpp"\n')
    base_minimal = load_command(build_dir, "AbsorberImpl.x.cpp", want_minimal=True)
    base_ordinary = load_command(build_dir, "AbsorberImpl.x.cpp", want_minimal=False)
    minimal_rc, minimal_present = cli_literal_present(
        replace_source(base_minimal, probe), build_dir, os.path.join(log_dir, "cli-minimal.log")
    )
    ordinary_rc, ordinary_present = cli_literal_present(
        replace_source(base_ordinary, probe), build_dir, os.path.join(log_dir, "cli-ordinary.log")
    )
    print(f"[cli] minimal fieldAbsorber registration: rc={minimal_rc} present={minimal_present}")
    print(f"[cli] ordinary fieldAbsorber registration: rc={ordinary_rc} present={ordinary_present}")
    if minimal_rc != 0 or ordinary_rc != 0:
        failures.append("fieldAbsorber CLI preprocessing did not succeed")
    if minimal_present:
        failures.append("minimal mode still advertises the fieldAbsorber CLI option")
    if not ordinary_present:
        failures.append("ordinary mode no longer advertises the fieldAbsorber CLI option")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("\nOK: minimal excludes PML; ordinary retains it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
