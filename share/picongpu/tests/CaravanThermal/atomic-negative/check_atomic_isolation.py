#!/usr/bin/env python3
"""Reproducible validation of minimal-mode atomic-physics/IPD isolation.

Usage:
    check_atomic_isolation.py <configured-thermal-build-dir> [log-dir]

The build directory must already be configured (for example with
-DPICONGPU_MINIMAL_CARAVAN_THERMAL=ON and CMAKE_EXPORT_COMPILE_COMMANDS=ON)
and contain compile_commands.json. A successful application build is not
required: all checks are preprocessing/syntax only.

Checks
------
1. Positive include isolation: preprocessing the minimal AtomicPhysics.x.cpp
   translation unit with -E -H must not open any atomic-physics/IPD
   implementation header, while the ordinary (non-minimal) translation unit
   must.
2. Negative rejection: the fixture in include-neg/ attaches an
   atomicPhysicsParticle<> flag for each of the four supported tags. Under
   minimal mode each must fail with an actual
   `error: static assertion failed:` diagnostic containing the expected text.

No object files are read or written: -c and -o are stripped and only
preprocessing (-E -H) or -fsyntax-only are used. Each case's full diagnostics
are saved under <log-dir> (default: <build-dir>/atomic-negative-logs).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "include-neg")

ATOMIC_HEADER_RE = re.compile(r"(StewartPyattIPD|SuperCellField)\.hpp|ionizationPotentialDepression")

# The four supported atomicPhysicsParticle tags and the diagnostic each must produce.
NEGATIVE_CASES = [
    ("NEG_EXPECT_ION", "does not support atomic-physics ion species"),
    ("NEG_EXPECT_ELECTRON", "does not support atomic-physics electron species"),
    ("NEG_EXPECT_ONLYIPDION", "does not support only-IPD ion species"),
    ("NEG_EXPECT_ONLYIPDEELECTRON", "does not support only-IPD electron species"),
]

# Drop options that would write an object or dependency file.
DEP_OPTIONS_WITH_ARG = ("-MF", "-MT", "-MQ")
DEP_OPTIONS_NO_ARG = ("-M", "-MM", "-MD", "-MMD", "-MG", "-MP")

# A real diagnostic, not just the expected substring appearing anywhere (e.g. in an
# include trace or an unrelated message).
STATIC_ASSERT_RE = re.compile(r"error: static assertion failed:")


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
    # Tolerate a leading `cd <dir> &&`.
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


def run(args: list[str], build_dir: str, *, capture_stdout: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=build_dir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def trace_headers(args: list[str], build_dir: str, log_path: str) -> set[str]:
    """Return the set of atomic implementation headers actually opened at preprocessing.

    Uses -E -H so the compiler only preprocesses: template/override errors in the
    tree cannot turn into spurious "headers" and stdout is discarded. Only the
    include trace on stderr (lines beginning with dots) is parsed.
    """
    completed = run(args + ["-E", "-H"], build_dir, capture_stdout=False)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"preprocessing failed (rc={completed.returncode}); see {log_path}")
    headers: set[str] = set()
    for line in completed.stderr.splitlines():
        match = re.match(r"^\.+ (.*)$", line)
        if not match:
            continue
        path = match.group(1).strip()
        if ATOMIC_HEADER_RE.search(path):
            headers.add(os.path.normpath(path))
    return headers


def inject_fixture(args: list[str], define: str) -> list[str]:
    args = list(args)
    first_include = next(i for i, arg in enumerate(args) if arg.startswith("-I"))
    args[first_include:first_include] = [f"-I{FIXTURE}", f"-D{define}"]
    return args


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    build_dir = os.path.abspath(sys.argv[1])
    log_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) == 3 else os.path.join(build_dir, "atomic-negative-logs")
    os.makedirs(log_dir, exist_ok=True)

    failures: list[str] = []
    minimal = load_command(build_dir, want_minimal=True)
    ordinary = load_command(build_dir, want_minimal=False)

    minimal_hits = trace_headers(minimal, build_dir, os.path.join(log_dir, "positive-minimal-h.log"))
    ordinary_hits = trace_headers(ordinary, build_dir, os.path.join(log_dir, "positive-ordinary-h.log"))
    print(f"[positive] minimal atomic implementation headers: {len(minimal_hits)} {sorted(minimal_hits)}")
    print(f"[positive] ordinary atomic implementation headers: {len(ordinary_hits)} {sorted(ordinary_hits)}")
    if minimal_hits:
        failures.append("minimal TU still opens atomic implementation headers:\n  " + "\n  ".join(sorted(minimal_hits)))
    if not ordinary_hits:
        failures.append("ordinary TU no longer opens atomic implementation headers (guard regression)")

    for define, expected in NEGATIVE_CASES:
        log_path = os.path.join(log_dir, f"negative-{define}.log")
        args = inject_fixture(minimal, define) + ["-fsyntax-only"]
        completed = run(args, build_dir)
        output = completed.stdout + completed.stderr
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(output)
        same_line = any(
            STATIC_ASSERT_RE.search(line) and expected in line for line in output.splitlines()
        )
        print(
            f"[negative] {define}: exit={completed.returncode} "
            f"static-assert-diagnostic={'yes' if same_line else 'NO'} log={log_path}"
        )
        if completed.returncode == 0:
            failures.append(f"{define}: unsupported configuration compiled successfully")
        if not same_line:
            failures.append(
                f"{define}: no 'error: static assertion failed:' containing {expected!r} (see {log_path})"
            )

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print("  - " + failure)
        return 1
    print("\nOK: atomic-physics/IPD isolation and four-tag rejection verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
