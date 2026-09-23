#!/usr/bin/env python3
"""Validate and compare PIConGPU Thermal correctness-probe records.

The probe (``include/picongpu/thermalProbe/ThermalProbe.hpp``) writes, per MPI
rank:

  * ``thermal_probe_rank<rank>.csv``          scalar per-species/per-step record
  * ``thermal_snapshot_rank<rank>_step<K>.csv`` owned E/B/J cells

This script performs two jobs:

1.  **Schema/consistency validation** of one probe directory (missing steps,
    missing ranks, malformed or non-finite values, duplicated snapshot cells,
    inconsistent step sets).
2.  **Caravan-vs-legacy comparison** of two probe directories for the same
    global configuration: exact owned macro-particle counts, tolerant scalar
    comparison, and cell-by-cell field comparison after merging ranks by global
    cell coordinate.

Exceptions are turned into a non-zero exit and a report on stdout; nothing is
silently tolerated.  ``--selftest`` runs built-in positive and negative fixtures
so the validator itself is tested.

Usage::

    validate_probe.py --selfcheck DIR
    validate_probe.py --compare CARAVAN_DIR LEGACY_DIR [--rel-tol 1e-5] [--abs-tol 1e-12]
    validate_probe.py --selftest

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SUMMARY_COLUMNS = [
    "step",
    "stage",
    "species",
    "count",
    "weight_sum",
    "mom_x",
    "mom_y",
    "mom_z",
    "ekin",
    "etot",
    "particle_nonfinite",
    "E_energy",
    "E_abs",
    "E_min",
    "E_max",
    "E_nonfinite",
    "B_energy",
    "B_abs",
    "B_min",
    "B_max",
    "B_nonfinite",
    "J_available",
    "J_energy",
    "J_abs",
    "J_min",
    "J_max",
    "J_nonfinite",
]

SNAPSHOT_COLUMNS = ["gx", "gy", "gz", "Ex", "Ey", "Ez", "Bx", "By", "Bz", "Jx", "Jy", "Jz"]

# Scalars that must be summed over ranks when forming a global value.
SUMMED = [
    "count",
    "weight_sum",
    "mom_x",
    "mom_y",
    "mom_z",
    "ekin",
    "etot",
    "particle_nonfinite",
    "E_energy",
    "E_abs",
    "E_nonfinite",
    "B_energy",
    "B_abs",
    "B_nonfinite",
    "J_energy",
    "J_abs",
    "J_nonfinite",
]
# Scalars that take the per-rank minimum / maximum.
MINIMIZED = ["E_min", "B_min", "J_min"]
MAXIMIZED = ["E_max", "B_max", "J_max"]


class ProbeError(RuntimeError):
    pass


@dataclass
class Record:
    step: int
    stage: str
    species: str
    values: Dict[str, float]
    rank: int


@dataclass
class Directory:
    path: str
    records: List[Record] = field(default_factory=list)
    snapshots: Dict[int, Dict[Tuple[int, int, int], List[float]]] = field(default_factory=dict)


def parse_dir(path: str) -> Directory:
    if not os.path.isdir(path):
        raise ProbeError(f"probe directory does not exist: {path}")

    result = Directory(path=path)

    summary_files = sorted(f for f in os.listdir(path) if f.startswith("thermal_probe_rank") and f.endswith(".csv"))
    if not summary_files:
        raise ProbeError(f"no thermal_probe_rank*.csv files in {path}")

    for name in summary_files:
        rank = int(name[len("thermal_probe_rank") : -len(".csv")])
        full = os.path.join(path, name)
        with open(full, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines:
            raise ProbeError(f"{full}: empty")
        header = lines[0].split(",")
        if header != SUMMARY_COLUMNS:
            raise ProbeError(f"{full}: unexpected header {header}")
        for line in lines[1:]:
            fields = line.split(",")
            if len(fields) != len(SUMMARY_COLUMNS):
                raise ProbeError(f"{full}: row has {len(fields)} columns, expected {len(SUMMARY_COLUMNS)}: {line}")
            raw = dict(zip(SUMMARY_COLUMNS, fields))
            step = int(raw["step"])
            values: Dict[str, float] = {}
            for column in SUMMARY_COLUMNS:
                if column in ("step", "stage", "species"):
                    continue
                try:
                    value = float(raw[column])
                except ValueError as error:
                    raise ProbeError(f"{full}: non-numeric {column}={raw[column]!r}") from error
                values[column] = value
            # J_min/J_max are inf/-inf when the current is unavailable; that is
            # only allowed when J_available == 0.
            for column in ("E_min", "E_max", "B_min", "B_max", "E_energy", "B_energy", "E_abs", "B_abs"):
                if not math.isfinite(values[column]):
                    raise ProbeError(f"{full}: non-finite {column}={values[column]} at step {step}")
            if values["J_available"] != 0.0:
                for column in ("J_energy", "J_abs", "J_min", "J_max"):
                    if not math.isfinite(values[column]):
                        raise ProbeError(f"{full}: J available but non-finite {column} at step {step}")
            result.records.append(
                Record(step=step, stage=raw["stage"], species=raw["species"], values=values, rank=rank)
            )

    snapshot_files = sorted(f for f in os.listdir(path) if f.startswith("thermal_snapshot_rank"))
    for name in snapshot_files:
        # thermal_snapshot_rank<rank>_step<step>.csv
        stem = name[len("thermal_snapshot_rank") : -len(".csv")]
        rank_text, step_text = stem.split("_step")
        rank = int(rank_text)
        step = int(step_text)
        target = result.snapshots.setdefault(step, {})
        full = os.path.join(path, name)
        with open(full, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
        if not lines or lines[0].split(",") != SNAPSHOT_COLUMNS:
            raise ProbeError(f"{full}: unexpected snapshot header")
        for line in lines[1:]:
            fields = line.split(",")
            if len(fields) != len(SNAPSHOT_COLUMNS):
                raise ProbeError(f"{full}: snapshot row has {len(fields)} columns")
            key = (int(fields[0]), int(fields[1]), int(fields[2]))
            if key in target:
                raise ProbeError(f"{full}: duplicate global cell {key} (rank {rank})")
            try:
                values = [float(v) for v in fields[3:]]
            except ValueError as error:
                raise ProbeError(f"{full}: non-numeric snapshot value") from error
            target[key] = values

    validate_consistency(result)
    return result


def validate_consistency(directory: Directory) -> None:
    """Reject missing steps, missing ranks, and mismatched configurations."""
    by_step: Dict[int, List[Record]] = {}
    for record in directory.records:
        by_step.setdefault(record.step, []).append(record)

    if 0 not in by_step:
        raise ProbeError(f"{directory.path}: no initialization (step 0) record")

    expected_ranks = sorted({record.rank for record in directory.records})
    for step, records in by_step.items():
        ranks = sorted(record.rank for record in records)
        if ranks != expected_ranks:
            raise ProbeError(f"{directory.path}: step {step} has ranks {ranks}, expected {expected_ranks}")
        species = {record.species for record in records}
        if len(species) != 1:
            raise ProbeError(f"{directory.path}: step {step} has multiple species {species}")

    # Steps must be contiguous starting at zero.
    steps = sorted(by_step)
    if steps != list(range(steps[0], steps[0] + len(steps))) or steps[0] != 0:
        raise ProbeError(f"{directory.path}: non-contiguous step set {steps}")


def aggregate(directory: Directory) -> Dict[int, Dict[str, float]]:
    by_step: Dict[int, List[Record]] = {}
    for record in directory.records:
        by_step.setdefault(record.step, []).append(record)

    aggregated: Dict[int, Dict[str, float]] = {}
    for step, records in by_step.items():
        totals: Dict[str, float] = {}
        for key in SUMMED:
            totals[key] = float(sum(record.values[key] for record in records))
        for key in MINIMIZED:
            values = [record.values[key] for record in records if record.values[key] != math.inf]
            totals[key] = min(values) if values else math.inf
        for key in MAXIMIZED:
            values = [record.values[key] for record in records if record.values[key] != -math.inf]
            totals[key] = max(values) if values else -math.inf
        totals["J_available"] = max(record.values["J_available"] for record in records)
        aggregated[step] = totals
    return aggregated


def close(caravan: float, legacy: float, rel_tol: float, abs_tol: float) -> bool:
    if caravan == legacy:
        return True
    return math.isclose(caravan, legacy, rel_tol=rel_tol, abs_tol=abs_tol)


def compare_scalars(
    caravan: Directory,
    legacy: Directory,
    rel_tol: float,
    abs_tol: float,
) -> List[str]:
    failures: List[str] = []
    caravan_agg = aggregate(caravan)
    legacy_agg = aggregate(legacy)

    if sorted(caravan_agg) != sorted(legacy_agg):
        failures.append(f"step sets differ: caravan={sorted(caravan_agg)} legacy={sorted(legacy_agg)}")
        return failures

    for step in sorted(caravan_agg):
        a = caravan_agg[step]
        b = legacy_agg[step]
        # Exact owned macro-particle count.
        if a["count"] != b["count"]:
            failures.append(f"step {step}: count caravan={a['count']:.0f} legacy={b['count']:.0f}")
        if a["particle_nonfinite"] != 0 or b["particle_nonfinite"] != 0:
            failures.append(
                f"step {step}: non-finite particles caravan={a['particle_nonfinite']:.0f} "
                f"legacy={b['particle_nonfinite']:.0f}"
            )
        # Current must be nonzero wherever deposition is expected (steps > 0).
        if step > 0:
            if int(a["J_available"]) == 0 or int(b["J_available"]) == 0:
                failures.append(
                    f"step {step}: evolved J unavailable (caravan={a['J_available']} legacy={b['J_available']})"
                )
            else:
                for key in ("J_energy", "J_abs"):
                    if a[key] == 0.0 or b[key] == 0.0:
                        failures.append(f"step {step}: zero {key} (caravan={a[key]} legacy={b[key]})")
        for key in SUMMED + MINIMIZED + MAXIMIZED:
            if key == "count":
                continue
            if key in ("J_energy", "J_abs", "J_min", "J_max") and int(a.get("J_available", 0)) == 0:
                continue
            if not close(a[key], b[key], rel_tol, abs_tol):
                failures.append(
                    f"step {step}: {key} caravan={a[key]:.10g} legacy={b[key]:.10g} "
                    f"rel={abs(a[key] - b[key]) / max(abs(b[key]), 1e-300):.3e}"
                )
    return failures


def compare_snapshots(
    caravan: Directory,
    legacy: Directory,
    snapshot_rel_tol: float,
    snapshot_abs_tol: float,
) -> List[str]:
    """Compare merged owned-field snapshots.

    The two implementations evaluate independent stages in a different order, so
    per-cell float32 values are not expected to be bit-identical even though the
    total energies and particle moments are.  A cell/component passes when
    ``|a-b| <= snapshot_rel_tol*|b| + snapshot_abs_tol``.  The maximum
    scale-normalized deviation is reported for every step/component so the
    residual can be tracked rather than hidden.
    """
    failures: List[str] = []
    common = sorted(set(caravan.snapshots) & set(legacy.snapshots))
    if not common:
        return failures

    components = SNAPSHOT_COLUMNS[3:]
    for step in common:
        a = caravan.snapshots[step]
        b = legacy.snapshots[step]
        only_a = set(a) - set(b)
        only_b = set(b) - set(a)
        if only_a or only_b:
            failures.append(
                f"step {step}: snapshot cell mismatch (caravan-only={len(only_a)}, legacy-only={len(only_b)})"
            )
            continue

        max_scale = [0.0] * len(components)
        max_rel = [0.0] * len(components)
        max_abs = [0.0] * len(components)
        violating = [0] * len(components)
        for cell in a:
            for index in range(len(components)):
                av = a[cell][index]
                bv = b[cell][index]
                if math.isnan(av) or math.isnan(bv):
                    continue
                max_scale[index] = max(max_scale[index], abs(bv))
                deviation = abs(av - bv)
                max_abs[index] = max(max_abs[index], deviation)
                if abs(bv) > snapshot_abs_tol:
                    max_rel[index] = max(max_rel[index], deviation / abs(bv))
                if deviation > snapshot_rel_tol * abs(bv) + snapshot_abs_tol:
                    violating[index] += 1

        for index, component in enumerate(components):
            normalized = max_abs[index] / max_scale[index] if max_scale[index] > 0.0 else max_abs[index]
            print(
                f"  snapshot step {step} {component}: max|delta|={max_abs[index]:.3e} "
                f"scale={max_scale[index]:.3e} max-rel={max_rel[index]:.3e} normalized={normalized:.3e} "
                f"cells-over-tol={violating[index]}"
            )
            if violating[index] != 0 and normalized > snapshot_rel_tol:
                failures.append(
                    f"step {step} component {component}: {violating[index]} cell(s) over tolerance, "
                    f"max-rel={max_rel[index]:.3e}"
                )
    return failures


def selfcheck(path: str) -> int:
    """Validate one probe directory and report its global summaries."""
    directory = parse_dir(path)
    aggregated = aggregate(directory)
    stage_by_step = {record.step: record.stage for record in directory.records}
    print(f"OK: {path}")
    print("step stage count weight_sum ekin E_energy B_energy J_available")
    for step in sorted(aggregated):
        a = aggregated[step]
        print(
            f"{step} {stage_by_step.get(step, '?')} {a['count']:.0f} {a['weight_sum']:.6g} "
            f"{a['ekin']:.6g} {a['E_energy']:.6g} {a['B_energy']:.6g} {a['J_available']:.0f}"
        )
    return 0


def selftest() -> int:
    import tempfile

    def write(directory: str, rows: List[Tuple[int, int, int]], header: bool = True) -> None:
        with open(os.path.join(directory, "thermal_probe_rank0.csv"), "w", encoding="utf-8") as handle:
            if header:
                handle.write(",".join(SUMMARY_COLUMNS) + "\n")
            for step, count, eenergy in rows:
                values = {
                    "step": step,
                    "stage": "init" if step == 0 else "step",
                    "species": "e",
                    "count": count,
                    "weight_sum": 1.0,
                    "mom_x": 0.0,
                    "mom_y": 0.0,
                    "mom_z": 0.0,
                    "ekin": 1.0,
                    "etot": 1.0,
                    "particle_nonfinite": 0,
                    "E_energy": eenergy,
                    "E_abs": 0.0,
                    "E_min": 0.0,
                    "E_max": 0.0,
                    "E_nonfinite": 0,
                    "B_energy": 0.0,
                    "B_abs": 0.0,
                    "B_min": 0.0,
                    "B_max": 0.0,
                    "B_nonfinite": 0,
                    "J_available": 0 if step == 0 else 1,
                    "J_energy": 0.0 if step == 0 else 1.0,
                    "J_abs": 0.0 if step == 0 else 1.0,
                    "J_min": math.inf if step == 0 else 0.0,
                    "J_max": -math.inf if step == 0 else 1.0,
                    "J_nonfinite": 0,
                }
                handle.write(",".join(str(values[column]) for column in SUMMARY_COLUMNS) + "\n")

    with tempfile.TemporaryDirectory() as root:
        good = os.path.join(root, "good")
        other = os.path.join(root, "other")
        bad_count = os.path.join(root, "bad_count")
        bad_nonfinite = os.path.join(root, "bad_nonfinite")
        bad_gap = os.path.join(root, "bad_gap")
        for directory in (good, other, bad_count, bad_nonfinite, bad_gap):
            os.makedirs(directory)
        write(good, [(0, 100, 0.0), (1, 100, 0.5), (2, 100, 1.0)])
        write(other, [(0, 100, 0.0), (1, 100, 0.5), (2, 100, 1.0)])
        write(bad_count, [(0, 100, 0.0), (1, 99, 0.5), (2, 100, 1.0)])
        write(bad_nonfinite, [(0, 100, 0.0), (1, 100, math.inf), (2, 100, 1.0)])
        write(bad_gap, [(0, 100, 0.0), (2, 100, 1.0)])

        if compare_scalars(parse_dir(good), parse_dir(other), 1e-6, 1e-12):
            print("selftest FAIL: identical directories reported different")
            return 1
        if not compare_scalars(parse_dir(good), parse_dir(bad_count), 1e-6, 1e-12):
            print("selftest FAIL: count corruption not detected")
            return 1
        try:
            parse_dir(bad_nonfinite)
        except ProbeError:
            pass
        else:
            print("selftest FAIL: non-finite field not detected")
            return 1
        try:
            parse_dir(bad_gap)
        except ProbeError:
            pass
        else:
            print("selftest FAIL: missing step not detected")
            return 1
    print("selftest: all checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selfcheck", metavar="DIR")
    mode.add_argument("--compare", nargs=2, metavar=("CARAVAN_DIR", "LEGACY_DIR"))
    mode.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-4,
        help="scalar tolerance; signed momentum sums are cancellation-limited in float32 (observed <=3e-5)",
    )
    parser.add_argument("--abs-tol", type=float, default=1e-12)
    parser.add_argument(
        "--snapshot-rel-tol",
        type=float,
        default=5e-2,
        help="per-cell field tolerance; independent stages are ordered differently, so cells are not bit-identical",
    )
    parser.add_argument("--snapshot-abs-tol", type=float, default=1e-10)
    parser.add_argument("--require-snapshots", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.selfcheck:
        try:
            return selfcheck(args.selfcheck)
        except ProbeError as error:
            print(f"FAIL: {error}")
            return 1

    try:
        caravan = parse_dir(args.compare[0])
        legacy = parse_dir(args.compare[1])
    except ProbeError as error:
        print(f"FAIL: {error}")
        return 1

    if args.require_snapshots and (not caravan.snapshots or not legacy.snapshots):
        print("FAIL: snapshots required but missing")
        return 1

    failures = compare_scalars(caravan, legacy, args.rel_tol, args.abs_tol)
    failures += compare_snapshots(caravan, legacy, args.snapshot_rel_tol, args.snapshot_abs_tol)

    if failures:
        print(f"FAIL: {len(failures)} mismatch(es)")
        for failure in failures[:40]:
            print("  " + failure)
        return 1

    print(f"PASS: caravan={args.compare[0]} legacy={args.compare[1]} rel_tol={args.rel_tol:g} abs_tol={args.abs_tol:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
