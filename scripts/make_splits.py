#!/usr/bin/env python3
"""Stratified 80/10/10 board-level split for the wood-knot dataset.

Reads out/analysis/board_features.json (so board_features.py must have been
run first) and writes:
    out/analysis/splits.csv
    out/analysis/splits.json
plus a stdout summary covering per-stratum allocation, frame/annotation
counts per split, and a balance audit on every numeric feature that was
not used as a stratification axis.

The split unit is the BOARD. This is forced by the dataset's 50% frame
overlap: a knot annotated in frame N is annotated again in frame N+1,
so any frame-level split leaks knots between train/val/test. Splitting
by board removes that leak.

Stratification dimensions are configurable via repeatable --stratify flags
of the form FEATURE:cut1[,cut2,...]. Each cut splits boards into N+1 tiers
using the convention "value <= C goes to the lower tier". Defaults give
3 length-tiers x 2 cluster-tiers = 6 strata.

Reproducibility: same --seed + same inputs + same flags = identical output.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from stage_util import (
    add_config_arg,
    apply_config_defaults,
    load_config_section,
    save_run_meta,
    stage_timer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = "make_splits"


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_stratify_spec(spec: str) -> tuple[str, list[float]]:
    """Parse 'frames:10,23' into ('frames', [10.0, 23.0])."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--stratify expects FEATURE:cut1[,cut2,...], got {spec!r}"
        )
    name, raw = spec.split(":", 1)
    name = name.strip()
    cuts = sorted(float(c) for c in raw.split(",") if c.strip())
    if not name or not cuts:
        raise argparse.ArgumentTypeError(f"empty feature or cuts in {spec!r}")
    for a, b in zip(cuts, cuts[1:]):
        if a == b:
            raise argparse.ArgumentTypeError(f"duplicate cut in {spec!r}")
    return name, cuts


def parse_id_list(spec: str) -> list[int]:
    if not spec:
        return []
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_ratios(spec: str) -> tuple[float, float, float]:
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--ratios needs exactly 3 numbers")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("--ratios must sum to a positive number")
    return tuple(p / total for p in parts)  # normalised


def tier_for(value: float, cuts: list[float]) -> int:
    """Return tier index for `value` given sorted `cuts`.

    Convention: value <= cuts[i] but > cuts[i-1] -> tier i.
    Equivalent: bisect_right-style.
    """
    for i, c in enumerate(cuts):
        if value <= c:
            return i
    return len(cuts)


def tier_label(value: float, cuts: list[float]) -> str:
    """Human-readable bracket label like '(10,23]' for the tier of `value`."""
    idx = tier_for(value, cuts)
    lo = "-inf" if idx == 0 else f"{cuts[idx - 1]:g}"
    hi = "inf" if idx == len(cuts) else f"{cuts[idx]:g}"
    left = "(" if idx > 0 else "["
    return f"{left}{lo},{hi}]"


def largest_remainder(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate n items into 3 groups by ratios with largest-remainder rounding.

    The order is (train, val, test); ties prefer earlier groups.
    """
    raw = [n * r for r in ratios]
    floors = [int(x) for x in raw]
    remainder = n - sum(floors)
    fractions = [(raw[i] - floors[i], i) for i in range(3)]
    # Stable tiebreak: larger fractional part first; if equal, earlier index first.
    fractions.sort(key=lambda fi: (-fi[0], fi[1]))
    out = list(floors)
    for k in range(remainder):
        out[fractions[k][1]] += 1
    return tuple(out)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(parser)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=REPO_ROOT / "out" / "analysis",
        help="Directory containing board_features.json and where splits.{csv,json} are written.",
    )
    parser.add_argument(
        "--frames-json",
        type=Path,
        default=None,
        help="Optional override for the per-frame file (used to count frames/annots per split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed; the only knob that changes the random portion of the split.",
    )
    parser.add_argument(
        "--ratios", type=str, default="80,10,10", help="Comma-separated train,val,test percentages."
    )
    parser.add_argument(
        "--stratify",
        action="append",
        type=parse_stratify_spec,
        default=None,
        help="Stratification axis as FEATURE:cut1[,cut2,...]. Repeatable. "
        "Default: 'frames:10,23' and 'cluster_frac:0'.",
    )
    parser.add_argument(
        "--min-frames-train",
        type=int,
        default=3,
        help="Boards with frames <= this go straight to train.",
    )
    parser.add_argument(
        "--force-train",
        type=parse_id_list,
        default=[],
        help="Comma-separated board IDs pinned to train.",
    )
    parser.add_argument(
        "--force-val",
        type=parse_id_list,
        default=[],
        help="Comma-separated board IDs pinned to val.",
    )
    parser.add_argument(
        "--force-test",
        type=parse_id_list,
        default=[],
        help="Comma-separated board IDs pinned to test.",
    )
    parser.add_argument(
        "--audit-threshold",
        type=float,
        default=0.20,
        help="Flag balance-audit rows with |split_mean/pop_mean - 1| > this.",
    )
    parser.add_argument(
        "--audit-features",
        type=str,
        default=None,
        help="Comma-separated list of features to audit. "
        "Defaults to every numeric column not used in --stratify.",
    )
    apply_config_defaults(parser, load_config_section(pre_args.config, STAGE))
    args = parser.parse_args()

    with stage_timer(STAGE) as timing:
        _run(args)
    save_run_meta(args.analysis_dir, STAGE, args, elapsed_sec=timing["elapsed_sec"])


def _run(args: argparse.Namespace) -> None:
    stratify_specs = args.stratify or [("frames", [10.0, 23.0]), ("cluster_frac", [0.0])]
    ratios = parse_ratios(args.ratios)

    bf_path = args.analysis_dir / "board_features.json"
    if not bf_path.is_file():
        raise SystemExit(f"Expected {bf_path}. Run scripts/board_features.py first.")
    boards = json.load(bf_path.open())
    if not boards:
        raise SystemExit("board_features.json is empty.")

    # Validate stratify features exist.
    first = boards[0]
    for fname, _ in stratify_specs:
        if fname not in first:
            raise SystemExit(
                f"--stratify feature {fname!r} not found in board_features.json columns: "
                f"{sorted(first.keys())}"
            )

    # ----- Conflict checks on forced overrides -----
    forced: dict[int, str] = {}
    for split_name, ids in (
        ("train", args.force_train),
        ("val", args.force_val),
        ("test", args.force_test),
    ):
        for bid in ids:
            if bid in forced and forced[bid] != split_name:
                raise SystemExit(f"board {bid} forced to multiple splits.")
            forced[bid] = split_name

    # ----- Partition: forced + auto-train-by-length vs. stratification pool -----
    assignment: dict[int, str] = {}
    auto_train: list[int] = []
    pool: list[dict] = []
    for b in boards:
        bid = b["board"]
        if bid in forced:
            assignment[bid] = forced[bid]
            continue
        if b["frames"] <= args.min_frames_train:
            assignment[bid] = "train"
            auto_train.append(bid)
            continue
        pool.append(b)

    # ----- Bucket pool into strata -----
    strata: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for b in pool:
        key = tuple(f"{name}{tier_label(b[name], cuts)}" for name, cuts in stratify_specs)
        strata[key].append(b)

    # ----- Per-stratum 80/10/10 with seeded shuffle -----
    SPLIT_NAMES = ("train", "val", "test")
    stratum_alloc: list[tuple[str, int, tuple[int, int, int]]] = []
    for stratum_key in sorted(strata):
        boards_in = sorted(strata[stratum_key], key=lambda b: b["board"])
        rng = random.Random(f"{args.seed}|{'|'.join(stratum_key)}")
        rng.shuffle(boards_in)
        n_train, n_val, n_test = largest_remainder(len(boards_in), ratios)
        for b in boards_in[:n_train]:
            assignment[b["board"]] = "train"
        for b in boards_in[n_train : n_train + n_val]:
            assignment[b["board"]] = "val"
        for b in boards_in[n_train + n_val :]:
            assignment[b["board"]] = "test"
        stratum_alloc.append(("/".join(stratum_key), len(boards_in), (n_train, n_val, n_test)))

    # ----- Write splits.csv / splits.json -----
    # Re-derive per-board tier labels for the output (so the file is self-describing).
    out_rows = []
    for b in boards:
        row = {"board": b["board"], "split": assignment[b["board"]]}
        for name, cuts in stratify_specs:
            row[f"{name}_tier"] = tier_label(b[name], cuts)
        out_rows.append(row)
    out_rows.sort(key=lambda r: r["board"])

    csv_path = args.analysis_dir / "splits.csv"
    json_path = args.analysis_dir / "splits.json"
    write_csv(csv_path, out_rows)
    write_json(json_path, out_rows)

    # ----- Stdout summary -----
    print(f"make_splits: {len(boards)} boards")
    print(f"  seed={args.seed}  ratios={args.ratios}  " f"min_frames_train={args.min_frames_train}")
    print("  stratify:")
    for name, cuts in stratify_specs:
        print(f"    {name}: cuts={cuts}")
    forced_counts = {s: sum(1 for v in forced.values() if v == s) for s in SPLIT_NAMES}
    print(
        f"  forced: train={forced_counts['train']} val={forced_counts['val']} test={forced_counts['test']}"
        f"  auto-train-by-length: {len(auto_train)}"
    )

    print()
    print("Per-stratum allocation")
    header = f"  {'stratum':<48}  {'total':>5}  {'train':>5}  {'val':>5}  {'test':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, total, (nt, nv, nte) in stratum_alloc:
        print(f"  {key:<48}  {total:>5}  {nt:>5}  {nv:>5}  {nte:>5}")
        if total < 10:
            print(f"    warning: stratum has {total} boards; the smaller splits may be unreliable.")

    # Per-split totals (board / frames / annots)
    print()
    print("Totals (board / frames / annots)")
    frames_by_board = {b["board"]: b["frames"] for b in boards}
    annots_by_board = {b["board"]: b["annots"] for b in boards}
    for split in SPLIT_NAMES:
        members = [b for b in boards if assignment[b["board"]] == split]
        n_b = len(members)
        n_f = sum(frames_by_board[b["board"]] for b in members)
        n_a = sum(annots_by_board[b["board"]] for b in members)
        print(f"  {split:>5}: boards={n_b:>4}  frames={n_f:>5}  annots={n_a:>5}")
    print(
        f"  total: boards={len(boards):>4}  frames={sum(frames_by_board.values()):>5}  annots={sum(annots_by_board.values()):>5}"
    )

    # ----- Balance audit -----
    skip = {"board", "height", "height_mixed"} | {name for name, _ in stratify_specs}
    if args.audit_features:
        audit_features = [x.strip() for x in args.audit_features.split(",") if x.strip()]
    else:
        audit_features = [
            k
            for k, v in first.items()
            if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
    print()
    print(
        f"Balance audit (mean per split vs population; *** flags |delta| > {args.audit_threshold * 100:.0f}%)"
    )
    aud_header = f"  {'feature':<22} {'pop':>9}  {'train':>15}  {'val':>15}  {'test':>15}"
    print(aud_header)
    print("  " + "-" * (len(aud_header) - 2))
    pop_vals = {f: [b[f] for b in boards] for f in audit_features}
    for f in audit_features:
        pop = sum(pop_vals[f]) / len(pop_vals[f]) if pop_vals[f] else 0.0
        cells = [f"  {f:<22} {pop:>9.4g}"]
        for split in SPLIT_NAMES:
            vals = [b[f] for b in boards if assignment[b["board"]] == split]
            m = sum(vals) / len(vals) if vals else 0.0
            dev = (m / pop - 1.0) if pop != 0 else 0.0
            flag = " ***" if abs(dev) > args.audit_threshold else "    "
            cells.append(f"  {m:>7.4g} ({dev:+5.0%}){flag}")
        print("".join(cells))

    print()
    print(f"written: {rel_to_root(csv_path)}, {rel_to_root(json_path)}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([r.get(c) for c in cols])


def write_json(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
