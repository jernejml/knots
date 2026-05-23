#!/usr/bin/env python3
"""Stratified train/val/test partition of boards — single-stage dataset prep.

Walks data/{images,labels}/ once and produces the only artefact downstream
stages consume from this on-ramp:

    out/analysis/partitions.json

The shape is split-keyed lists of board IDs:

    {"train": [1, 4, 5, ...], "val": [2, 6, ...], "test": [0, 3, ...]}

Replaces the older analyze_dataset.py → board_features.py → make_splits.py
chain. The per-frame / per-annotation JSONs the chain used to write were
only consumed by the next stage in the chain; nothing else read them.
Stratification still happens on (frames, cluster_frac) but the only thing
this script computes per board is the two features it needs:

    frames        number of labelled frames for the board
    cluster_frac  fraction of frames containing >=1 cluster of 2+
                  near-touching annotations (axial-gap union-find,
                  threshold = max(near_T_min_px, near_T_rel * min_dim))

The split unit is the BOARD, not the frame: the 50% frame overlap means a
knot annotated in frame N is annotated again in frame N+1, so any frame-
level split would leak knots between train/val/test.

Reproducibility: same --seed + same inputs + same flags ⇒ identical output.

For richer per-frame / per-annotation inspection (geometry, darkness,
edge-touch counts, size buckets, contrast), see scripts/tools/inspect_dataset.py
— a manual debug tool, not part of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from stage_util import (
    add_config_arg,
    apply_config_defaults,
    iter_with_progress,
    load_config_section,
    save_run_meta,
    stage_timer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = "prepare"
FRAME_RE = re.compile(r"^(?P<board>\d+)_(?P<frame>\d+)\.png$")
SPLIT_NAMES = ("train", "val", "test")


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# -- Label parsing + cluster detection --------------------------------------
# Just enough of analyze_dataset.py to compute cluster_frac. We don't need
# the per-annotation features (size buckets, edge distances, contrast); the
# old pipeline computed them but they were never consumed by make_splits.

def parse_yolo_label(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Return pixel-space (x1, y1, x2, y2) tuples; empty list if file missing."""
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx_n, cy_n, w_n, h_n = (
            float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        )
        cx, cy, bw, bh = cx_n * w, cy_n * h, w_n * w, h_n * h
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def _axial_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    gx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (gx * gx + gy * gy) ** 0.5


def has_cluster(
    boxes: list[tuple[float, float, float, float]],
    near_T_min_px: float,
    near_T_rel: float,
) -> bool:
    """True iff at least one connected group of >=2 near-touching boxes exists.

    Two boxes are 'near' when their axial gap ≤ max(near_T_min_px,
    near_T_rel * min(min_dim_a, min_dim_b)). Same formula as the original
    analyze_dataset.py.cluster_metrics — we just collapse the rich output
    (touching_pairs, near_pairs, cluster sizes, …) to a single bool.
    """
    n = len(boxes)
    if n < 2:
        return False
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        ai = boxes[i]
        mdi = min(ai[2] - ai[0], ai[3] - ai[1])
        for j in range(i + 1, n):
            aj = boxes[j]
            mdj = min(aj[2] - aj[0], aj[3] - aj[1])
            threshold = max(near_T_min_px, near_T_rel * min(mdi, mdj))
            if _axial_gap(ai, aj) <= threshold:
                union(i, j)

    counts: dict[int, int] = defaultdict(int)
    for i in range(n):
        counts[find(i)] += 1
    return any(s >= 2 for s in counts.values())


# -- Stratification helpers (ported verbatim from make_splits.py) ----------

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
    return tuple(p / total for p in parts)


def tier_label(value: float, cuts: list[float]) -> str:
    """Human-readable bracket label like '(10,23]' for the tier of `value`."""
    for i, c in enumerate(cuts):
        if value <= c:
            lo = "-inf" if i == 0 else f"{cuts[i - 1]:g}"
            return f"{'[' if i == 0 else '('}{lo},{c:g}]"
    lo = f"{cuts[-1]:g}"
    return f"({lo},inf]"


def largest_remainder(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate n items into 3 groups by ratios with largest-remainder rounding."""
    raw = [n * r for r in ratios]
    floors = [int(x) for x in raw]
    remainder = n - sum(floors)
    fractions = [(raw[i] - floors[i], i) for i in range(3)]
    fractions.sort(key=lambda fi: (-fi[0], fi[1]))
    out = list(floors)
    for k in range(remainder):
        out[fractions[k][1]] += 1
    return tuple(out)


# -- Per-board feature pass -------------------------------------------------

def compute_board_features(
    data_dir: Path,
    near_T_min_px: float,
    near_T_rel: float,
) -> list[dict]:
    """Single pass over data/{images,labels}/: returns one row per board.

    Image pixels are never decoded — PIL's lazy header read gives us frame
    dims without touching pixel data. Heights vary by board but are
    effectively constant within a board (mixed-height frames inside one
    board were flagged as an anomaly by the old pipeline; we ignore that
    edge case since cluster_frac shifts only marginally if it occurs).
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"expected {images_dir} and {labels_dir} to exist")

    by_board: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for entry in images_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if m:
            by_board[int(m["board"])].append((int(m["frame"]), entry))
    for frames in by_board.values():
        frames.sort()

    boards_sorted = sorted(by_board.items())
    rows: list[dict] = []
    for board, frames in iter_with_progress(boards_sorted, "boards", every=10):
        # One header read per board for (w, h). Constant within a board.
        with Image.open(frames[0][1]) as im:
            w, h = im.size

        cluster_frames = 0
        for frame_idx, _ in frames:
            boxes = parse_yolo_label(labels_dir / f"{board}_{frame_idx}.txt", w, h)
            if has_cluster(boxes, near_T_min_px, near_T_rel):
                cluster_frames += 1

        n_frames = len(frames)
        rows.append({
            "board": board,
            "frames": n_frames,
            "cluster_frac": (cluster_frames / n_frames) if n_frames else 0.0,
        })
    return rows


# -- Splitter --------------------------------------------------------------

def stratify_and_split(
    boards: list[dict],
    stratify_specs: list[tuple[str, list[float]]],
    ratios: tuple[float, float, float],
    seed: int,
    min_frames_train: int,
    forced: dict[int, str],
) -> tuple[dict[int, str], list[tuple[str, int, tuple[int, int, int]]]]:
    """Returns (assignment, per_stratum_alloc).

    assignment is {board_id: split_name}.
    per_stratum_alloc carries the stdout summary (stratum key, total, (nt, nv, nte)).
    """
    if boards:
        for fname, _ in stratify_specs:
            if fname not in boards[0]:
                raise SystemExit(
                    f"--stratify feature {fname!r} not in computed columns "
                    f"({sorted(boards[0])})"
                )

    assignment: dict[int, str] = {}
    auto_train: list[int] = []
    pool: list[dict] = []
    for b in boards:
        bid = b["board"]
        if bid in forced:
            assignment[bid] = forced[bid]
            continue
        if b["frames"] <= min_frames_train:
            assignment[bid] = "train"
            auto_train.append(bid)
            continue
        pool.append(b)

    strata: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for b in pool:
        key = tuple(f"{name}{tier_label(b[name], cuts)}" for name, cuts in stratify_specs)
        strata[key].append(b)

    alloc: list[tuple[str, int, tuple[int, int, int]]] = []
    for stratum_key in sorted(strata):
        boards_in = sorted(strata[stratum_key], key=lambda b: b["board"])
        # Per-stratum seeded RNG: same seed → same shuffle regardless of
        # which other strata exist. Lets you add/remove a stratum without
        # disturbing the others' assignments.
        rng = random.Random(f"{seed}|{'|'.join(stratum_key)}")
        rng.shuffle(boards_in)
        n_train, n_val, n_test = largest_remainder(len(boards_in), ratios)
        for b in boards_in[:n_train]:
            assignment[b["board"]] = "train"
        for b in boards_in[n_train : n_train + n_val]:
            assignment[b["board"]] = "val"
        for b in boards_in[n_train + n_val :]:
            assignment[b["board"]] = "test"
        alloc.append(("/".join(stratum_key), len(boards_in), (n_train, n_val, n_test)))

    return assignment, alloc


# -- I/O -------------------------------------------------------------------

def write_partitions(path: Path, assignment: dict[int, str]) -> None:
    """Write {"train": [...], "val": [...], "test": [...]} with board IDs sorted."""
    grouped: dict[str, list[int]] = {s: [] for s in SPLIT_NAMES}
    for board, split in assignment.items():
        grouped[split].append(board)
    for split in SPLIT_NAMES:
        grouped[split].sort()
    with path.open("w") as fh:
        json.dump(grouped, fh, indent=2)


# -- Entrypoint ------------------------------------------------------------

def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(p)
    p.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data",
        help="Read-only input dir containing images/ and labels/.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "out" / "analysis",
        help="Where partitions.json is written.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed; the only knob that changes the random portion of the split.",
    )
    p.add_argument(
        "--ratios", type=str, default="80,10,10",
        help="Comma-separated train,val,test percentages.",
    )
    p.add_argument(
        "--stratify", action="append", type=parse_stratify_spec, default=None,
        help="Stratification axis as FEATURE:cut1[,cut2,...]. Repeatable. "
        "Default: 'frames:10,23' and 'cluster_frac:0'.",
    )
    p.add_argument(
        "--min-frames-train", type=int, default=3,
        help="Boards with frames <= this go straight to train.",
    )
    p.add_argument(
        "--force-train", type=parse_id_list, default=[],
        help="Comma-separated board IDs pinned to train.",
    )
    p.add_argument(
        "--force-val", type=parse_id_list, default=[],
        help="Comma-separated board IDs pinned to val.",
    )
    p.add_argument(
        "--force-test", type=parse_id_list, default=[],
        help="Comma-separated board IDs pinned to test.",
    )
    p.add_argument(
        "--near-T-min-px", type=float, default=5.0,
        help="Floor (px) of the hybrid near-touching threshold.",
    )
    p.add_argument(
        "--near-T-rel", type=float, default=0.25,
        help="Multiplier of min(min_dim_a, min_dim_b) for the hybrid threshold.",
    )
    apply_config_defaults(p, load_config_section(pre_args.config, STAGE))
    args = p.parse_args()

    with stage_timer(STAGE) as timing:
        _run(args)
    save_run_meta(args.output_dir, STAGE, args, elapsed_sec=timing["elapsed_sec"])


def _run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stratify_specs = args.stratify or [("frames", [10.0, 23.0]), ("cluster_frac", [0.0])]
    ratios = parse_ratios(args.ratios)

    # Forced-overrides conflict check.
    forced: dict[int, str] = {}
    for split_name, ids in (
        ("train", args.force_train),
        ("val", args.force_val),
        ("test", args.force_test),
    ):
        for bid in ids:
            if bid in forced and forced[bid] != split_name:
                raise SystemExit(f"board {bid} forced to multiple splits")
            forced[bid] = split_name

    print(f"prepare: scanning {rel_to_root(args.data_dir)}/")
    boards = compute_board_features(args.data_dir, args.near_T_min_px, args.near_T_rel)
    if not boards:
        raise SystemExit("no boards found under data/images/")

    assignment, alloc = stratify_and_split(
        boards, stratify_specs, ratios, args.seed, args.min_frames_train, forced
    )

    out_path = args.output_dir / "partitions.json"
    write_partitions(out_path, assignment)

    # Stdout summary.
    print(f"  boards={len(boards)}  seed={args.seed}  ratios={args.ratios}  "
          f"min_frames_train={args.min_frames_train}")
    print("  stratify:")
    for name, cuts in stratify_specs:
        print(f"    {name}: cuts={cuts}")
    forced_counts = {s: sum(1 for v in forced.values() if v == s) for s in SPLIT_NAMES}
    auto_train_n = sum(1 for b in boards
                       if b["board"] not in forced and b["frames"] <= args.min_frames_train)
    print(f"  forced: train={forced_counts['train']} val={forced_counts['val']} "
          f"test={forced_counts['test']}  auto-train-by-length={auto_train_n}")

    print()
    print("Per-stratum allocation")
    header = f"  {'stratum':<48}  {'total':>5}  {'train':>5}  {'val':>5}  {'test':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, total, (nt, nv, nte) in alloc:
        print(f"  {key:<48}  {total:>5}  {nt:>5}  {nv:>5}  {nte:>5}")
        if total < 10:
            print(f"    warning: stratum has {total} boards; smaller splits may be unreliable.")

    print()
    print("Totals")
    frames_by_board = {b["board"]: b["frames"] for b in boards}
    for split in SPLIT_NAMES:
        ids = [bid for bid, s in assignment.items() if s == split]
        n_f = sum(frames_by_board[bid] for bid in ids)
        print(f"  {split:>5}: boards={len(ids):>4}  frames={n_f:>5}")
    print(f"  total: boards={len(boards):>4}  frames={sum(frames_by_board.values()):>5}")

    print()
    print(f"written: {rel_to_root(out_path)}")


if __name__ == "__main__":
    main()
