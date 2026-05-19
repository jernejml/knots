#!/usr/bin/env python3
"""Per-board feature summary aggregated from analyze_dataset.py outputs.

Reads analysis/frames.json + analysis/annotations.json (so analyze_dataset.py
must have been run first) and prints one row per board with the features most
relevant for stratification:

    frames       number of labelled frames in this board
    gaps         missing frame indices inside [first, last]
    annots       total annotation lines across the board's frames
    annots/f     mean annotations per labelled frame (knot density)
    %large       fraction of this board's annotations in the 'large' bucket
    avg_dark     mean dark_wood_frac across the board's frames (darkness)
    max_dark     peak dark_wood_frac on any single frame
    clust_fr     fraction of the board's frames with cluster_count >= 1
    edge_fr      fraction of frames with at least one edge-touching annot
    height       frame height — should be constant within a board, "MIX" if not

This is a read-only consumer of analyze_dataset.py's output. No images are
re-opened and no labels re-parsed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SORT_KEYS = (
    "board", "frames", "gaps", "annots", "annots_per_frame", "frac_large",
    "avg_dark", "max_dark", "cluster_frac", "edge_frac", "height",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path, default=REPO_ROOT / "analysis",
                        help="Directory containing frames.json and annotations.json.")
    parser.add_argument("--sort", choices=SORT_KEYS, default="frames",
                        help="Property to sort boards by.")
    parser.add_argument("--order", choices=("asc", "desc"), default="asc",
                        help="Sort order; 'asc' = lowest first.")
    parser.add_argument("--limit", type=int, default=20,
                        help="Print at most this many boards. 0 = print all.")
    args = parser.parse_args()

    frames_path = args.analysis_dir / "frames.json"
    annots_path = args.analysis_dir / "annotations.json"
    if not frames_path.is_file() or not annots_path.is_file():
        raise SystemExit(
            f"Expected {frames_path} and {annots_path}. "
            "Run scripts/analyze_dataset.py first."
        )

    frames = json.load(frames_path.open())
    annots = json.load(annots_path.open())

    by_board: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by_board[f["board"]].append(f)

    bucket_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in annots:
        bucket_counts[a["board"]][a["size_bucket"]] += 1

    rows: list[dict] = []
    for board, fs in by_board.items():
        n_frames = len(fs)
        present_idx = sorted(f["frame"] for f in fs)
        gaps = (present_idx[-1] - present_idx[0] + 1) - n_frames

        heights = {f["height"] for f in fs}
        height_mixed = len(heights) > 1
        height_repr = next(iter(heights)) if not height_mixed else -1

        total_annots = sum(f["annot_count"] for f in fs)
        annots_per_frame = (total_annots / n_frames) if n_frames > 0 else 0.0

        n_large = bucket_counts[board].get("large", 0)
        frac_large = (n_large / total_annots) if total_annots > 0 else 0.0

        dark_fracs = [f["dark_wood_frac"] for f in fs]
        avg_dark = sum(dark_fracs) / n_frames if n_frames > 0 else 0.0
        max_dark = max(dark_fracs) if dark_fracs else 0.0

        cluster_frames = sum(1 for f in fs if f["cluster_count"] > 0)
        edge_frames = sum(1 for f in fs if f["edge_touching_annot_count"] > 0)
        cluster_frac = (cluster_frames / n_frames) if n_frames > 0 else 0.0
        edge_frac = (edge_frames / n_frames) if n_frames > 0 else 0.0

        rows.append({
            "board": board,
            "frames": n_frames,
            "gaps": gaps,
            "annots": total_annots,
            "annots_per_frame": annots_per_frame,
            "frac_large": frac_large,
            "avg_dark": avg_dark,
            "max_dark": max_dark,
            "cluster_frac": cluster_frac,
            "edge_frac": edge_frac,
            "height": height_repr,
            "height_mixed": height_mixed,
        })

    rows.sort(key=lambda r: r[args.sort], reverse=(args.order == "desc"))
    if args.limit > 0:
        rows = rows[: args.limit]

    limit_str = "all" if args.limit <= 0 else str(args.limit)
    print(f"boards={len(by_board)}  sorted by {args.sort} {args.order} (limit={limit_str})")
    print()

    header = (
        f"{'board':>6}  {'frames':>6}  {'gaps':>4}  "
        f"{'annots':>6}  {'a/f':>5}  {'%large':>6}  "
        f"{'avg_dark':>8}  {'max_dark':>8}  "
        f"{'clust_fr':>8}  {'edge_fr':>7}  {'height':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        h = "MIX" if r["height_mixed"] else str(r["height"])
        print(
            f"{r['board']:>6}  {r['frames']:>6}  {r['gaps']:>4}  "
            f"{r['annots']:>6}  {r['annots_per_frame']:>5.2f}  "
            f"{r['frac_large']*100:>5.1f}%  "
            f"{r['avg_dark']:>8.3f}  {r['max_dark']:>8.3f}  "
            f"{r['cluster_frac']*100:>7.1f}%  {r['edge_frac']*100:>6.1f}%  {h:>6}"
        )


if __name__ == "__main__":
    main()
