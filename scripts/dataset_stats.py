#!/usr/bin/env python3
"""Dataset statistics: print boards sorted by a chosen property.

Frames are named {board}_{frame}.png / .txt. Labels exist only for frames
that contain at least one annotation, so a "gap" inside a board may mean
either (a) the source frame had no knots and was dropped, or (b) the
frame is genuinely missing. "first" / "last" frame are the lowest / highest
observed indices per board — lower / upper bounds on the board's true
extent, since leading or trailing knotless frames are absent.

Each label line is one YOLO-format annotation (cls cx cy w h, normalised).
Frames overlap 50% along X (stride 320 of 640 px), so one physical knot is
typically annotated in two adjacent frames; the `annots` column counts raw
label lines and does NOT de-duplicate across that overlap.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from stage_util import (
    add_config_arg,
    apply_config_defaults,
    load_config_section,
    stage_timer,
)

FRAME_RE = re.compile(r"^(?P<board>\d+)_(?P<frame>\d+)\.(?P<ext>png|txt)$")
STAGE = "dataset_stats"

SORT_KEYS = ("frames", "gaps", "span", "annots", "first", "last", "board")


@dataclass
class BoardStats:
    board: int
    first: int
    last: int
    frames: int  # number of present frame indices
    gaps: int  # missing indices in [first, last]
    annots: int  # total annotation lines across all frames

    @property
    def span(self) -> int:
        return self.last - self.first + 1


def collect(data_dir: Path) -> tuple[dict[int, BoardStats], int, int]:
    """Walk images/ and labels/ once; return per-board stats plus file totals."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to exist.")

    frames_by_board: dict[int, set[int]] = defaultdict(set)
    image_count = 0
    for entry in images_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if not m or m.group("ext") != "png":
            continue
        frames_by_board[int(m.group("board"))].add(int(m.group("frame")))
        image_count += 1

    annots_by_board: dict[int, int] = defaultdict(int)
    label_count = 0
    for entry in labels_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if not m or m.group("ext") != "txt":
            continue
        board = int(m.group("board"))
        frame = int(m.group("frame"))
        frames_by_board[board].add(frame)
        with entry.open() as fh:
            annots_by_board[board] += sum(1 for line in fh if line.strip())
        label_count += 1

    stats: dict[int, BoardStats] = {}
    for board, frames in frames_by_board.items():
        present = sorted(frames)
        first, last = present[0], present[-1]
        gaps = (last - first + 1) - len(present)
        stats[board] = BoardStats(
            board=board,
            first=first,
            last=last,
            frames=len(present),
            gaps=gaps,
            annots=annots_by_board.get(board, 0),
        )
    return stats, image_count, label_count


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "--sort",
        choices=SORT_KEYS,
        default="frames",
        help="Property to sort boards by.",
    )
    parser.add_argument(
        "--order",
        choices=("asc", "desc"),
        default="asc",
        help="Sort order. 'asc' = lowest first.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Print at most this many boards. 0 = print all.",
    )
    apply_config_defaults(parser, load_config_section(pre_args.config, STAGE))
    args = parser.parse_args()

    with stage_timer(STAGE):
        _run(args)


def _run(args: argparse.Namespace) -> None:
    stats, image_count, label_count = collect(args.data_dir)

    print(
        f"boards={len(stats)}  images={image_count}  labels={label_count}  "
        f"annots={sum(s.annots for s in stats.values())}"
    )
    limit_str = "all" if args.limit <= 0 else str(args.limit)
    print(f"sorted by {args.sort} {args.order} (limit={limit_str})")
    print()

    rows = sorted(
        stats.values(),
        key=lambda s: getattr(s, args.sort),
        reverse=(args.order == "desc"),
    )
    if args.limit > 0:
        rows = rows[: args.limit]

    header = f"{'board':>6}  {'frames':>6}  {'gaps':>4}  {'span':>4}  {'first':>5}  {'last':>4}  {'annots':>6}"
    print(header)
    print("-" * len(header))
    for s in rows:
        print(
            f"{s.board:>6}  {s.frames:>6}  {s.gaps:>4}  {s.span:>4}  {s.first:>5}  {s.last:>4}  {s.annots:>6}"
        )


if __name__ == "__main__":
    main()
