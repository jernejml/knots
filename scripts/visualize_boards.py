#!/usr/bin/env python3
"""Per-board stitched visualisations with predicted and (optionally) GT polygons overlaid.

For each <board>.json under --pred-dir, this script:

  * Reads the per-board polygon list (output of `knots run`).
  * Loads the source frame PNGs for that board from --images-dir.
  * Stitches them into one wide image of size board_width x board_height
    (board_* are taken from the pred JSON, computed by the C++ stitcher).
  * Overlays predicted polygons in --pred-color and, when a matching JSON
    exists under --gt-dir, GT polygons in --gt-color.
  * Saves the result as JPEG to --output-dir/<board>.jpg.

GT overlay is opportunistic: if --gt-dir is missing or empty, output is
pred-only with a stderr note. That keeps `./run.sh viz` working in the
precommitted-model demo flow where there's no ground-truth pass.

Frame overlap handling: frames are 640 px wide on a 320 px stride, so
adjacent frames overlap 50%. Pasting each frame at (frame_idx * 320, 0)
lets later frames overwrite the right half of earlier ones — same content
modulo registration noise. No explicit dedup needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = "visualize_boards"

from PIL import Image, ImageDraw

# Boards can easily exceed PIL's default 89 MP guard (e.g. 30 k * 4 k = 120 MP),
# which is the whole point of stitching. We are the trusted producer of the
# input here, so disabling the guard is fine.
Image.MAX_IMAGE_PIXELS = None

from stage_util import (
    add_config_arg,
    apply_config_defaults,
    iter_with_progress,
    load_config_section,
    save_run_meta,
    stage_timer,
)

FRAME_RE = re.compile(r"^(?P<board>\d+)_(?P<frame>\d+)\.png$")


def parse_color(s: str) -> tuple[int, int, int]:
    """Parse 'R,G,B' (each 0-255) into a PIL-compatible RGB tuple."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--*-color expects 'R,G,B', got {s!r}")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--*-color components must be integers: {e}") from e
    if any(c < 0 or c > 255 for c in rgb):
        raise argparse.ArgumentTypeError(f"--*-color components must be 0-255, got {rgb}")
    return rgb  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_arg(p)
    p.add_argument(
        "--pred-dir",
        type=Path,
        default=REPO_ROOT / "out" / "boards" / "pred",
        help="Per-board prediction JSONs (output of `knots run`).",
    )
    p.add_argument(
        "--gt-dir",
        type=Path,
        default=REPO_ROOT / "out" / "boards" / "gt",
        help="Per-board GT JSONs (rebuilt by `knots eval --labels-dir`). "
        "If missing/empty, output is pred-only.",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=REPO_ROOT / "data" / "images",
        help="Source frame PNGs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "out" / "boards" / "viz",
        help="Where the per-board JPEGs go.",
    )
    p.add_argument(
        "--pred-color",
        type=parse_color,
        default=(0, 255, 0),
        help="RGB triple for predicted polygon outlines (default '0,255,0', green).",
    )
    p.add_argument(
        "--gt-color",
        type=parse_color,
        default=(255, 0, 0),
        help="RGB triple for GT polygon outlines (default '255,0,0', red).",
    )
    p.add_argument("--line-width", type=int, default=3, help="Polygon outline thickness in pixels.")
    p.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality 1-100 (default 92).")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (default: skip).",
    )

    apply_config_defaults(p, load_config_section(pre_args.config, STAGE))
    return p.parse_args()


def collect_board_frames(images_dir: Path) -> dict[int, list[tuple[int, Path]]]:
    """Return {board: [(frame_idx, png_path), ...]} sorted by frame_idx."""
    by_board: dict[int, list[tuple[int, Path]]] = {}
    for png in images_dir.iterdir():
        m = FRAME_RE.match(png.name)
        if m is None:
            continue
        board = int(m.group("board"))
        frame = int(m.group("frame"))
        by_board.setdefault(board, []).append((frame, png))
    for k in by_board:
        by_board[k].sort()
    return by_board


def stitch_board_image(
    frames: list[tuple[int, Path]], board_width: int, board_height: int, stride_px: int
) -> Image.Image:
    """Compose a wide board image from per-frame PNGs.

    Each frame is pasted at (frame_idx * stride_px, 0). Adjacent frames
    overlap by (frame_width - stride_px) px; the second paste overwrites
    the first in the overlap region, which is fine since the content is
    nominally identical (50% overlap of the same physical board).
    """
    canvas = Image.new("RGB", (board_width, board_height), (0, 0, 0))
    for frame_idx, png_path in frames:
        with Image.open(png_path) as fr:
            fr = fr.convert("RGB")
            canvas.paste(fr, (frame_idx * stride_px, 0))
    return canvas


def load_polygons(board_json: Path) -> list[list[tuple[int, int]]]:
    """Read the per-board JSON written by `knots run` / `knots eval`."""
    with board_json.open() as f:
        data = json.load(f)
    out: list[list[tuple[int, int]]] = []
    for knot in data.get("knots", []):
        poly = knot.get("polygon", [])
        if len(poly) >= 3:
            out.append([(int(p[0]), int(p[1])) for p in poly])
    return out


def draw_polygons(
    img: Image.Image,
    polygons: list[list[tuple[int, int]]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    """Draw polygon outlines on `img` in-place."""
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        draw.polygon(poly, outline=color, width=width)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with stage_timer(STAGE) as t:
        if not args.pred_dir.is_dir():
            raise SystemExit(f"--pred-dir not found: {args.pred_dir}")
        pred_jsons = sorted(args.pred_dir.glob("*.json"))
        # Filter out run_meta_*.json that other stages may co-locate here.
        pred_jsons = [p for p in pred_jsons if not p.stem.startswith("run_meta")]
        if not pred_jsons:
            raise SystemExit(f"no per-board prediction JSONs in {args.pred_dir}")

        if not args.images_dir.is_dir():
            raise SystemExit(f"--images-dir not found: {args.images_dir}")
        frames_by_board = collect_board_frames(args.images_dir)
        if not frames_by_board:
            raise SystemExit(f"no PNG frames under {args.images_dir}")

        gt_available = args.gt_dir.is_dir() and any(args.gt_dir.glob("[0-9]*.json"))
        if not gt_available:
            print(
                f"NB: no GT JSONs under {args.gt_dir} — rendering pred-only",
                file=sys.stderr,
            )

        n_written = 0
        n_skipped = 0
        n_no_frames = 0
        for pred_json in iter_with_progress(pred_jsons, label="boards"):
            try:
                board = int(pred_json.stem)
            except ValueError:
                continue
            if board not in frames_by_board:
                print(
                    f"  WARN board {board}: no source frames under {args.images_dir}",
                    file=sys.stderr,
                )
                n_no_frames += 1
                continue
            out_path = args.output_dir / f"{board}.jpg"
            if out_path.exists() and not args.force:
                n_skipped += 1
                continue
            with pred_json.open() as f:
                pred_data = json.load(f)
            board_w = int(pred_data["board_width"])
            board_h = int(pred_data["board_height"])
            stride = int(pred_data.get("stride_px", 320))

            canvas = stitch_board_image(frames_by_board[board], board_w, board_h, stride)

            # GT first so predictions overlay on top — reviewer's eye lands
            # on the model's output, with GT as the reference underneath.
            if gt_available:
                gt_path = args.gt_dir / f"{board}.json"
                if gt_path.exists():
                    draw_polygons(canvas, load_polygons(gt_path), args.gt_color, args.line_width)
            draw_polygons(canvas, load_polygons(pred_json), args.pred_color, args.line_width)

            canvas.save(out_path, "JPEG", quality=args.jpeg_quality, optimize=True)
            n_written += 1

        print(
            f"\nResult\n"
            f"  boards written={n_written}  skipped={n_skipped}  "
            f"no-frames={n_no_frames}\n"
            f"  output dir: {args.output_dir}",
            file=sys.stderr,
        )

    save_run_meta(args.output_dir, STAGE, args, elapsed_sec=t["elapsed_sec"])


if __name__ == "__main__":
    main()
