#!/usr/bin/env python3
"""Render per-board comparison composites.

For each requested board this script:
  1. Loads all {board}_{frame}.png frames, sorts by frame_idx.
  2. Stitches them into a board image: each frame pasted at
     (frame_idx * stride_px, 0). Adjacent frames overlap by 50% and overwrite
     each other; that's fine because both frames show the same pixels there.
  3. Renders one or more overlay panels (raw, bbox, knots) and stacks them
     vertically (default) or horizontally.
  4. Optionally downscales to --max-width — boards can be 10000+ px wide.

Built-in panels:
  raw    stitched board image, no overlay
  bbox   stitched + GT bboxes from data/labels/{board}_{frame}.txt
         (each bbox translated by frame_idx * stride_px into board coords)
  knots  stitched + per-board polygons from <boards-pred-dir>/{board}.json
         (output of `knots stitch`; coords already in board space)

Inputs (all read-only):
  --data-dir/images/{board}_{frame}.png
  --bbox-labels-dir/{board}_{frame}.txt
  --boards-pred-dir/{board}.json

Output:
  --output-dir/{board}.png

Board IDs come from exactly one of --board, --boards, --boards-file, or --all
(discover from --boards-pred-dir contents).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]

STRIDE_PX_DEFAULT = 320

FRAME_RE = re.compile(r"^(\d+)_(\d+)\.png$")


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_hex_color(s: str) -> tuple[int, int, int]:
    raw = s.strip().lstrip("#")
    if len(raw) != 6:
        raise argparse.ArgumentTypeError(f"expected #rrggbb, got {s!r}")
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def parse_yolo_bboxes(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Return frame-local (x1, y1, x2, y2) per bbox."""
    if not label_path.exists():
        return []
    out: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h
        x1 = max(0, int(round(cx - bw / 2)))
        y1 = max(0, int(round(cy - bh / 2)))
        x2 = min(w, int(round(cx + bw / 2)))
        y2 = min(h, int(round(cy + bh / 2)))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def load_frames_for_board(images_dir: Path, board: int) -> list[tuple[int, Path]]:
    """Return sorted [(frame_idx, image_path)] for `board`."""
    result: list[tuple[int, Path]] = []
    for entry in images_dir.iterdir():
        m = FRAME_RE.match(entry.name)
        if not m:
            continue
        if int(m.group(1)) != board:
            continue
        result.append((int(m.group(2)), entry))
    result.sort(key=lambda t: t[0])
    return result


def stitch_board(images_dir: Path, board: int,
                 stride_px: int) -> tuple[Image.Image, list[tuple[int, int, int]]] | None:
    """Build the stitched board image. Returns (canvas, frame_info) or None.

    `frame_info` is a list of (frame_idx, frame_w, frame_h) the overlay panels
    need to denormalise per-frame labels back to pixel coords.
    """
    frames = load_frames_for_board(images_dir, board)
    if not frames:
        return None

    # Probe height + last-frame width without keeping all images open.
    with Image.open(frames[0][1]) as first:
        board_h = first.height
    last_idx, last_path = frames[-1]
    with Image.open(last_path) as last_img:
        last_w = last_img.width
    board_w = last_idx * stride_px + last_w

    canvas = Image.new("RGB", (board_w, board_h), (0, 0, 0))
    frame_info: list[tuple[int, int, int]] = []
    for frame_idx, path in frames:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            if rgb.height != board_h:
                print(f"  WARN board {board} frame {frame_idx}: "
                      f"H={rgb.height} vs board H={board_h}", file=sys.stderr)
            canvas.paste(rgb, (frame_idx * stride_px, 0))
            frame_info.append((frame_idx, rgb.width, rgb.height))
    return canvas, frame_info


def render_raw(canvas: Image.Image, **_kwargs) -> Image.Image:
    return canvas.copy()


def render_bbox(canvas: Image.Image, *,
                bbox_labels_dir: Path, board: int,
                frame_info: list[tuple[int, int, int]],
                stride_px: int,
                bbox_color: tuple[int, int, int],
                **_kwargs) -> Image.Image:
    out = canvas.copy()
    draw = ImageDraw.Draw(out)
    for frame_idx, w, h in frame_info:
        boxes = parse_yolo_bboxes(bbox_labels_dir / f"{board}_{frame_idx}.txt", w, h)
        dx = frame_idx * stride_px
        for (x1, y1, x2, y2) in boxes:
            draw.rectangle([(x1 + dx, y1), (x2 + dx - 1, y2 - 1)],
                           outline=bbox_color, width=2)
    return out


def render_knots(canvas: Image.Image, *,
                 boards_pred_dir: Path, board: int,
                 knots_color: tuple[int, int, int],
                 knots_alpha: float,
                 **_kwargs) -> Image.Image:
    path = boards_pred_dir / f"{board}.json"
    if not path.exists():
        return canvas.copy()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  WARN board {board}: invalid JSON in {path}", file=sys.stderr)
        return canvas.copy()

    base = canvas.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    fill_alpha = max(0, min(255, int(round(255 * knots_alpha))))
    fill = knots_color + (fill_alpha,)
    outline = knots_color + (255,)
    o_draw = ImageDraw.Draw(overlay)
    for knot in data.get("knots", []):
        poly = knot.get("polygon", [])
        pts = [(int(x), int(y)) for x, y in poly]
        if len(pts) >= 3:
            o_draw.polygon(pts, fill=fill, outline=outline)
    return Image.alpha_composite(base, overlay).convert("RGB")


PANELS = {
    "raw": render_raw,
    "bbox": render_bbox,
    "knots": render_knots,
}


def add_label(panel: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if font is not None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = 6 * len(text), 11
    pad = 3
    box_w, box_h = tw + 2 * pad, th + 2 * pad
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return panel


def compose(panels: list[Image.Image], direction: str, sep_px: int = 2) -> Image.Image:
    if direction == "vertical":
        width = max(p.width for p in panels)
        total_h = sum(p.height for p in panels) + sep_px * (len(panels) - 1)
        canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
        y = 0
        for p in panels:
            canvas.paste(p, (0, y))
            y += p.height + sep_px
        return canvas
    height = max(p.height for p in panels)
    total_w = sum(p.width for p in panels) + sep_px * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + sep_px
    return canvas


def downscale(img: Image.Image, max_width: int) -> Image.Image:
    if max_width <= 0 or img.width <= max_width:
        return img
    new_h = int(round(img.height * max_width / img.width))
    return img.resize((max_width, new_h), Image.LANCZOS)


def load_board_ids(args: argparse.Namespace) -> list[int]:
    if args.board is not None:
        return [args.board]
    if args.boards is not None:
        return [int(s.strip()) for s in args.boards.split(",") if s.strip()]
    if args.boards_file is not None:
        ids: list[int] = []
        for line in args.boards_file.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(int(s))
        return ids
    if args.all:
        ids = []
        if args.boards_pred_dir.is_dir():
            for entry in args.boards_pred_dir.iterdir():
                if entry.suffix == ".json" and entry.stem.isdigit():
                    ids.append(int(entry.stem))
        return sorted(ids)
    return []


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--bbox-labels-dir", type=Path, default=None,
                    help="YOLO bbox labels dir (default: --data-dir/labels).")
    ap.add_argument("--boards-pred-dir", type=Path, default=REPO_ROOT / "boards_out",
                    help="Per-board JSONs from `knots stitch`.")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "viz" / "boards",
                    help="Composites are written here.")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--board", type=int, default=None, help="Single board ID.")
    src.add_argument("--boards", type=str, default=None,
                     help="Comma-separated board IDs, e.g. 0,5,100.")
    src.add_argument("--boards-file", type=Path, default=None,
                     help="File with one board ID per line; '#' comments allowed.")
    src.add_argument("--all", action="store_true",
                     help="Render every board with a JSON in --boards-pred-dir.")

    ap.add_argument("--panels", type=str, default="raw,bbox,knots",
                    help=f"Comma list of panel names in render order. "
                         f"Available: {','.join(PANELS)}.")
    ap.add_argument("--stack", choices=("vertical", "horizontal"), default="vertical")
    ap.add_argument("--label-panels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stride-px", type=int, default=STRIDE_PX_DEFAULT,
                    help="Frame stride; must match `knots stitch` (default 320).")
    ap.add_argument("--max-width", type=int, default=4096,
                    help="Downscale composite to fit this width. 0 = no downscale.")
    ap.add_argument("--bbox-color", type=parse_hex_color, default=parse_hex_color("#ff3030"))
    ap.add_argument("--knots-color", type=parse_hex_color, default=parse_hex_color("#00e676"))
    ap.add_argument("--knots-alpha", type=float, default=0.35)
    args = ap.parse_args()

    images_dir = args.data_dir / "images"
    bbox_dir = args.bbox_labels_dir or (args.data_dir / "labels")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_names = [s.strip() for s in args.panels.split(",") if s.strip()]
    for name in panel_names:
        if name not in PANELS:
            raise SystemExit(f"unknown panel {name!r}; available: {','.join(PANELS)}")

    board_ids = load_board_ids(args)
    if not board_ids:
        raise SystemExit("no boards to render")

    print(f"compare_boards: {len(board_ids)} board(s)  panels={','.join(panel_names)}  stack={args.stack}")
    print(f"  output={rel_to_root(args.output_dir)}/")

    n_ok = 0
    for board in board_ids:
        result = stitch_board(images_dir, board, args.stride_px)
        if result is None:
            print(f"  WARN board {board}: no frames found", file=sys.stderr)
            continue
        canvas, frame_info = result

        panel_imgs: list[Image.Image] = []
        for name in panel_names:
            panel = PANELS[name](
                canvas,
                bbox_labels_dir=bbox_dir,
                boards_pred_dir=args.boards_pred_dir,
                board=board,
                frame_info=frame_info,
                stride_px=args.stride_px,
                bbox_color=args.bbox_color,
                knots_color=args.knots_color,
                knots_alpha=args.knots_alpha,
            )
            if args.label_panels:
                panel = add_label(panel, name)
            panel_imgs.append(panel)

        composite = compose(panel_imgs, args.stack)
        composite = downscale(composite, args.max_width)
        out_path = args.output_dir / f"{board}.png"
        composite.save(out_path)
        n_ok += 1

    print(f"wrote {n_ok}/{len(board_ids)} composites to {rel_to_root(args.output_dir)}/")


if __name__ == "__main__":
    main()
