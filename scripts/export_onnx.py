#!/usr/bin/env python3
"""Export a trained YOLOv11-seg best.pt to ONNX for the C++ inference path.

NMS is baked into the ONNX graph (`nms=True`). The C++ side just feeds an
image and reads final boxes/masks — no postprocessing required. The cost
is that the confidence/IoU thresholds become fixed at export time; we
accept that for now.

Defaults: load the most-recent best.pt under out/runs/segment/*/weights/
and write best.onnx **next to that best.pt** so each training run owns its
own exported model. A relative symlink at out/models/best.onnx is refreshed
to point at the freshest export, so callers using the fixed path
(e.g. `knots eval --model out/models/best.onnx`) keep working.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from stage_util import (
    add_config_arg,
    apply_config_defaults,
    load_config_section,
    save_run_meta,
    stage_timer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = "export_onnx"


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latest_best_weights(runs_dir: Path) -> Path:
    candidates = list(runs_dir.glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"no best.pt under {runs_dir}/")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    add_config_arg(pre)
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arg(ap)
    ap.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to best.pt; default = most recent under --runs-dir.",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "out" / "runs" / "segment",
        help="Where to look for runs (auto-pick latest best.pt).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write best.onnx; default = next to best.pt under the "
        "training run's weights/ dir. out/models/best.onnx is always "
        "refreshed as the 'latest' symlink regardless.",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Must match training imgsz; YOLO letterboxes to this square.",
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version; 17 has wider runtime compatibility than 19+.",
    )
    ap.add_argument(
        "--simplify",
        action="store_true",
        help="Run onnxslim on the graph (requires onnxslim installed).",
    )
    ap.add_argument(
        "--device", default="cpu", help="Export device; CPU is fine and avoids GPU init."
    )
    apply_config_defaults(ap, load_config_section(pre_args.config, STAGE))
    args = ap.parse_args()

    with stage_timer(STAGE) as timing:
        out_dir = _run(args)
    save_run_meta(out_dir, STAGE, args, elapsed_sec=timing["elapsed_sec"])


LATEST_LINK = REPO_ROOT / "out" / "models" / "best.onnx"


def _refresh_latest_link(target: Path) -> None:
    """Update LATEST_LINK to point at `target` via a relative symlink."""
    LATEST_LINK.parent.mkdir(parents=True, exist_ok=True)
    target_resolved = target.resolve()
    # If LATEST_LINK is itself the target file (e.g. user passed --out-dir
    # out/models explicitly), don't try to symlink a file onto itself.
    if LATEST_LINK.resolve(strict=False) == target_resolved:
        return
    if LATEST_LINK.is_symlink() or LATEST_LINK.exists():
        LATEST_LINK.unlink()
    rel = os.path.relpath(target_resolved, LATEST_LINK.parent.resolve())
    LATEST_LINK.symlink_to(rel)


def _run(args: argparse.Namespace) -> Path:
    weights = args.weights or latest_best_weights(args.runs_dir)
    out_dir = args.out_dir if args.out_dir is not None else weights.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "best.onnx"

    print(f"source weights: {rel_to_root(weights)}")
    print(f"target ONNX:    {rel_to_root(dst)}")
    print(f"input size:     {args.imgsz}x{args.imgsz}  (fixed)")
    print(f"opset:          {args.opset}")
    print(f"NMS:            baked into the graph (nms=True)\n")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=False,
            half=False,
            nms=True,
            device=args.device,
        )
    )
    if exported.resolve() != dst.resolve():
        dst.write_bytes(exported.read_bytes())
    size_mb = dst.stat().st_size / 1e6
    print(f"\nexported → {rel_to_root(dst)}  ({size_mb:.1f} MB)")

    _refresh_latest_link(dst)
    if LATEST_LINK.is_symlink():
        print(f"latest pointer: {rel_to_root(LATEST_LINK)} → {os.readlink(LATEST_LINK)}")

    return out_dir


if __name__ == "__main__":
    main()
