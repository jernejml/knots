#!/usr/bin/env python3
"""Export a trained YOLOv11-seg best.pt to ONNX for the C++ inference path.

NMS is baked into the ONNX graph (`nms=True`). The C++ side just feeds an
image and reads final boxes/masks — no postprocessing required. The cost
is that the confidence/IoU thresholds become fixed at export time; we
accept that for now.

Defaults: load the most-recent best.pt under runs/segment/*/weights/ and
write to models/best.onnx as the stable handoff path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--weights", type=Path, default=None,
                    help="Path to best.pt; default = most recent under --runs-dir.")
    ap.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs" / "segment",
                    help="Where to look for runs (auto-pick latest best.pt).")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "models",
                    help="Stable destination for the exported ONNX file.")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Must match training imgsz; YOLO letterboxes to this square.")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version; 17 has wider runtime compatibility than 19+.")
    ap.add_argument("--simplify", action="store_true",
                    help="Run onnxslim on the graph (requires onnxslim installed).")
    ap.add_argument("--device", default="cpu",
                    help="Export device; CPU is fine and avoids GPU init.")
    args = ap.parse_args()

    weights = args.weights or latest_best_weights(args.runs_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / "best.onnx"

    print(f"source weights: {rel_to_root(weights)}")
    print(f"target ONNX:    {rel_to_root(dst)}")
    print(f"input size:     {args.imgsz}x{args.imgsz}  (fixed)")
    print(f"opset:          {args.opset}")
    print(f"NMS:            baked into the graph (nms=True)\n")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
        half=False,
        nms=True,
        device=args.device,
    ))
    if exported.resolve() != dst.resolve():
        dst.write_bytes(exported.read_bytes())
    size_mb = dst.stat().st_size / 1e6
    print(f"\nexported → {rel_to_root(dst)}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
