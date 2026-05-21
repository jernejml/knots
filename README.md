# knots

Per-board wood-knot polygon extraction from sequential color-camera frames.

Frames overlap 50 % — width 640 px, stride 320 px, so every board pixel is
seen twice. Input is those frames plus rectangle annotations; output is
per-board polygons for every knot.

Three design choices shape the rest:

- **SAM2 amplifies rectangle labels into polygon labels offline**, so the
  segmentation model trains on polygons without anyone hand-tracing them.
- **Cross-frame overlap is collapsed by raster-union, not polygon matching.**
  Each frame's polygons are pasted onto a board-sized mask;
  `findContours(RETR_EXTERNAL)` re-vectorises whatever fused. One
  `fillPoly` call replaces an N²-pair cross-frame deduplicator.
- **The Python ↔ C++ boundary is one ONNX file.** Python does data prep
  and training (SAM2, Ultralytics YOLOv11-seg); a small C++ binary on
  OpenCV + ONNX Runtime owns the runtime pipeline.

The dataset isn't committed; populate `data/images/` and `data/labels/`
after cloning.

## 1. Prerequisites

Verifies the host can run the pipeline (bash, docker, NVIDIA stack, end-to-end probe).

```bash
./prerequisites.sh   # use --help for additional options
```

## 2. Running the pipeline

Single entry point; defaults to running every stage end-to-end.

```bash
./run.sh   # use --help for stages, env vars, and examples
```

## 3. Architecture

### Pipeline data flow

```
Training (offline)

  analyze ──► features ──► split ─┐
                                  │
  data/labels/ ──► sam ───────────┤
                                  ▼
                                train ──► export ──► out/models/best.onnx


Runtime (uses out/models/best.onnx)

  data/images/ ──► infer ──► out/boards/pred/<board>.json
                                            │
  data/labels/ ──► gt    ──► out/boards/gt/<board>.json
                                            │
                                            ├──► viz  ──► out/boards/viz/<board>.jpg
                                            │
                                            └──► eval ──► eval_boards.json
```

`infer` and `gt` are independent. `viz` consumes both (predictions in
green, GT in red on the stitched board image); `eval` consumes both via
`knots eval` Mode A — bbox-IoU greedy matching + per-pair mask IoU on the
cached per-board polygons. No inference at eval time, no GPU.

### Docker images

| Image | Base / contents | Used by stages |
|---|---|---|
| `knots-data` | Python 3.13 slim + NumPy / SciPy / Pillow | analyze, features, split, viz |
| `knots-train` | PyTorch 2.7 + CUDA 12.8 + Ultralytics (YOLO + SAM2) | sam, train, export |
| `knots-infer` | CUDA 12.8 + cuDNN + ONNX Runtime GPU + the `knots` C++ binary | infer, gt, eval |

### Generated artefacts (`out/`)

- `out/analysis/` — per-frame / per-board / split JSONs
- `out/runs/segment/<name>/` — weights, ONNX, `run_meta_*.json`, eval JSON
- `out/models/best.onnx` — symlink to the latest export
- `out/boards/pred/` — per-board predicted polygons
- `out/boards/gt/` — per-board GT polygons
- `out/boards/viz/` — per-board stitched JPEGs with overlays

