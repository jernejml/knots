# knots

Per-board wood-knot polygon extraction from sequential color-camera frames.

Frames tile each board along its long axis with a 50 % overlap (640 px
wide, 320 px stride). The pipeline upgrades the provided rectangle
annotations to polygons with SAM2, trains a YOLOv11-seg model, runs it
from C++ via ONNX Runtime, and raster-unions per-frame polygons into
per-board polygons.

## 1. Prerequisites

```bash
./prerequisites.sh   # use --help for additional options
```

---

## 2. Running the pipeline

```bash
./run.sh   # use --help for stages, env vars, and examples
```

---

## 3. Architecture

### Pipeline data flow

```
analyze ──► features ──► split   ──►   out/analysis/splits.csv

sam ──► train ──► export                 (SAM2 polygons → YOLO seg → ONNX)
                    │
                    ▼
              out/models/best.onnx
                    │
                    ▼
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

### Per-board inference algorithm

For board B with N frames, frames overlap 50 % (width 640, stride 320):

```
   frame 0      frame 1      frame 2      ...     frame N-1
  [0,  640]   [320, 960]   [640, 1280]            [(N-1)·320, (N-1)·320+640]
     │            │            │                       │
     ▼            ▼            ▼                       ▼
   ┌─── YOLO11-seg via ONNX Runtime (per-frame inference) ───┐
     │            │            │                       │
     ▼            ▼            ▼                       ▼
   polys₀       polys₁       polys₂                 polys_{N-1}     (frame-local px)
                              │
                              ▼   translate by (frame_idx · 320, 0)
                              │
                              ▼   allocate board-sized mask
                              │
                              ▼   cv::fillPoly all translated polygons
                              │
                              ▼   cv::findContours(RETR_EXTERNAL) + approxPolyDP
                              │
                              ▼
                  per-board polygon list (board coords)
```

The raster-union (`fillPoly` + `findContours(RETR_EXTERNAL)`) dedups
polygons across the 50 % frame overlap automatically: touching or
overlapping shapes fuse into a single external contour. No explicit
cross-frame matching code is needed.

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

