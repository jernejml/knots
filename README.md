# knots

Per-board wood-knot polygon extraction from sequential color-camera frames.

Frames tile each board along its long axis with a 50 % overlap (frame width
640 px, stride 320 px), so every physical knot is seen in roughly two
adjacent frames. The pipeline upgrades the provided rectangle annotations
to polygons with SAM2, trains a YOLOv11-seg model, runs it from C++ via
ONNX Runtime, and raster-unions per-frame polygons into per-board polygons.

## Build

Requires Docker + NVIDIA Container Toolkit (CUDA 12.8 base images):

```bash
docker build -f docker/Dockerfile.data  -t knots-data  .   # dataset analysis
docker build -f docker/Dockerfile.train -t knots-train .   # SAM2 + YOLO + ONNX export
docker build -f docker/Dockerfile.infer -t knots-infer .   # C++ knots binary (build + run)
```

`knots-infer` is the image that builds and runs the C++ program; the
other two cover the offline Python steps. The C++ binary is built
inside `knots-infer` during `docker build` - no host toolchain needed.

## Run

The C++ inference path expects `models/best.onnx`, which is produced by
the offline Python pipeline (analysis → splits → SAM2 polygon upgrade →
YOLOv11-seg training → ONNX export):

```bash
docker run --rm \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/analysis:/work/analysis" \
  knots-data make all

docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/labels_seg:/work/labels_seg" \
  knots-train python3 scripts/sam_polygons.py

docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/labels_seg_l:/work/labels_seg_l" \
  -v "$PWD/analysis:/work/analysis:ro" \
  -v "$PWD/yolo_dataset:/work/yolo_dataset" \
  -v "$PWD/runs:/work/runs" \
  knots-train python3 scripts/train_yolo.py

docker run --rm --gpus all \
  -v "$PWD/runs:/work/runs:ro" \
  -v "$PWD/models:/work/models" \
  knots-train python3 scripts/export_onnx.py
```

With `models/best.onnx` in place, run inference and the test mode:

```bash
# Per-frame inference.
docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/models:/work/models:ro" \
  -v "$PWD/analysis:/work/analysis:ro" \
  -v "$PWD/cpp_out:/work/cpp_out" \
  knots-infer knots infer \
    --model /work/models/best.onnx \
    --input-dir /work/data/images \
    --output-dir /work/cpp_out \
    --splits-csv /work/analysis/splits.csv --split test

# Stitch per-frame results into per-board polygons.
docker run --rm \
  -v "$PWD/cpp_out:/work/cpp_out:ro" \
  -v "$PWD/boards_out:/work/boards_out" \
  knots-infer knots stitch \
    --input-dir /work/cpp_out --output-dir /work/boards_out

# Test mode: stitch GT bboxes through the same pipeline, then compare.
docker run --rm \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/analysis:/work/analysis:ro" \
  -v "$PWD/boards_gt:/work/boards_gt" \
  knots-infer knots gt-stitch \
    --labels-dir /work/data/labels \
    --images-dir /work/data/images \
    --output-dir /work/boards_gt \
    --splits-csv /work/analysis/splits.csv --split test

docker run --rm \
  -v "$PWD/boards_out:/work/boards_out:ro" \
  -v "$PWD/boards_gt:/work/boards_gt:ro" \
  -v "$PWD/analysis:/work/analysis" \
  knots-infer knots eval \
    --pred-dir /work/boards_out --gt-dir /work/boards_gt
```

Each script and `knots` subcommand accepts `--help` for full options.
