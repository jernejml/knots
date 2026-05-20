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

With `models/best.onnx` in place, run the one-shot pipeline and the test mode:

```bash
# Inference + per-board stitching in one pass (no intermediate per-frame JSON).
docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/models:/work/models:ro" \
  -v "$PWD/analysis:/work/analysis:ro" \
  -v "$PWD/boards_out:/work/boards_out" \
  knots-infer knots run \
    --model /work/models/best.onnx \
    --input-dir /work/data/images \
    --output-dir /work/boards_out \
    --splits-csv /work/analysis/splits.csv --split test

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

For finer-grained control, the pipeline can be run as two stages instead
of one: `knots infer --output-dir <per-frame JSONs>` followed by
`knots stitch --input-dir <per-frame JSONs> --output-dir <per-board JSONs>`.
Use the two-stage form when you want to inspect raw per-frame detections
or resume an interrupted run at frame granularity.

Each script and `knots` subcommand accepts `--help` for full options.
