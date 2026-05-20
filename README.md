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

All generated artefacts (analysis CSVs, SAM labels, training runs, the
exported ONNX model, per-frame and per-board JSONs, visualisations) live
under one ignored `out/` tree. The C++ inference path expects
`out/models/best.onnx`, produced by the offline Python pipeline:

```bash
docker run --rm \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-data make all

docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-train python3 scripts/sam_polygons.py

docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-train python3 scripts/train_yolo.py

docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-train python3 scripts/export_onnx.py
```

With `out/models/best.onnx` in place, the test mode is a single
invocation — inference, GT stitching, and metric aggregation happen in
one process:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-infer knots eval \
    --model /work/out/models/best.onnx \
    --images-dir /work/data/images \
    --labels-dir /work/data/labels \
    --splits-csv /work/out/analysis/splits.csv --split test
```

To produce per-board polygons without evaluation, use `knots run`:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-infer knots run \
    --model /work/out/models/best.onnx \
    --input-dir /work/data/images \
    --output-dir /work/out/boards/pred \
    --splits-csv /work/out/analysis/splits.csv --split test
```

For finer-grained debugging, the pipeline can also be run stage-by-stage:
`knots infer` (per-frame JSONs in `out/frames/`) → `knots stitch`
(per-board JSONs in `out/boards/pred/`) → `knots gt-stitch` (per-board
GT in `out/boards/gt/`) → `knots eval --pred-dir P --gt-dir G`. Use the
stage-by-stage form to inspect intermediate outputs or resume at frame
granularity.

Each script and `knots` subcommand accepts `--help` for full options.
