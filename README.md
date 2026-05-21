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

All generated artefacts (analysis JSONs, SAM labels, training runs, the
exported ONNX model, per-frame and per-board JSONs, visualisations) live
under one ignored `out/` tree. The C++ inference path expects
`out/models/best.onnx`, produced by the offline Python pipeline.

The simplest path is the top-level orchestrator, which chains every stage
end-to-end on the default test split:

```bash
./run.sh                       # full pipeline, --split test
SPLIT=val ./run.sh             # evaluate on val instead of test
SKIP_TRAIN=1 ./run.sh          # reuse existing best.pt under out/runs/segment/
RUN_NAME=iter5 ./run.sh        # name the training run; exported ONNX and
                               # eval JSON co-locate under out/runs/segment/iter5/
CONFIG=configs/foo.toml ./run.sh   # alternate TOML; empty disables --config
```

Each training run owns its artefacts: `out/runs/segment/<name>/` holds the
weights, `best.onnx` next to `best.pt`, every stage's `run_meta_*.json`, and
the eval JSON. `out/models/best.onnx` is a relative symlink to the latest
export, so consumers using the fixed path still work. Use
`python3 scripts/list_runs.py` to see a one-row-per-run table.

To drive a single stage by hand (useful for iteration), each script accepts
`--config configs/default.toml` and standard CLI flags. The Python pipeline,
in dependency order:

```bash
docker run --rm \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-data python3 scripts/analyze_dataset.py

docker run --rm \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-data python3 scripts/board_features.py

docker run --rm \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-data python3 scripts/make_splits.py

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-train python3 scripts/sam_polygons.py

docker run --rm --gpus all --ipc=host \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/out:/work/out" \
  knots-train python3 scripts/train_yolo.py

docker run --rm --gpus all --ipc=host \
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
