# knots

Per-board wood-knot polygon extraction from sequential color-camera frames.

Frames overlap 50 % — width 640 px, stride 320 px, so every board pixel is
seen twice. Input is those frames plus rectangle annotations; output is
per-board polygons for every knot.

Three design choices shape the rest:

- **The supplied labels are just rectangles drawn around each knot. We use
  SAM2 (a general-purpose "outline-anything" model from Meta) to turn each
  rectangle into an outline of the knot inside it, once, up front.**
  The main model then trains on those outlines.
- **Duplicate detections across overlapping frames are merged at the pixel
  level, not by comparing shapes.** Adjacent frames overlap, so a knot near
  the seam usually gets predicted twice. We draw every prediction onto one
  board-sized image — overlapping shapes naturally fuse — then trace the
  merged outlines back out.
- **One file crosses the Python ↔ C++ boundary: a trained model exported
  to ONNX.** Python handles dataset analysis, SAM2 polygon generation, and
  YOLOv11-seg training, producing `out/models/model.onnx`. From there a small C++ binary
  on OpenCV + ONNX Runtime owns the runtime pipeline. No shared Python
  runtime; the model is the only thing that crosses.

## Output

The pipeline writes one JSON per board to `out/boards/pred/`:

```json
{
  "board": 0,
  "board_width": 19840,
  "board_height": 220,
  "stride_px": 320,
  "knots": [
    { "polygon": [[1240, 87], [1252, 84], [1261, 92], [1255, 101], [1242, 99]] }
  ]
}
```

Coordinates are integer pixel positions in the full stitched board, not
in any single frame. Ground-truth polygons under `out/boards/gt/` use the
same schema.

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

No trained model ships with the repo. The default `./run.sh` trains one
(hours on a single GPU).

The two main outputs:

- **Per-board polygons:** `./run.sh infer` → `out/boards/pred/<board>.json`
- **Test mode:** `./run.sh eval` → per-board table on stdout plus
  `eval_boards.json` (aggregate precision / recall / F1 / mean mask IoU,
  greedy bbox-IoU matching at 0.5).

Test mode against an arbitrary annotated directory is two steps: predict
polygons, then compare. `eval` rebuilds the per-board GT from the raw
bbox labels in the same pass (no separate gt-stitch step).

```bash
# 1. predicted polygons from the model
docker run --rm --gpus all \
  -v /path/to/test/data:/work/data:ro \
  -v "$PWD/out:/work/out" \
  knots-infer knots run \
    --model /work/out/models/model.onnx \
    --input-dir /work/data/images \
    --output-dir /work/out/boards/pred

# 2. compare against GT (stitched on the fly from the labels)
docker run --rm \
  -v /path/to/test/data:/work/data:ro \
  -v "$PWD/out:/work/out" \
  knots-infer knots eval \
    --pred-dir /work/out/boards/pred \
    --gt-dir   /work/out/boards/gt \
    --labels-dir /work/data/labels \
    --images-dir /work/data/images
```

Equivalently, with the data at `./data/`: `./run.sh infer eval`.

## 3. Architecture

### Pipeline data flow

```
Training (offline)

  data/{images,labels}/ ──► prepare ─┐
                                     │
  data/labels/ ──► sam ──────────────┤
                                     ▼
                                   train (trains + exports ONNX) ──► out/models/model.onnx


Runtime (uses out/models/model.onnx)

  data/images/ ──► infer ──► out/boards/pred/<board>.json
                                            │
  data/labels/ ─────────────────────────────┤
                                            ▼
                                          eval ──► out/boards/gt/<board>.json
                                            │       (rebuilds GT inline)
                                            │
                                            ├──► eval_boards.json
                                            │
                                            └──► viz ──► out/boards/viz/<board>.jpg
```

Stages at a glance:

- `prepare`   stratified train / val / test partition of boards, written
              to `out/analysis/partitions.json`. By board, not by frame —
              the 50 % overlap would otherwise leak knots between splits.
- `sam`       SAM2 turns bbox labels into polygon labels
- `train`     YOLOv11-seg fine-tuning on those polygons; exports a checkpoint
              (best or last, default last) to `out/models/model.onnx` in the
              same step
- `infer`     per-board predicted polygons from the trained model
- `eval`      precision / recall / F1 / IoU comparing pred vs GT. Stitches
              per-board GT from raw bbox labels (`out/boards/gt/`) as a
              precondition, using the same stitching primitive as `infer`
              so pred and GT are directly comparable
- `viz`       per-board JPEG with pred (green) and GT (red) overlays
              (opportunistic: drops the GT layer if `out/boards/gt/` is empty)

`eval` runs after `infer`. Bbox-IoU greedy matching + per-pair mask IoU on
the cached per-board polygons. No inference at eval time, no GPU.

### Docker images

| Image | Base / contents | Used by stages |
|---|---|---|
| `knots-data` | Python 3.13 slim + NumPy / SciPy / Pillow | prepare, viz |
| `knots-train` | PyTorch 2.7 + CUDA 12.8 + Ultralytics (YOLO + SAM2) | sam, train |
| `knots-infer` | CUDA 12.8 + cuDNN + ONNX Runtime GPU + the `knots` C++ binary | infer, eval |

### Generated artefacts (`out/`)

- `out/analysis/` — `partitions.json` (board → train/val/test) + `run_meta_prepare.json`
- `out/runs/segment/<name>/` — weights, ONNX, `run_meta_*.json`, eval JSON
- `out/models/model.onnx` — symlink to the latest export (best or last; run_meta records which)
- `out/boards/pred/` — per-board predicted polygons
- `out/boards/gt/` — per-board GT polygons
- `out/boards/viz/` — per-board stitched JPEGs with overlays

