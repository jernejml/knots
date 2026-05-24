# Knot-detection model — `model.onnx`

`out/models/model.onnx` is a YOLOv11-seg model that predicts per-frame knot
polygons. The C++ runtime consumes it directly, so a reviewer can run inference
without retraining:

```bash
./run.sh runtime        # infer + eval + viz, using this committed model
```

## How it was built

The offline pipeline (`./run.sh offline` = prepare → sam → train):

| Step    | What                                                                |
|---------|---------------------------------------------------------------------|
| prepare | Stratified board-level train/val/test split (seed 42, 80/10/10)     |
| sam     | SAM2 (`sam2.1_s.pt`) turns the rectangle labels into polygon labels |
| train   | YOLOv11-seg fine-tuning, then ONNX export                           |

Training settings (from this run's `run_meta_train_yolo.json`):

- base model:        `yolo11n-seg.pt`
- epochs:            50  (early-stopping patience 10)
- image size:        640
- batch:             32
- exported:          `last` checkpoint → ONNX, opset 17 (NMS baked into the graph)
- train wall time:   ~15.5 min on a single GPU

`export_weights = last` (not `best`) because the validation labels are SAM
pseudo-labels, so best-by-val-metric selection is a weak signal. The knob lives
in `configs/default.toml` if you want to flip it.

## Reproduce

```bash
./run.sh offline        # rebuilds out/models/model.onnx from data/
```

- Full provenance for any training run is dumped to
  `out/runs/segment/<name>/run_meta_train_yolo.json` (resolved args + timing).
- Metrics for the committed model: `./run.sh eval` → `out/boards/eval_boards.json`.

> Note: these figures describe the currently committed model. If you retrain,
> regenerate this card (or read the run's `run_meta_train_yolo.json`).
