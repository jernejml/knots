#include "knots/cli_options.hpp"

#include "knots/commands.hpp"

namespace knots::cli {

void AddInferenceOpts(CLI::App* app, InferenceOpts& opts) {
    app->add_option("--model", opts.model, "ONNX model path")->required();
    app->add_option("--conf", opts.conf, "confidence threshold")
        ->capture_default_str();
    app->add_flag("--cpu", opts.cpu_only, "force CPU execution provider");
}

void AddStitchOpts(CLI::App* app, StitchOpts& opts) {
    app->add_option("--stride-px", opts.stride_px, "frame stride in board px")
        ->capture_default_str();
    app->add_option("--simplify-eps", opts.simplify_eps_px,
                    "approxPolyDP eps in board px")
        ->capture_default_str();
}

void AddBoardsFilterOpts(CLI::App* app, BoardsFilter& f) {
    app->add_option("--boards", f.boards, "board IDs to restrict to")
        ->delimiter(',');
    app->add_option("--boards-file", f.boards_file,
                    "one board ID per line ('#' comments allowed)");
}

void AddSplitsFilterOpts(CLI::App* app, SplitsFilter& f) {
    auto* csv = app->add_option("--splits-csv", f.splits_csv, "analysis/splits.csv");
    auto* sp =
        app->add_option("--split", f.split, "restrict to this split")
            ->check(CLI::IsMember({"train", "val", "test"}));
    // CLI11 enforces "--splits-csv requires --split" symmetrically: if either
    // is given, both must be. The runtime check the old code had ("--splits-
    // csv requires --split") was one-directional; tightening to both
    // directions catches "--split test" with no splits.csv too.
    csv->needs(sp);
    sp->needs(csv);
}

void AddFramesFilterOpts(CLI::App* app, FramesFilter& f) {
    app->add_option("--frames", f.frames,
                    "frame stems to restrict to, e.g. 0_5,100_3")
        ->delimiter(',');
    app->add_option("--frames-file", f.frames_file,
                    "one frame stem per line");
}

// -- Per-subcommand assemblies -----------------------------------------------

void AddRunOptions(CLI::App* app, RunArgs& args) {
    AddInferenceOpts(app, args.inference);
    app->add_option("--input-dir", args.input_dir, "frame PNGs")->required();
    app->add_option("--output-dir", args.output_dir, "per-board JSON outputs")
        ->required();
    AddFramesFilterOpts(app, args.frames);
    AddSplitsFilterOpts(app, args.splits);
    AddStitchOpts(app, args.stitch);
    app->add_flag("--force", args.force, "overwrite existing per-board outputs");
}

void AddInferOptions(CLI::App* app, InferArgs& args) {
    AddInferenceOpts(app, args.inference);
    app->add_option("--input-dir", args.input_dir, "frame PNGs")->required();
    app->add_option("--output-dir", args.output_dir, "per-frame JSON outputs")
        ->required();
    AddFramesFilterOpts(app, args.frames);
    AddSplitsFilterOpts(app, args.splits);
    app->add_flag("--force", args.force, "overwrite existing outputs");
}

void AddStitchOptions(CLI::App* app, StitchArgs& args) {
    app->add_option("--input-dir", args.input_dir,
                    "per-frame JSONs from `knots infer`")
        ->required();
    app->add_option("--output-dir", args.output_dir,
                    "per-board JSONs (one .json per board)")
        ->required();
    AddStitchOpts(app, args.stitch);
    app->add_flag("--force", args.force, "overwrite existing outputs");
}

void AddGtStitchOptions(CLI::App* app, GtStitchArgs& args) {
    app->add_option("--labels-dir", args.labels_dir,
                    "YOLO bbox labels (per-frame .txt)")
        ->required();
    app->add_option("--images-dir", args.images_dir,
                    "frame PNGs (for frame dimensions)")
        ->required();
    app->add_option("--output-dir", args.output_dir, "per-board GT JSONs")
        ->required();
    AddStitchOpts(app, args.stitch);
    AddBoardsFilterOpts(app, args.boards);
    AddSplitsFilterOpts(app, args.splits);
    app->add_flag("--force", args.force, "overwrite existing outputs");
}

void AddEvalOptions(CLI::App* app, EvalArgs& args) {
    // Mode A.
    app->add_option("--pred-dir", args.pred_dir,
                    "Mode A: per-board prediction JSONs");
    app->add_option("--gt-dir", args.gt_dir,
                    "Mode A: per-board GT JSONs");

    // Mode B. --model / --conf / --cpu are not required at the CLI11 level
    // because eval can run in Mode A without a model; the callback validates
    // mode coherence after parse.
    app->add_option("--model", args.inference.model, "Mode B: ONNX model");
    app->add_option("--conf", args.inference.conf,
                    "Mode B: confidence threshold")
        ->capture_default_str();
    app->add_flag("--cpu", args.inference.cpu_only,
                  "Mode B: force CPU execution provider");
    app->add_option("--images-dir", args.images_dir, "Mode B: frame PNGs");
    app->add_option("--labels-dir", args.labels_dir,
                    "Mode B: YOLO bbox labels");
    AddStitchOpts(app, args.stitch);
    AddSplitsFilterOpts(app, args.splits);

    // Shared.
    AddBoardsFilterOpts(app, args.boards);
    app->add_option("--out", args.out_json,
                    "JSON output path (default out/analysis/eval_boards.json)");
    app->add_option("--match-iou", args.match_iou,
                    "bbox IoU threshold for matching")
        ->capture_default_str();
    app->add_flag("--no-write", args.no_write,
                  "skip JSON output; print to stdout only");
}

}  // namespace knots::cli
