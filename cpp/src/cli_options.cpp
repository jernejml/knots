#include "knots/cli_options.hpp"

#include "knots/commands.hpp"

namespace knots::cli {

void AddInferenceOpts(CLI::App* app, InferenceOpts& opts) {
    // Not ->required(): RunArgs validates conditionally — when
    // --from-frame-jsons is set, no inference happens and no model is needed.
    app->add_option("--model", opts.model, "ONNX model path");
    app->add_option("--conf", opts.conf, "confidence threshold")->capture_default_str();
    app->add_flag("--cpu", opts.cpu_only, "force CPU execution provider");
}

void AddStitchOpts(CLI::App* app, StitchOpts& opts) {
    app->add_option("--stride-px", opts.stride_px, "frame stride in board px")
        ->capture_default_str();
    app->add_option("--simplify-eps", opts.simplify_eps_px, "approxPolyDP eps in board px")
        ->capture_default_str();
}

void AddBoardsFilterOpts(CLI::App* app, BoardsFilter& f) {
    app->add_option("--boards", f.boards, "board IDs to restrict to")->delimiter(',');
    app->add_option("--boards-file", f.boards_file, "one board ID per line ('#' comments allowed)");
}

void AddSplitsFilterOpts(CLI::App* app, SplitsFilter& f) {
    auto* json =
        app->add_option("--partitions-json", f.partitions_json, "analysis/partitions.json");
    auto* sp = app->add_option("--split", f.split, "restrict to this split")
                   ->check(CLI::IsMember({"train", "val", "test"}));
    // CLI11 enforces "--partitions-json requires --split" symmetrically: if
    // either is given, both must be. Catches "--split test" with no
    // partitions file too.
    json->needs(sp);
    sp->needs(json);
}

void AddFramesFilterOpts(CLI::App* app, FramesFilter& f) {
    app->add_option("--frames", f.frames, "frame stems to restrict to, e.g. 0_5,100_3")
        ->delimiter(',');
    app->add_option("--frames-file", f.frames_file, "one frame stem per line");
}

// -- Per-subcommand assemblies -----------------------------------------------

void AddRunOptions(CLI::App* app, RunArgs& args) {
    AddInferenceOpts(app, args.inference);
    // Not ->required(): when --from-frame-jsons is set we read per-frame
    // JSONs directly and never touch images. Conditional validation in CmdRun.
    app->add_option("--input-dir", args.input_dir, "frame PNGs");
    app->add_option("--output-dir", args.output_dir, "per-board JSON outputs")->required();
    AddFramesFilterOpts(app, args.frames);
    AddSplitsFilterOpts(app, args.splits);
    AddStitchOpts(app, args.stitch);

    auto* dump = app->add_option("--dump-per-frame", args.dump_per_frame_dir,
                                 "also write one {stem}.json per frame to DIR");
    auto* from =
        app->add_option("--from-frame-jsons", args.from_frame_jsons_dir,
                        "skip inference; read per-frame JSONs from DIR (debug / re-stitch)");
    // Reading cached frames AND re-dumping them is meaningless; force a choice.
    dump->excludes(from);

    app->add_flag("--force", args.force, "overwrite existing per-board outputs");
}

void AddEvalOptions(CLI::App* app, EvalArgs& args) {
    app->add_option("--pred-dir", args.pred_dir,
                    "per-board prediction JSONs (output of `knots run`)")
        ->required();
    app->add_option("--gt-dir", args.gt_dir,
                    "per-board GT JSONs (rebuilt from --labels-dir if missing)")
        ->required();
    AddBoardsFilterOpts(app, args.boards);
    app->add_option("--out", args.out_json,
                    "JSON output path (default out/analysis/eval_boards.json)");
    app->add_option("--match-iou", args.match_iou, "bbox IoU threshold for matching")
        ->capture_default_str();
    app->add_flag("--no-write", args.no_write, "skip JSON output; print to stdout only");

    // Optional GT rebuild. Pass both to enable; eval will stitch any
    // missing per-board GT JSONs from labels before comparing.
    app->add_option("--labels-dir", args.labels_dir,
                    "YOLO bbox labels; if set with --images-dir, GT is "
                    "stitched into --gt-dir for any boards missing it");
    app->add_option("--images-dir", args.images_dir,
                    "frame PNGs (for frame dimensions; only used with --labels-dir)");
    AddStitchOpts(app, args.stitch);
}

}  // namespace knots::cli
