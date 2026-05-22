// knots — C++ pipeline for wood-knot polygon extraction.
//
// One CLI11 app, five subcommands. Per-subcommand option sets live in
// cli_options.cpp; the actual work lives in run.cpp / infer.cpp / stitch.cpp
// / gt_stitch.cpp / eval.cpp behind CmdXxx(args) entry points.
//
//   knots run         per-frame infer + per-board stitch (one-shot)
//   knots infer       per-frame YOLO11-seg inference, one JSON per frame
//   knots stitch      per-board raster-union from per-frame JSONs
//   knots gt-stitch   per-board raster-union from per-frame YOLO GT bboxes
//   knots eval        compare per-board predictions to GT (Mode A or B)

#include <CLI/CLI.hpp>

#include "knots/cli_options.hpp"
#include "knots/commands.hpp"

int main(int argc, char** argv) {
    CLI::App app{"knots — wood-knot polygon extraction"};
    app.require_subcommand(1);
    app.set_help_all_flag("--help-all", "expand help for all subcommands");

    // One Args struct per subcommand; the matching callback fires after
    // parse and forks into the corresponding CmdXxx implementation.
    knots::RunArgs run_args;
    auto* run = app.add_subcommand(
        "run", "one-shot: per-frame infer + per-board stitch");
    knots::cli::AddRunOptions(run, run_args);

    knots::InferArgs infer_args;
    auto* infer = app.add_subcommand("infer", "per-frame YOLO11-seg inference");
    knots::cli::AddInferOptions(infer, infer_args);

    knots::StitchArgs stitch_args;
    auto* stitch = app.add_subcommand(
        "stitch", "per-board raster-union of per-frame inference JSONs");
    knots::cli::AddStitchOptions(stitch, stitch_args);

    knots::GtStitchArgs gt_args;
    auto* gt = app.add_subcommand(
        "gt-stitch", "per-board raster-union of per-frame GT bboxes");
    knots::cli::AddGtStitchOptions(gt, gt_args);

    knots::EvalArgs eval_args;
    auto* eval = app.add_subcommand(
        "eval", "compare per-board predictions to GT polygons (Mode A or B)");
    knots::cli::AddEvalOptions(eval, eval_args);

    // Only the parsed subcommand's callback runs; `rc` holds its return code.
    int rc = 0;
    run->callback([&] { rc = knots::CmdRun(run_args); });
    infer->callback([&] { rc = knots::CmdInfer(infer_args); });
    stitch->callback([&] { rc = knots::CmdStitch(stitch_args); });
    gt->callback([&] { rc = knots::CmdGtStitch(gt_args); });
    eval->callback([&] { rc = knots::CmdEval(eval_args); });

    CLI11_PARSE(app, argc, argv);
    return rc;
}
