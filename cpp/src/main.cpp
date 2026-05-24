// knots — C++ pipeline for wood-knot polygon extraction.
//
// One CLI11 app, two subcommands. Per-subcommand option sets live in
// cli_options.cpp; the actual work lives in cmd_run.cpp / cmd_eval.cpp
// behind CmdXxx(args) entry points.
//
//   knots run    per-frame infer + per-board stitch.
//                Debug flags: --dump-per-frame DIR (also write per-frame
//                inference JSONs), --from-frame-jsons DIR (skip inference;
//                re-stitch from previously-dumped JSONs).
//   knots eval   compare per-board predictions to GT (greedy bbox-IoU
//                match + per-pair mask IoU). Rebuilds missing GT from
//                --labels-dir / --images-dir when both are passed.

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
    auto* run =
        app.add_subcommand("run", "per-frame infer + per-board stitch (with optional cache flags)");
    knots::cli::AddRunOptions(run, run_args);

    knots::EvalArgs eval_args;
    auto* eval = app.add_subcommand("eval", "compare per-board predictions to GT polygons");
    knots::cli::AddEvalOptions(eval, eval_args);

    // Only the parsed subcommand's callback runs; `rc` holds its return code.
    int rc = 0;
    run->callback([&] { rc = knots::CmdRun(run_args); });
    eval->callback([&] { rc = knots::CmdEval(eval_args); });

    CLI11_PARSE(app, argc, argv);
    return rc;
}
