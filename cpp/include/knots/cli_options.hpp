#pragma once

// CLI11 option-group registration. Each AddXxxOpts function registers a
// fixed set of flags on a CLI::App* (typically a subcommand) and binds them
// to the matching field in `args`. main.cpp composes these per subcommand
// so identical flags appear with identical semantics everywhere.

#include <CLI/CLI.hpp>

#include "knots/commands.hpp"

namespace knots::cli {

// --model (required) / --conf / --cpu. Registered as required so CLI11
// prints a "missing required option" error if --model is omitted.
void AddInferenceOpts(CLI::App* app, InferenceOpts& opts);

// --stride-px / --simplify-eps with kStridePxDefault / 1.0 defaults.
void AddStitchOpts(CLI::App* app, StitchOpts& opts);

// --boards LIST / --boards-file PATH.
void AddBoardsFilterOpts(CLI::App* app, BoardsFilter& f);

// --splits-csv PATH / --split {train,val,test}.
// `splits_csv_needs_split` (default true) wires CLI11 to fail if --splits-csv
// is given without --split. Kept togglable because some call sites validate
// this differently.
void AddSplitsFilterOpts(CLI::App* app, SplitsFilter& f);

// --frames LIST / --frames-file PATH.
void AddFramesFilterOpts(CLI::App* app, FramesFilter& f);

// Per-subcommand assemblies: one call sets up every flag the subcommand
// accepts, including required positional/option flags and defaults.
void AddRunOptions(CLI::App* app, RunArgs& args);
void AddInferOptions(CLI::App* app, InferArgs& args);
void AddStitchOptions(CLI::App* app, StitchArgs& args);
void AddGtStitchOptions(CLI::App* app, GtStitchArgs& args);
void AddEvalOptions(CLI::App* app, EvalArgs& args);

}  // namespace knots::cli
