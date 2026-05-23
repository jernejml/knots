#pragma once

// File-IO helpers used by the CLI option callbacks in cli_options.cpp and by
// the subcommand pipelines. Intentionally CLI11-free so the unit-test binary
// can link against this TU without pulling in CLI11.

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace knots::cli {

// Read --partitions-json (analysis/partitions.json shape: top-level object
// with "train" / "val" / "test" arrays of board IDs). Returns the set of
// board IDs in the `want` split. Throws std::runtime_error on IO failure
// or malformed JSON.
std::unordered_set<int> LoadBoardsInSplit(const std::filesystem::path& path,
                                          const std::string& want);

// --boards-file FILE: one ID per line; '#' comments and blank lines allowed.
// Throws std::runtime_error if the file cannot be opened.
std::unordered_set<int> ParseBoardsFile(const std::filesystem::path& p);

// --frames-file FILE: one stem per line; '#' comments and blank lines allowed.
// Throws std::runtime_error if the file cannot be opened.
std::vector<std::string> ParseFramesFile(const std::filesystem::path& p);

// "0_5" → (board=0, frame_idx=5). Returns false on malformed input.
bool ParseFrameStem(const std::string& stem, int& board, int& frame_idx);

// Build the combined explicit-stems list from --frames + --frames-file. The
// file (if non-empty) is appended to the CLI list; result is the order
// `knots run`/`knots infer` pass to CollectFramesByBoard.
std::vector<std::string> CollectExplicitStems(const std::vector<std::string>& frames,
                                              const std::filesystem::path& frames_file);

// Compose the board-id filter from the three flag groups every subcommand
// exposes. Semantics:
//   1. --boards LIST and --boards-file PATH are alternatives; if both are set
//      the file wins (last-write semantics).
//   2. --partitions-json + --split intersect with whatever the previous step
//      set. If no --boards/--boards-file was given, the split set is the
//      filter.
//   3. Empty result means "no filter" (all boards admitted) — only when no
//      flag was passed; an explicit but empty intersection is still empty.
//
// Used by `knots eval` (Mode B) and `knots gt-stitch` so the same flags mean
// the same thing in both.
std::unordered_set<int> BuildBoardsFilter(const std::vector<int>& boards,
                                          const std::filesystem::path& boards_file,
                                          const std::filesystem::path& partitions_json,
                                          const std::string& split);

}  // namespace knots::cli
