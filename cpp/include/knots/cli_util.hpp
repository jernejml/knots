#pragma once

// Shared CLI helpers for the `knots` subcommand dispatchers. Each helper here
// previously had 2-5 copies scattered across the individual cmd_*.cpp files.

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace knots::cli {

// Validate that argv[i+1] exists. Prints "missing value for FLAG" to stderr
// and returns false otherwise. Caller is expected to surface a usage message.
bool RequireNext(const std::string& flag, int i, int argc);

// CSV-line splitter that respects double-quoted fields. Trailing empty
// fields are preserved; the returned vector always has at least one element.
std::vector<std::string> SplitCsvLine(const std::string& line);

// Read --splits-csv (analysis/splits.csv shape: header row must include
// 'board' and 'split' columns). Returns the set of board IDs whose 'split'
// equals `want`. Throws std::runtime_error on IO failure or missing columns.
std::unordered_set<int> LoadBoardsInSplit(const std::filesystem::path& csv,
                                          const std::string& want);

// --boards LIST: "1,2,3" → {1,2,3}. Silent on malformed tokens.
std::unordered_set<int> ParseBoardsList(const std::string& csv);

// --boards-file FILE: one ID per line; '#' comments and blank lines allowed.
// Throws std::runtime_error if the file cannot be opened.
std::unordered_set<int> ParseBoardsFile(const std::filesystem::path& p);

// --frames LIST: "0_5,100_3" → {"0_5","100_3"}. Trims surrounding whitespace;
// no stem validation here (caller decides what's acceptable).
std::vector<std::string> ParseFramesList(const std::string& csv);

// --frames-file FILE: one stem per line; '#' comments and blank lines allowed.
// Throws std::runtime_error if the file cannot be opened.
std::vector<std::string> ParseFramesFile(const std::filesystem::path& p);

// "0_5" → (board=0, frame_idx=5). Returns false on malformed input.
bool ParseFrameStem(const std::string& stem, int& board, int& frame_idx);

// Compose the board-id filter from the three flags every subcommand exposes.
// Semantics:
//   1. --boards LIST and --boards-file PATH are alternatives; if both are set
//      the file wins (last-write semantics, matching argparse-style flags).
//   2. --splits-csv + --split intersect with whatever the previous step set.
//      If no --boards/--boards-file was given, the splits set is the filter.
//   3. Empty result means "no filter" (all boards admitted) — only when no
//      flag was passed; an explicit but empty intersection is still empty.
//
// Used by `knots eval` (Mode B) and `knots gt-stitch` so the same flags mean
// the same thing in both.
std::unordered_set<int> BuildBoardsFilter(const std::string& boards_csv,
                                          const std::filesystem::path& boards_file,
                                          const std::filesystem::path& splits_csv,
                                          const std::string& split);

}  // namespace knots::cli
