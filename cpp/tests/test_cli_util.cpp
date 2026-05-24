// Unit tests for knots/cli_util.hpp helpers.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "knots/cli_util.hpp"

namespace fs = std::filesystem;

// -- ParseFrameStem ----------------------------------------------------------

TEST(ParseFrameStem, ValidStem) {
    int board = -1, frame = -1;
    ASSERT_TRUE(knots::cli::ParseFrameStem("12_345", board, frame));
    EXPECT_EQ(board, 12);
    EXPECT_EQ(frame, 345);
}

TEST(ParseFrameStem, ZeroValues) {
    int board = -1, frame = -1;
    ASSERT_TRUE(knots::cli::ParseFrameStem("0_0", board, frame));
    EXPECT_EQ(board, 0);
    EXPECT_EQ(frame, 0);
}

TEST(ParseFrameStem, RejectsMalformed) {
    int board = -1, frame = -1;
    EXPECT_FALSE(knots::cli::ParseFrameStem("abc", board, frame));
    EXPECT_FALSE(knots::cli::ParseFrameStem("5_a", board, frame));
    EXPECT_FALSE(knots::cli::ParseFrameStem("_5", board, frame));
    EXPECT_FALSE(knots::cli::ParseFrameStem("5_5_5", board, frame));
    EXPECT_FALSE(knots::cli::ParseFrameStem("", board, frame));
}

// -- LoadBoardsInSplit (JSON reader used by cmd_run for --split filtering) ---

namespace {

// Write a minimal partitions.json to a temp path. Returns the path; caller removes.
fs::path WriteTempPartitionsJson(const std::string& content) {
    fs::path p = fs::temp_directory_path() / ("knots_partitions_" + std::to_string(::getpid()) +
                                              "_" + std::to_string(rand()) + ".json");
    std::ofstream(p) << content;
    return p;
}

const std::string kThreeBoardPartitions =
    R"({"train": [1], "val": [], "test": [2, 3]})";

}  // namespace

TEST(LoadBoardsInSplit, ReturnsBoardsForNamedSplit) {
    auto json = WriteTempPartitionsJson(kThreeBoardPartitions);
    auto s = knots::cli::LoadBoardsInSplit(json, "test");
    EXPECT_EQ(s.size(), 2u);
    EXPECT_TRUE(s.count(2));
    EXPECT_TRUE(s.count(3));
    fs::remove(json);
}

TEST(LoadBoardsInSplit, EmptyArrayReturnsEmptySet) {
    auto json = WriteTempPartitionsJson(kThreeBoardPartitions);
    auto s = knots::cli::LoadBoardsInSplit(json, "val");
    EXPECT_TRUE(s.empty());
    fs::remove(json);
}

TEST(LoadBoardsInSplit, UnknownSplitReturnsEmptySet) {
    // Unknown split name is treated as 'no boards' (caller decides what to do).
    auto json = WriteTempPartitionsJson(kThreeBoardPartitions);
    auto s = knots::cli::LoadBoardsInSplit(json, "nonexistent");
    EXPECT_TRUE(s.empty());
    fs::remove(json);
}

TEST(LoadBoardsInSplit, MissingFileThrows) {
    EXPECT_THROW(
        knots::cli::LoadBoardsInSplit("/does/not/exist.json", "test"),
        std::runtime_error);
}
