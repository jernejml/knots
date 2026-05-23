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

// -- BuildBoardsFilter (the precedence-fix from item 1) ----------------------
//
// `boards` is the parsed vector from --boards (CLI11's `->delimiter(',')`
// transform handles the CSV split, so BuildBoardsFilter no longer parses
// strings). Tests pass the parsed form directly.

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

TEST(BuildBoardsFilter, AllFlagsEmptyReturnsEmpty) {
    auto f = knots::cli::BuildBoardsFilter({}, "", "", "");
    EXPECT_TRUE(f.empty());
}

TEST(BuildBoardsFilter, OnlyBoardsListIsUsedDirectly) {
    auto f = knots::cli::BuildBoardsFilter({1, 2, 3}, "", "", "");
    EXPECT_EQ(f.size(), 3u);
    EXPECT_TRUE(f.count(1));
    EXPECT_TRUE(f.count(2));
    EXPECT_TRUE(f.count(3));
}

TEST(BuildBoardsFilter, OnlyPartitionsJsonReturnsThoseBoards) {
    auto json = WriteTempPartitionsJson(kThreeBoardPartitions);
    auto f = knots::cli::BuildBoardsFilter({}, "", json, "test");
    EXPECT_EQ(f.size(), 2u);
    EXPECT_TRUE(f.count(2));
    EXPECT_TRUE(f.count(3));
    fs::remove(json);
}

TEST(BuildBoardsFilter, BoardsListIntersectsWithSplit) {
    // --boards passes 1,2,3 — but --split test only has 2,3. Intersection = {2,3}.
    auto json = WriteTempPartitionsJson(kThreeBoardPartitions);
    auto f = knots::cli::BuildBoardsFilter({1, 2, 3}, "", json, "test");
    EXPECT_EQ(f.size(), 2u);
    EXPECT_TRUE(f.count(2));
    EXPECT_TRUE(f.count(3));
    EXPECT_FALSE(f.count(1));
    fs::remove(json);
}

TEST(BuildBoardsFilter, BoardsListDisjointFromSplitEmptyIntersection) {
    // --boards selects only board 1 (train); --split test → empty intersection.
    // Crucially, this is NOT the same as "no filter": the result is an
    // explicitly empty filter that should restrict everything downstream.
    // Documenting this behaviour so any future refactor preserves it.
    auto json = WriteTempPartitionsJson(kThreeBoardPartitions);
    auto f = knots::cli::BuildBoardsFilter({1}, "", json, "test");
    EXPECT_TRUE(f.empty());
    fs::remove(json);
}
