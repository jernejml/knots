#include "knots/cli_util.hpp"

#include <algorithm>
#include <fstream>
#include <regex>

namespace knots::cli {

namespace {

// CSV-line splitter that respects double-quoted fields. Trailing empty
// fields are preserved. File-local because the only caller is the
// splits.csv reader below; no caller outside this TU needs the helper.
std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    fields.push_back(cur);
    return fields;
}

}  // namespace

std::unordered_set<int> LoadBoardsInSplit(const std::filesystem::path& csv,
                                          const std::string& want) {
    std::ifstream f(csv);
    if (!f) throw std::runtime_error("cannot open " + csv.string());
    std::string line;
    if (!std::getline(f, line)) throw std::runtime_error("empty splits CSV");
    auto header = SplitCsvLine(line);
    int board_col = -1, split_col = -1;
    for (size_t c = 0; c < header.size(); ++c) {
        if (header[c] == "board") board_col = static_cast<int>(c);
        if (header[c] == "split") split_col = static_cast<int>(c);
    }
    if (board_col < 0 || split_col < 0) {
        throw std::runtime_error("splits CSV missing 'board' or 'split' column");
    }
    std::unordered_set<int> boards;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto cols = SplitCsvLine(line);
        if (static_cast<int>(cols.size()) <= std::max(board_col, split_col)) continue;
        if (cols[split_col] != want) continue;
        try {
            boards.insert(std::stoi(cols[board_col]));
        } catch (...) {
            // skip malformed
        }
    }
    return boards;
}

std::unordered_set<int> ParseBoardsFile(const std::filesystem::path& p) {
    std::unordered_set<int> out;
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::string line;
    while (std::getline(f, line)) {
        auto a = line.find_first_not_of(" \t");
        if (a == std::string::npos) continue;
        if (line[a] == '#') continue;
        try {
            out.insert(std::stoi(line.substr(a)));
        } catch (...) {
        }
    }
    return out;
}

std::vector<std::string> ParseFramesFile(const std::filesystem::path& p) {
    std::vector<std::string> out;
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::string line;
    while (std::getline(f, line)) {
        auto a = line.find_first_not_of(" \t");
        if (a == std::string::npos) continue;
        if (line[a] == '#') continue;
        auto b = line.find_last_not_of(" \t");
        out.push_back(line.substr(a, b - a + 1));
    }
    return out;
}

std::vector<std::string> CollectExplicitStems(const std::vector<std::string>& frames,
                                              const std::filesystem::path& frames_file) {
    std::vector<std::string> out = frames;
    if (!frames_file.empty()) {
        auto from_file = ParseFramesFile(frames_file);
        out.insert(out.end(), from_file.begin(), from_file.end());
    }
    return out;
}

bool ParseFrameStem(const std::string& stem, int& board, int& frame_idx) {
    static const std::regex re(R"(^(\d+)_(\d+)$)");
    std::smatch m;
    if (!std::regex_match(stem, m, re)) return false;
    try {
        board = std::stoi(m[1]);
        frame_idx = std::stoi(m[2]);
        return true;
    } catch (...) {
        return false;
    }
}

std::unordered_set<int> BuildBoardsFilter(const std::vector<int>& boards,
                                          const std::filesystem::path& boards_file,
                                          const std::filesystem::path& splits_csv,
                                          const std::string& split) {
    std::unordered_set<int> filter(boards.begin(), boards.end());
    if (!boards_file.empty()) filter = ParseBoardsFile(boards_file);
    if (!splits_csv.empty()) {
        auto split_boards = LoadBoardsInSplit(splits_csv, split);
        if (filter.empty()) {
            filter = std::move(split_boards);
        } else {
            std::unordered_set<int> isect;
            for (int b : filter) {
                if (split_boards.count(b)) isect.insert(b);
            }
            filter = std::move(isect);
        }
    }
    return filter;
}

}  // namespace knots::cli
