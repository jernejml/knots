#include "knots/cli_util.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>

namespace knots::cli {

std::unordered_set<int> LoadBoardsInSplit(const std::filesystem::path& path,
                                          const std::string& want) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    nlohmann::json j;
    try {
        f >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("malformed partitions JSON " + path.string() + ": " + e.what());
    }
    if (!j.is_object()) {
        throw std::runtime_error("partitions JSON must be an object with split keys: " +
                                 path.string());
    }
    std::unordered_set<int> boards;
    if (!j.contains(want)) return boards;  // unknown split name → empty set
    const auto& arr = j[want];
    if (!arr.is_array()) {
        throw std::runtime_error("partitions JSON key '" + want + "' must be an array of ints");
    }
    for (const auto& v : arr) {
        if (v.is_number_integer()) boards.insert(v.get<int>());
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

}  // namespace knots::cli
