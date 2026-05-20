// knots — C++ pipeline for wood-knot polygon extraction.
//
// Subcommand dispatch. Two stages:
//   knots infer    per-frame YOLO11-seg inference, writes one JSON per frame.
//   knots stitch   reads per-frame JSONs, projects to board coordinates,
//                  raster-unions overlapping polygons, writes one JSON per
//                  board (the final per-board polygon list).

#include <iostream>
#include <string>

#include "knots/commands.hpp"

namespace {

void PrintTopUsage(const char* prog) {
    std::cerr <<
        "usage: " << prog << " <subcommand> [options]\n"
        "  infer       per-frame YOLO11-seg inference\n"
        "  stitch      per-board raster-union of per-frame inference polygons\n"
        "  gt-stitch   per-board raster-union of per-frame GT bboxes\n"
        "Run `" << prog << " <subcommand> --help` for subcommand-specific options.\n";
}

}  // namespace


int main(int argc, char** argv) {
    if (argc < 2) {
        PrintTopUsage(argv[0]);
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "infer")     return knots::CmdInfer(argc - 1, argv + 1);
    if (cmd == "stitch")    return knots::CmdStitch(argc - 1, argv + 1);
    if (cmd == "gt-stitch") return knots::CmdGtStitch(argc - 1, argv + 1);
    if (cmd == "-h" || cmd == "--help") {
        PrintTopUsage(argv[0]);
        return 0;
    }
    std::cerr << "unknown subcommand: " << cmd << "\n";
    PrintTopUsage(argv[0]);
    return 2;
}
