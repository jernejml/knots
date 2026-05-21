# C++ knots program: per-frame YOLOv11-seg inference + cross-frame dedup.
#
# Step 1 (current): smoke-test ONNX Runtime by loading best.onnx and printing
# input/output names + shapes + types. Validates the build setup before any
# real pipeline logic is added.
#
# Build:
#   cd cpp && mkdir build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release \
#         -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
#   cmake --build . -j
#
# Run:
#   ./knots ../../models/best.onnx

cmake_minimum_required(VERSION 3.20)
project(knots LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(nlohmann_json REQUIRED)
find_package(OnnxRuntime REQUIRED)

add_executable(knots
    src/main.cpp
    src/infer.cpp
    src/stitch.cpp
    src/gt_stitch.cpp
    src/stitching.cpp
    src/inference.cpp
    src/eval.cpp
    src/run.cpp
    src/cli_util.cpp
    src/pipeline.cpp
    src/geometry.cpp
)

target_include_directories(knots PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(knots PRIVATE
    OnnxRuntime::OnnxRuntime
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

target_compile_options(knots PRIVATE
    -Wall -Wextra -Wpedantic
)

# -- Tests -------------------------------------------------------------------
# Linked against the geometry + cli_util source files directly (no ORT, no
# ultralytics, no I/O), so the test binary builds in seconds and stays
# isolated from the heavy inference dependencies. Enabled by default; disable
# with -DBUILD_TESTS=OFF when iterating on the main binary.

option(BUILD_TESTS "Build the unit tests" ON)

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()

    add_executable(knots_tests
        tests/test_geometry.cpp
        tests/test_cli_util.cpp
        src/geometry.cpp
        src/cli_util.cpp
    )
    target_include_directories(knots_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(knots_tests PRIVATE
        GTest::gtest_main
        ${OpenCV_LIBS}
    )
    target_compile_options(knots_tests PRIVATE
        -Wall -Wextra -Wpedantic
    )

    include(GoogleTest)
    gtest_discover_tests(knots_tests)
endif()
