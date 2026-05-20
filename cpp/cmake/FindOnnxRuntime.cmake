# Locate ONNX Runtime C/C++ API.
#
# Honors:
#   ONNXRUNTIME_ROOT  (CMake cache var or environment var)
#
# Falls back to standard system paths if the variable is not set. On Linux
# the layout is expected to be:
#   ${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h
#   ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
#
# On success defines the imported target OnnxRuntime::OnnxRuntime that
# pulls in headers and the library.

find_path(OnnxRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ONNXRUNTIME_ROOT}/include
        $ENV{ONNXRUNTIME_ROOT}/include
    PATH_SUFFIXES onnxruntime
)

find_library(OnnxRuntime_LIBRARY
    NAMES onnxruntime
    HINTS
        ${ONNXRUNTIME_ROOT}/lib
        $ENV{ONNXRUNTIME_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
    REQUIRED_VARS OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY
)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
    set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
        IMPORTED_LOCATION ${OnnxRuntime_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${OnnxRuntime_INCLUDE_DIR}
    )
endif()

mark_as_advanced(OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY)
