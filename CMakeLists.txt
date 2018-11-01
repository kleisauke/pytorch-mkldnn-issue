cmake_minimum_required(VERSION 3.11)
project(pytorch_mkldnn_issue)

set(CMAKE_CXX_STANDARD 11)

include(ExternalProject)

# Find Torch (optional)
find_package(Torch "1.0.0" QUIET)

option(BUILD_TORCH "Build PyTorch's C++ API" OFF)

# Build PyTorch's C++ API, if necessary
if (BUILD_TORCH)
    add_subdirectory(third_party/pytorch)
endif()

add_executable(cartpole cartpole.cpp)
target_compile_features(cartpole PUBLIC cxx_range_for)
target_link_libraries(cartpole ${TORCH_LIBRARIES})
