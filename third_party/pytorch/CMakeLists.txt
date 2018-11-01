# https://github.com/pytorch/pytorch/commit/70db53661b5ef80957030eeef727264ce1a67781
set(PYTORCH_COMMIT 70db536)
set(PYTORCH_ROOT ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND PYTORCH_FLAGS
        # Build libcaffe2.so
        -DBUILD_SHARED_LIBS=ON
        # Build libtorch.so
        -DBUILD_TORCH=ON
        # Enable multiprocessing features
        -DUSE_OPENMP=ON
        # Enable Intel's performance library
        # TODO: https://github.com/pytorch/pytorch/pull/13303#issuecomment-435017808
        -DUSE_MKLDNN=ON
        # Build Python binaries
        -DBUILD_PYTHON=ON
        # Disable distributed training
        -DUSE_DISTRIBUTED=OFF
        # Disable mobile-optimized library
        -DUSE_QNNPACK=OFF
        # Disable OpenCV
        -DUSE_OPENCV=OFF
        # Don't build tests
        -DBUILD_TEST=OFF
        # Don't build documentation
        -DBUILD_DOCS=OFF
        # See: https://github.com/intel/ideep/issues/23
        -DCMAKE_CXX_FLAGS=-Wno-format-truncation
        )

# Only 17 / 28 submodules are currently used by Caffe2 and Torch
# (No need to log, benchmark, compile for IOS, ...)
list(APPEND PYTORCH_SUBMODULES
        third_party/cpuinfo
        third_party/cub
        third_party/eigen
        third_party/FP16
        third_party/FXdiv
        third_party/gloo
        third_party/NNPACK
        third_party/ideep
        third_party/onnx
        third_party/protobuf
        third_party/psimd
        third_party/pthreadpool
        third_party/pybind11
        third_party/python-enum
        third_party/python-peachpy
        third_party/python-six
        third_party/sleef
        )

ExternalProject_Add(pytorch
        PREFIX ${PYTORCH_ROOT}
        GIT_REPOSITORY "https://github.com/pytorch/pytorch.git"
        GIT_SUBMODULES ${PYTORCH_SUBMODULES}
        GIT_TAG ${PYTORCH_COMMIT}
        CMAKE_ARGS += ${PYTORCH_FLAGS}
        )

add_library(libtorch SHARED IMPORTED GLOBAL)

set(TORCH_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(TORCH_LIBRARY "${TORCH_INSTALL_PREFIX}/lib/libtorch.so")
set(TORCH_INCLUDE_DIRS
        ${TORCH_INSTALL_PREFIX}/include
        ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
        )
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
set_target_properties(libtorch PROPERTIES
        IMPORTED_LOCATION "${TORCH_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
        INTERFACE_COMPILE_OPTIONS "${TORCH_CXX_FLAGS}"
        INTERFACE_COMPILE_FEATURES "cxx_std_11"
        )
add_dependencies(libtorch pytorch)

add_library(libcaffe2 SHARED IMPORTED GLOBAL)

set(CAFFE2_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CAFFE2_LIBRARY "${CAFFE2_INSTALL_PREFIX}/lib/libcaffe2.so")
set(CAFFE2_INCLUDE_DIRS
        ${CAFFE2_INSTALL_PREFIX}/include
        )
set(CAFFE2_LINK_LIBRARIES "protobuf::libprotobuf;c10;Threads::Threads;caffe2::mkl;glog::glog")
set(CAFFE2_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
set_target_properties(libcaffe2 PROPERTIES
        IMPORTED_LOCATION "${CAFFE2_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CAFFE2_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${CAFFE2_LINK_LIBRARIES}"
        INTERFACE_COMPILE_OPTIONS "${CAFFE2_CXX_FLAGS}"
        INTERFACE_COMPILE_FEATURES "cxx_std_11"
        )
add_dependencies(libcaffe2 pytorch)

# Finally, set the Caffe2 and Torch properties in the parent scope.
set(CAFFE2_INCLUDE_DIRS ${CAFFE2_INCLUDE_DIRS} PARENT_SCOPE)
set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS} PARENT_SCOPE)
set(TORCH_LIBRARIES libtorch libcaffe2 PARENT_SCOPE)
