cmake_minimum_required(VERSION 3.18)

project(localcorr_cuda_pyramid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

find_path(
    TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        ENV TRT_ROOT
        ENV TENSORRT_ROOT
    PATH_SUFFIXES
        include
        include/x86_64-linux-gnu
        include/aarch64-linux-gnu
    PATHS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/include/aarch64-linux-gnu
)

find_library(
    TENSORRT_LIB
    NAMES nvinfer
    HINTS
        ENV TRT_ROOT
        ENV TENSORRT_ROOT
    PATH_SUFFIXES
        lib
        lib64
        lib/x86_64-linux-gnu
        lib/aarch64-linux-gnu
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find NvInfer.h. Set TRT_ROOT or TENSORRT_ROOT to your TensorRT installation.")
endif()

if(NOT TENSORRT_LIB)
    message(FATAL_ERROR "Could not find libnvinfer.so. Set TRT_ROOT or TENSORRT_ROOT to your TensorRT installation.")
endif()

add_library(localcorr_cuda_pyramid_plugin SHARED
    localcorr_cuda_pyramid_plugin.cpp
    localcorr_cuda_pyramid_kernel.cu
)

target_include_directories(localcorr_cuda_pyramid_plugin PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(localcorr_cuda_pyramid_plugin PRIVATE
    ${TENSORRT_LIB}
    CUDA::cudart
)

set_target_properties(localcorr_cuda_pyramid_plugin PROPERTIES
    OUTPUT_NAME localcorr_cuda_pyramid_plugin
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES_OR_NATIVE)
    # Compile for the architecture of the GPU present on this machine, so
    # the fatbin always contains a native image (a plain nvcc default such
    # as sm_52/sm_75 would require driver JIT or fail outright).
    set(_detected_arch "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        # "native" queries the GPUs through the CUDA runtime; works on
        # x86 workstations and on Jetson modules alike.
        set(_detected_arch "native")
    else()
        # Discrete GPUs: query the compute capability with nvidia-smi.
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _GPU_ARCH_LIST
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _NVIDIA_SMI_RESULT
        )
        if(_NVIDIA_SMI_RESULT EQUAL 0)
            string(REPLACE " " "" _GPU_ARCH_LIST "${_GPU_ARCH_LIST}")
            string(REPLACE "\n" ";" _GPU_ARCH_LIST "${_GPU_ARCH_LIST}")
            list(GET _GPU_ARCH_LIST 0 _detected_arch)
            string(REPLACE "." "" _detected_arch "${_detected_arch}")
        elseif(EXISTS "/proc/device-tree/model")
            # Jetson modules: nvidia-smi does not exist there; map the SoC
            # family from the device tree model string instead.
            file(READ "/proc/device-tree/model" _JETSON_MODEL)
            string(REGEX MATCH "Orin" _IS_ORIN "${_JETSON_MODEL}")
            string(REGEX MATCH "Xavier" _IS_XAVIER "${_JETSON_MODEL}")
            if(_IS_ORIN)
                set(_detected_arch 87)
            elseif(_IS_XAVIER)
                set(_detected_arch 72)
            endif()
        endif()
        if(_detected_arch STREQUAL "")
            # Detection impossible: embed PTX only for a range of common
            # architectures and let the driver JIT-compile at load time.
            set(_detected_arch "70-virtual;72-virtual;75-virtual;80-virtual;86-virtual;87-virtual;89-virtual")
        endif()
    endif()
    set(CMAKE_CUDA_ARCHITECTURES_OR_NATIVE "${_detected_arch}")
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES_OR_NATIVE}")
endif()

set_target_properties(localcorr_cuda_pyramid_plugin PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_OR_NATIVE}"
)
