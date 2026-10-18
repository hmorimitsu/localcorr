#!/usr/bin/env bash
set -euo pipefail

# Build the localcorr_cuda_pyramid TensorRT plugin (AOT-compiled CUDA
# kernel + IPluginV3 wrapper) into localcorr/cuda_pyramid/build/.
#
# Requirements: CUDA toolkit (nvcc), CMake >= 3.18, TensorRT headers and
# libnvinfer. Set TRT_ROOT or TENSORRT_ROOT if TensorRT is not in a
# default system location.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"
build_jobs="${BUILD_JOBS:-$(nproc)}"

cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --config Release -j"${build_jobs}"

echo "Plugin library: ${build_dir}/liblocalcorr_cuda_pyramid_plugin.so"
