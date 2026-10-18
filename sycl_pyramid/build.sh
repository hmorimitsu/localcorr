#!/usr/bin/env bash
set -euo pipefail

# Build the localcorr SYCL kernel and its OpenVINO custom-operation
# extension into localcorr/sycl_pyramid/build/.
#
# Requirements: oneAPI (icpx with -fsycl) and the OpenVINO runtime
# development files. By default the headers/libs of the OpenVINO pip
# package inside the current Python environment are used; point
# OV_INSTALL_DIR at an OpenVINO installation to override.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"
build_jobs="${BUILD_JOBS:-$(nproc)}"

python_bin="${PYTHON_BIN:-python}"
if ! command -v "$python_bin" > /dev/null 2>&1; then
    python_bin=python3
fi

ov_root="${OV_INSTALL_DIR:-$("$python_bin" -c 'import openvino, os; print(os.path.dirname(openvino.__file__))')}"
ov_include="${ov_root}/include"
ov_libs="${ov_root}/libs"

oneapi_root="${ONEAPI_ROOT:-/opt/intel/oneapi}"
# Skip the source when setvars.sh has already been run by the calling
# shell: it refuses to re-execute and returns 3, which would abort this
# script under set -e. Its return code is not fatal in general either;
# a missing icpx fails loudly at the compile step below.
if [ "${SETVARS_COMPLETED:-0}" != "1" ] && [ -f "${oneapi_root}/setvars.sh" ]; then
    # setvars.sh references unbound variables; relax -u while sourcing.
    set +u
    # shellcheck disable=SC1091
    source "${oneapi_root}/setvars.sh" > /dev/null || true
    set -u
fi

mkdir -p "${build_dir}"

common_flags=(-std=c++20 -O2 -fPIC -fsycl -fsycl-device-code-split=per_kernel
              -I"${script_dir}")

# The oneAPI compiler runtime (libsycl, libsvml, ...) is linked with an
# rpath so the libraries load inside processes that have not sourced
# setvars.sh (e.g. the Python interpreter importing openvino).
compiler_lib="$(dirname "$(command -v icpx)")/../lib"
common_flags+=(-Wl,-rpath,"${compiler_lib}")

# The device code is split per kernel so that the runtime JIT only
# compiles the kernel instantiations that are actually launched.
icpx "${common_flags[@]}" -shared \
    "${script_dir}/localcorr_cuda_pyramid_kernel.dp.cpp" \
    -o "${build_dir}/liblocalcorr_sycl_pyramid.so"

# The OpenVINO extension links libopenvino from the detected installation;
# the rpath keeps it loadable outside the Python process too. Pip packages
# only ship the versioned soname (libopenvino.so.26xx), so link it by path.
ov_lib_path="$(ls "${ov_libs}"/libopenvino.so.* 2>/dev/null | head -n 1)"
if [ -z "${ov_lib_path}" ]; then
    ov_lib_path="$(ls "${ov_libs}"/libopenvino.so 2>/dev/null | head -n 1)"
fi
icpx "${common_flags[@]}" -shared \
    -I"${ov_include}" \
    "${script_dir}/localcorr_sycl_ov_extension.cpp" \
    -L"${build_dir}" -llocalcorr_sycl_pyramid \
    "${ov_lib_path}" \
    -Wl,-rpath,"${build_dir}" \
    -Wl,-rpath,"${ov_libs}" \
    -o "${build_dir}/liblocalcorr_sycl_pyramid_ov.so"

if [ "${SKIP_TEST:-0}" != "1" ]; then
    icpx "${common_flags[@]}" -I"${script_dir}" \
        "${script_dir}/test_kernel.cpp" \
        -L"${build_dir}" -llocalcorr_sycl_pyramid \
        -Wl,-rpath,"${build_dir}" \
        -o "${build_dir}/test_kernel" &
    wait
fi

echo "Kernel library:      ${build_dir}/liblocalcorr_sycl_pyramid.so"
echo "OpenVINO extension:  ${build_dir}/liblocalcorr_sycl_pyramid_ov.so"
if [ "${SKIP_TEST:-0}" != "1" ]; then
    echo "Correctness test:    ${build_dir}/test_kernel"
fi
