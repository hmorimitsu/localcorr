#!/usr/bin/env bash
set -euo pipefail

# Build the localcorr SYCL kernel as a PyTorch XPU extension into
# localcorr/xpu/build/ (importable as localcorr_sycl_xpu).
#
# Requirements: a PyTorch XPU build in the active Python environment and
# oneAPI (icpx). Compile the extension with the same interpreter that
# will import it (PYTHON_BIN, default: python).

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"

python_bin="${PYTHON_BIN:-python}"

# torch must be an XPU build for the extension to make sense.
"$python_bin" - <<'EOF'
import sys
import torch
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    print("ERROR: torch.xpu is not available; install a PyTorch XPU build "
          "(pip install torch --index-url https://download.pytorch.org/whl/xpu).",
          file=sys.stderr)
    sys.exit(1)
EOF

oneapi_root="${ONEAPI_ROOT:-/opt/intel/oneapi}"
# Skip the source when setvars.sh has already been run by the calling
# shell: it refuses to re-execute and returns 3, which would abort this
# script under set -e. Its return code is not fatal in general either;
# the icpx check below reports a broken oneAPI setup.
if [ "${SETVARS_COMPLETED:-0}" != "1" ] && [ -f "${oneapi_root}/setvars.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    source "${oneapi_root}/setvars.sh" > /dev/null || true
    set -u
fi
if ! command -v icpx > /dev/null 2>&1; then
    echo "ERROR: icpx not found; install oneAPI or source setvars.sh." >&2
    exit 1
fi

mkdir -p "${build_dir}"
cd "${script_dir}"
CC=icpx CXX=icpx "$python_bin" setup.py build_ext --build-temp "${build_dir}/tmp" \
    --build-lib "${build_dir}"

echo "Extension: ${build_dir}/localcorr_sycl_xpu$( \
    "$python_bin" -c 'import sys; print(f".cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so")')"
