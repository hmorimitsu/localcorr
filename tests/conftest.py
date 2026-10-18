"""pytest configuration for the localcorr unit tests.

Run from the repository root (the usual CI flow) with:

    pytest tests

(`python -m pytest` would add the checkout itself to sys.path, where the
triton/ package directory shadows the real triton package; from the
parent directory, `pytest localcorr/tests` also works.)

The tests only require CPU; CUDA / XPU / onnx / onnxruntime dependent
tests skip automatically when unavailable.
"""

import os
import sys
from pathlib import Path

import pytest

# Make `import localcorr` work when running the tests straight from the
# repository, without installing the package first.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import localcorr  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on invocation
    pytest.exit(
        f"could not import localcorr ({exc}). Run the tests from the "
        "repository root (pytest localcorr/tests) or install the package "
        "first (pip install ./localcorr).",
        returncode=4,
    )

_MISSING = object()


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Snapshot and restore every piece of module-level state tests touch."""
    import localcorr.backend as backend
    import localcorr.blocks as blocks
    import localcorr.spatial_corr as spatial_corr

    saved = {
        "backend": backend._onnx_backend,
        "triton_kernels": blocks._TRITON_KERNELS,
        "cuda_lib": blocks._CUDA_PYRAMID_LIB,
        "cuda_lib_loaded": blocks._CUDA_PYRAMID_LIB_LOADED,
        "triton_spatial_fn": spatial_corr._TRITON_SPATIAL_FN,
        "sycl_ext": spatial_corr._SYCL_XPU_EXT,
        "sycl_ext_loaded": spatial_corr._SYCL_XPU_EXT_LOADED,
        "triton_module": sys.modules.get("localcorr.triton", _MISSING),
        "onnx_backend_env": os.environ.get("LOCALCORR_ONNX_BACKEND"),
        "sycl_preload_env": os.environ.get("LOCALCORR_SYCL_PRELOAD"),
    }
    yield
    backend._onnx_backend = saved["backend"]
    blocks._TRITON_KERNELS = saved["triton_kernels"]
    blocks._CUDA_PYRAMID_LIB = saved["cuda_lib"]
    blocks._CUDA_PYRAMID_LIB_LOADED = saved["cuda_lib_loaded"]
    spatial_corr._TRITON_SPATIAL_FN = saved["triton_spatial_fn"]
    spatial_corr._SYCL_XPU_EXT = saved["sycl_ext"]
    spatial_corr._SYCL_XPU_EXT_LOADED = saved["sycl_ext_loaded"]
    if saved["triton_module"] is _MISSING:
        sys.modules.pop("localcorr.triton", None)
    else:
        sys.modules["localcorr.triton"] = saved["triton_module"]
    for key, var in (
        ("onnx_backend_env", "LOCALCORR_ONNX_BACKEND"),
        ("sycl_preload_env", "LOCALCORR_SYCL_PRELOAD"),
    ):
        if saved[key] is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = saved[key]


@pytest.fixture
def onnx_export_mode(monkeypatch):
    """Pretend that torch.onnx.export (legacy tracer) is running.

    Returns a helper that can turn the in-export flag on/off; select the
    export backend with localcorr.set_onnx_backend().
    """

    def set_flag(active: bool = True) -> None:
        import torch

        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: active)

    set_flag(True)
    return set_flag
