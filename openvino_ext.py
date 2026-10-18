"""OpenVINO integration for the localcorr SYCL kernels.

This module wires the custom_ops::LocalCorrCudaPyramid ONNX node (the
same node the TensorRT plugin of localcorr/cuda_pyramid resolves) into
OpenVINO through the SYCL-based custom operation extension built from
localcorr/sycl_pyramid/ (liblocalcorr_sycl_pyramid_ov.so).

IMPORTANT: import this module BEFORE importing openvino in the process.
The SYCL runtime inside the extension discovers its OpenCL CPU platform
through the oneAPI libOpenCL, which must be loaded into the process
before OpenVINO (or any other package) loads a different libOpenCL
first. The one-time preload below is idempotent and best-effort; set
LOCALCORR_SYCL_PRELOAD=0 to skip it.

Main entry points:
- add_localcorr_extension(core): register the custom-op extension on an
  OpenVINO Core (no-op if the library is missing).
- localcorr_extension_path(): path of the extension .so or None.
- onnx_has_localcorr_nodes(path): whether an ONNX model contains
  custom_ops::LocalCorrCudaPyramid nodes.
"""

import ctypes
import os
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

_EXTENSION_NAME = "liblocalcorr_sycl_pyramid_ov.so"
_KERNEL_NAME = "liblocalcorr_sycl_pyramid.so"

_PRELOAD_LIBS = [
    "libintlc.so.5",
    "libsvml.so",
    "libimf.so",
    "libirng.so",
    "libtbb.so.12",
    "libumf.so.1",
    "libcommon_clang.so",
    "libOpenCL.so",
]


def _oneapi_lib_dirs() -> list:
    root = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
    candidates = []
    latest = Path(root) / "compiler" / "latest" / "lib"
    if latest.exists():
        candidates.append(latest)
    for entry in sorted(Path(root).glob("compiler/*/lib"), reverse=True):
        if entry not in candidates:
            candidates.append(entry)
    shared = Path(root) / "lib"
    if shared.exists():
        candidates.append(shared)
    for entry in sorted(Path(root).glob("*/lib"), reverse=True):
        if entry not in candidates:
            candidates.append(entry)
    tbb = Path(root) / "tbb" / "latest" / "lib"
    if tbb.exists():
        candidates.append(tbb)
    for entry in sorted(Path(root).glob("tbb/*/lib"), reverse=True):
        if entry not in candidates:
            candidates.append(entry)
    return candidates


def _find_oneapi_lib(name) -> Path:
    for lib_dir in _oneapi_lib_dirs():
        candidate = lib_dir / name
        if candidate.exists():
            return candidate
    return None


def _preload_oneapi_runtime() -> None:
    """Load libintelocl's oneAPI dependencies with RTLD_LOCAL.

    Loading with RTLD_LOCAL is enough for the ICD scanner to then load
    libintelocl itself successfully: a library that is already loaded is
    reused by soname when resolving DT_NEEDED entries, so GLOBAL symbol
    export is not required.

    libintelocl.so is deliberately NOT preloaded: it bundles its own
    opencl-clang/SPIR-V toolchain, and exporting those symbols
    process-wide breaks the OpenVINO GPU plugin's JIT compiler
    (clBuildProgram fails with "unknown argument '--spirv-ext=...'",
    a symbol clash with the newer toolchain the GPU plugin uses).

    RTLD_LOCAL (not RTLD_GLOBAL) is essential for the same reason:
    exporting the oneAPI libcommon_clang.so symbols process-wide makes
    the GPU plugin's JIT resolve clang symbols against the older oneAPI
    toolchain, which breaks compilation of JIT-generated kernels (e.g.
    the i8 Concat of NNCF-quantized models fails with "Failed to select
    implementation for ... Concat").
    """
    if os.environ.get("LOCALCORR_SYCL_PRELOAD", "1") == "0":
        return

    lib_dirs = _oneapi_lib_dirs()

    def try_load(name):
        for lib_dir in lib_dirs:
            lib_path = lib_dir / name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_LOCAL)
                return True
            except OSError:
                continue
        return False

    # Dependencies of libintelocl.so that live outside the default loader
    # search path (order matters for their own interdependencies).
    # RTLD_LOCAL: they must be resolvable by soname for libintelocl, but
    # their symbols must NOT be exported process-wide (see docstring).
    for name in _PRELOAD_LIBS:
        try_load(name)


_preload_oneapi_runtime()


def localcorr_extension_path():
    """Return the path of the built extension library, or None."""
    here = Path(__file__).parent.resolve()
    so = here / "sycl_pyramid" / "build" / _EXTENSION_NAME
    return so if so.exists() else None


def localcorr_kernel_path():
    """Return the path of the built SYCL kernel library, or None."""
    here = Path(__file__).parent.resolve()
    so = here / "sycl_pyramid" / "build" / _KERNEL_NAME
    return so if so.exists() else None


def add_localcorr_extension(core):
    """Register the localcorr SYCL custom-op extension on an OpenVINO Core.

    Returns True if the extension was added, False if the library has not
    been built (build it with: python -m localcorr.build_native sycl).
    """
    so = localcorr_extension_path()
    if so is None:
        return False
    core.add_extension(str(so))
    return True


def onnx_has_localcorr_nodes(onnx_path) -> bool:
    """Check whether an ONNX model contains LocalCorrCudaPyramid nodes."""
    try:
        import onnx
    except ImportError:
        # Fall back to a raw string scan of the file.
        try:
            with open(onnx_path, "rb") as f:
                return b"LocalCorrCudaPyramid" in f.read()
        except OSError:
            return False
    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception:
        try:
            with open(onnx_path, "rb") as f:
                return b"LocalCorrCudaPyramid" in f.read()
        except OSError:
            return False
    return any(node.op_type == "LocalCorrCudaPyramid" for node in model.graph.node)
