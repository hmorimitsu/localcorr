"""localcorr: local correlation kernels for optical flow networks.

Provides fast local correlation implementations for RAFT-style and
PWCNet-style models, for both eager PyTorch execution (with training
support) and TensorRT inference:

- localcorr_triton: Triton JIT kernels. No compilation required, just
  `pip install triton`. Powers the eager forward/backward of the
  localcorr_triton and localcorr_onnx corr modes.
  triton ships Linux-only wheels on PyPI; on other platforms the package
  still imports without it (pure-PyTorch and ONNX-export paths), and the
  eager kernels raise a clear ImportError on first use.
- localcorr_cuda: an AOT-compiled CUDA kernel + TensorRT
  IPluginV3 plugin that computes all correlation pyramid levels in one
  kernel launch. Used for eager CUDA inference (called through ctypes,
  forward-only) and to build/run TensorRT engines.
- localcorr_sycl_pyramid: a SYCL port of the fused kernel plus an
  OpenVINO custom-op extension (localcorr.openvino_ext) that resolves
  the LocalCorrCudaPyramid ONNX node on Intel platforms. Required only
  to convert/run OpenVINO models with the custom node. Together with
  the localcorr/xpu extension it powers the localcorr_sycl corr mode:
  eager execution on torch.xpu and the OpenVINO custom op share the same
  SYCL kernel.
- onnx-generic: when exporting to ONNX for runtimes other than
  TensorRT/OpenVINO, the correlation can be emitted with standard ONNX
  ops (grid_sample based) instead of the custom node. Select it with
  set_onnx_backend("onnx") before export, or use the localcorr_onnx
  corr mode in the ptlflow models, which forces the standard ops at
  export time regardless of the global backend.
- localcorr_auto: available in both corr-mode factories
  (get_localcorr_block / get_localcorr_sampler); picks the eager engine
  automatically from the device of the tensors — the AOT CUDA kernel on
  NVIDIA GPUs when its library has been built (trainable), falling back
  to the Triton kernel, and the SYCL kernel on torch.xpu when the xpu
  extension is built (Triton fallback otherwise). The PWCNet-style
  implementation is the device-dispatching LocalCorrCudaPyramidSampler.

The AOT components are built on demand, from any directory (installed
package or checkout alike), with a single command:

    python -m localcorr.build_native cuda|sycl|xpu|all

See README.md for build and usage instructions.
"""

from .backend import get_onnx_backend, set_onnx_backend
from .blocks import (
    get_cuda_pyramid_library,
    get_localcorr_sampler,
    LocalCorrCudaPyramidFunction,
    LocalCorrCudaPyramidSpatialBlock,
    LocalCorrCudaPyramidSpatialOnnx,
    get_localcorr_block,
    LocalCorrCudaPyramidBlock,
    LocalCorrCudaPyramidOnnx,
    LocalCorrCudaSpatialSampler,
    LocalCorrOnnxBlock,
    LocalCorrOnnxSpatialSampler,
    LocalCorrSyclBlock,
    LocalCorrSyclSpatialSampler,
    LocalCorrTritonBlock,
    LocalCorrTritonFusedBlock,
)
from .spatial_corr import (
    get_sycl_xpu_extension,
    LocalCorrCudaPyramidSampler,
    LocalCorrCudaPyramidSpatialOnnx,
)

try:
    from .triton import (
        LocalCorrTriton,
        LocalCorrTritonFused,
        TritonSpatialCorrelationFunction,
        TritonSpatialCorrelationSampler,
    )
except (ImportError, OSError):
    # triton publishes Linux-only wheels on PyPI. Without it the package
    # still imports and the pure-PyTorch / ONNX-export paths work; the
    # kernel symbols are set to None and the eager blocks raise a clear
    # ImportError on first use (see blocks._get_triton_kernels).
    LocalCorrTriton = None
    LocalCorrTritonFused = None
    TritonSpatialCorrelationFunction = None
    TritonSpatialCorrelationSampler = None

__version__ = "0.1.0"

__all__ = [
    "get_localcorr_block",
    "get_localcorr_sampler",
    "get_cuda_pyramid_library",
    "get_sycl_xpu_extension",
    "get_onnx_backend",
    "set_onnx_backend",
    "LocalCorrCudaPyramidBlock",
    "LocalCorrCudaPyramidFunction",
    "LocalCorrCudaPyramidOnnx",
    "LocalCorrCudaPyramidSpatialBlock",
    "LocalCorrCudaPyramidSpatialOnnx",
    "LocalCorrCudaPyramidSampler",
    "LocalCorrCudaPyramidSpatialOnnx",
    "LocalCorrCudaSpatialSampler",
    "LocalCorrOnnxBlock",
    "LocalCorrOnnxSpatialSampler",
    "LocalCorrSyclBlock",
    "LocalCorrSyclSpatialSampler",
    "LocalCorrTritonBlock",
    "LocalCorrTritonFusedBlock",
    "LocalCorrTriton",
    "LocalCorrTritonFused",
    "TritonSpatialCorrelationFunction",
    "TritonSpatialCorrelationSampler",
]
