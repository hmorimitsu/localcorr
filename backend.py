"""Selection of the backend used when exporting the local correlation to ONNX.

Two backends are available:

- "tensorrt" (default): the correlation is emitted as a
  "custom_ops::LocalCorrCudaPyramid" ONNX node, which the TensorRT ONNX
  parser resolves to the IPluginV3 plugin built from localcorr/cuda_pyramid.
- "onnx": the correlation is emitted with standard ONNX ops only
  (matmul + grid_sample), so the exported model can be converted and run
  by any ONNX-compatible runtime such as OpenVINO, without any plugin.

The backend only affects ONNX export; eager execution always uses the
Triton kernels. The default can also be set through the
LOCALCORR_ONNX_BACKEND environment variable.
"""

import os

_TENSORRT_BACKEND = "tensorrt"
_ONNX_BACKEND = "onnx"

_BACKENDS = (_TENSORRT_BACKEND, _ONNX_BACKEND)
_ENV_VAR = "LOCALCORR_ONNX_BACKEND"

_onnx_backend = None


def set_onnx_backend(backend: str) -> None:
    """Select the backend used for the local correlation during ONNX export.

    Args:
        backend: "tensorrt" (custom plugin node) or "onnx" (standard ops).
    """
    global _onnx_backend
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown ONNX backend: {backend!r}. Expected one of {_BACKENDS}."
        )
    _onnx_backend = backend


def get_onnx_backend() -> str:
    """Return the ONNX export backend ("tensorrt" or "onnx")."""
    global _onnx_backend
    if _onnx_backend is None:
        _onnx_backend = os.environ.get(_ENV_VAR, _TENSORRT_BACKEND)
        if _onnx_backend not in _BACKENDS:
            raise ValueError(
                f"Invalid value for {_ENV_VAR}: {_onnx_backend!r}. "
                f"Expected one of {_BACKENDS}."
            )
    return _onnx_backend


def use_onnx_generic_ops() -> bool:
    """Whether the ONNX export should emit standard ops instead of the plugin node."""
    import torch

    return torch.onnx.is_in_onnx_export() and get_onnx_backend() == _ONNX_BACKEND
