"""PWCNet-style spatial correlation sampler.

Equivalent to IterSpatialCorrelationSampler(kernel_size=1, patch_size,
padding=0), with two execution paths:

- Eager: the Triton spatial correlation kernel (see
  localcorr/triton/spatial_corr.py), which provides a fast forward and a
  memory-efficient backward for training. No build step required.
- ONNX export: one LocalCorrCudaPyramid plugin node (levels=1, ymajor)
  computed by the fused CUDA kernel from localcorr/cuda_pyramid/ with a
  single kernel launch (the fastest inference path). With the "onnx"
  backend selected via set_onnx_backend(), or with onnx_generic=True on
  the sampler, the correlation is exported with standard ONNX ops instead
  (see localcorr/onnx_generic.py), for runtimes such as OpenVINO or for
  plugin-free TensorRT engines.
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .backend import use_onnx_generic_ops
from .onnx_generic import spatial_corr_onnx_generic

logger = logging.getLogger(__name__)

_SYCL_XPU_EXT = None
_SYCL_XPU_EXT_LOADED = False

_TRITON_SPATIAL_FN = None


class LocalCorrSyclFunction(torch.autograd.Function):
    """Autograd wrapper of the fused SYCL pyramid kernel (torch.xpu).

    forward() launches the fused correlation kernel through the
    localcorr/xpu extension; backward() launches its backward kernel
    (gradients for fmap1 and fmap2_packed; zeros for coords, matching
    the Triton backward and RAFT's detached coordinates). Inputs must
    be contiguous XPU tensors in the layout documented by the extension
    (fmap1 NHWC, fmap2_packed batch-major, coords planar x/y).
    """

    @staticmethod
    def forward(ctx, fmap1, fmap2_packed, coords, radius, levels, dilation,
                ymajor):
        ext = get_sycl_xpu_extension()
        corr = ext.forward(fmap1, fmap2_packed, coords, radius, levels,
                           dilation, ymajor)
        ctx.save_for_backward(fmap1, fmap2_packed, coords)
        ctx.kernel_params = (int(radius), int(levels), int(dilation),
                             bool(ymajor))
        return corr

    @staticmethod
    def backward(ctx, corr_grad):
        fmap1, fmap2_packed, coords = ctx.saved_tensors
        radius, levels, dilation, ymajor = ctx.kernel_params
        ext = get_sycl_xpu_extension()
        if not hasattr(ext, "backward"):
            raise RuntimeError(
                "The localcorr SYCL XPU extension is too old to provide a "
                "backward pass; rebuild it with: "
                "python -m localcorr.build_native xpu"
            )
        corr_grad = corr_grad.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = ext.backward(
            fmap1, fmap2_packed, coords, corr_grad, radius, levels, dilation,
            ymajor,
        )
        return fmap1_grad, fmap2_grad, coords_grad, None, None, None, None


def _get_triton_spatial_function():
    """Import the Triton spatial correlation kernel lazily and cache it.

    triton publishes Linux-only wheels on PyPI, so it is an optional
    dependency; the eager kernel raises a clear error on first use when
    it is unavailable.
    """
    global _TRITON_SPATIAL_FN
    if _TRITON_SPATIAL_FN is None:
        try:
            from .triton import TritonSpatialCorrelationFunction
        except (ImportError, OSError) as exc:
            raise ImportError(
                "The localcorr eager kernels require the triton package, "
                "which could not be imported. Install it with: "
                "pip install triton"
            ) from exc
        _TRITON_SPATIAL_FN = TritonSpatialCorrelationFunction
    return _TRITON_SPATIAL_FN


def get_sycl_xpu_extension():
    """Return the localcorr SYCL XPU extension module, or None.

    The extension (localcorr/xpu/build/localcorr_sycl_xpu*.so) wraps the
    fused correlation SYCL kernel for torch.xpu tensors. Built with
    localcorr/xpu/build.sh; the import is attempted once and the result
    cached. Returns None when unavailable (e.g. not built, or a CPU-only
    torch).
    """
    global _SYCL_XPU_EXT, _SYCL_XPU_EXT_LOADED
    if _SYCL_XPU_EXT_LOADED:
        return _SYCL_XPU_EXT
    _SYCL_XPU_EXT_LOADED = True
    build_dir = Path(__file__).parent / "xpu" / "build"
    if not build_dir.exists():
        return None
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    try:
        import localcorr_sycl_xpu as ext

        _SYCL_XPU_EXT = ext
    except ImportError as exc:
        logger.warning(
            "Could not import the localcorr SYCL XPU extension from %s "
            "(%s); XPU tensors fall back to the Triton kernel. Build it "
            "with: python -m localcorr.build_native xpu",
            build_dir,
            exc,
        )
    return _SYCL_XPU_EXT


class LocalCorrCudaPyramidSpatialOnnx(torch.autograd.Function):
    """ONNX-export wrapper of the single-level spatial correlation.

    forward() returns an empty tensor with the correct shape so tracing
    can continue; the output is produced by symbolic(), which records a
    "custom_ops::LocalCorrCudaPyramid" node that the TensorRT ONNX parser
    resolves to the LocalCorrCudaPyramid IPluginV3 plugin. Eager execution
    goes through the Triton kernel instead.
    """

    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, radius):
        # fmap1/fmap2: [B, H, W, C] NHWC; coords: [B, 2, H, W] planar
        # (x in channel 0).
        radius = int(radius)
        batch_size, height_1, width_1, _ = fmap1.shape
        rd = 2 * radius + 1
        return fmap1.new_empty((batch_size, 1, rd * rd, height_1, width_1))

    @staticmethod
    def symbolic(g, fmap1, fmap2, coords, radius):
        # The correlation is computed for a single level (fmap2 has the
        # same resolution as fmap1), so the fused multi-level kernel is
        # used with levels=1. PWCNet uses the y-major displacement channel
        # order (d = dy_idx * rd + dx_idx), so ymajor=1.
        fmap1_sizes = fmap1.type().sizes() if fmap1.type() is not None else None
        fmap2_sizes = fmap2.type().sizes() if fmap2.type() is not None else None
        # fmap2 can lose its static shape during tracing (e.g. when it is
        # produced by a grid_sample fed with an interpolate whose size is
        # read from another tensor's .size()). The batch size needed to
        # pack fmap2 to [B, -1] is the same for both inputs, so fall back
        # to fmap1's type when fmap2's is unavailable.
        if fmap2_sizes is not None:
            batch_size = fmap2_sizes[0]
        elif fmap1_sizes is not None:
            batch_size = fmap1_sizes[0]
        else:
            raise RuntimeError(
                "Could not infer the batch size of the LocalCorrCudaPyramid "
                "ONNX node: both fmap1 and fmap2 have unknown static shapes. "
                "Ensure at least one correlation input keeps a static shape "
                "when tracing."
            )
        fmap2_packed = g.op(
            "Reshape",
            fmap2,
            g.op(
                "Constant",
                value_t=torch.tensor([batch_size, -1], dtype=torch.int64),
            ),
        )
        node = g.op(
            "custom_ops::LocalCorrCudaPyramid",
            fmap1,
            fmap2_packed,
            coords,
            radius_i=int(radius),
            levels_i=1,
            dilation_i=1,
            ymajor_i=1,
        )
        coords_sizes = coords.type().sizes()
        if fmap1_sizes is not None and coords_sizes is not None:
            rd = 2 * int(radius) + 1
            output_sizes = [
                coords_sizes[0],
                1,
                rd * rd,
                fmap1_sizes[1],
                fmap1_sizes[2],
            ]
            node.setType(fmap1.type().with_sizes(output_sizes))
        return node


class LocalCorrCudaPyramidSampler(nn.Module):
    """Spatial correlation sampler equivalent to
    IterSpatialCorrelationSampler(kernel_size=1, patch_size, padding=0).

    Execution paths:
    - CUDA tensors: by default the fused AOT CUDA kernel from
      localcorr/cuda_pyramid/ (the localcorr_cuda engine, with backward
      for training); falls back to the Triton spatial correlation kernel
      when the CUDA kernel library has not been built (or cannot run,
      e.g. unsupported dtype or an inference-only build under training).
    - XPU tensors (Intel GPUs, torch.xpu): by default the fused SYCL
      kernel built from localcorr/xpu/ (the CUDA-plugin equivalent,
      fastest inference, with backward for training when the extension
      provides it); with xpu_backend="triton" the Triton kernel instead
      (requires triton-xpu). Falls back to Triton when the SYCL
      extension has not been built.
    - ONNX export: one LocalCorrCudaPyramid plugin node (levels=1, ymajor)
      or, with onnx_generic=True, standard ops.

    Requires triton (as the CUDA fallback, and on XPU via triton-xpu) or
    one of the native extensions.
    """

    def __init__(
        self,
        kernel_size: int = 1,
        patch_size: int = 9,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        dilation_patch: int = 1,
        onnx_generic: bool = False,
        xpu_backend: str = "sycl",
    ) -> None:
        super(LocalCorrCudaPyramidSampler, self).__init__()
        assert kernel_size == 1, "Only kernel_size=1 is supported."
        assert stride == 1, "Only stride=1 is supported."
        assert padding == 0, "Only padding=0 is supported."
        assert dilation == 1, "Only dilation=1 is supported."
        assert dilation_patch == 1, "Only dilation_patch=1 is supported."
        assert xpu_backend in (
            "sycl",
            "triton",
        ), f"Unknown xpu_backend: {xpu_backend!r}"

        self.md = patch_size // 2
        self.patch_size = patch_size
        self.onnx_generic = onnx_generic
        self.xpu_backend = xpu_backend
        self._coords = None

    def _identity_coords(self, b, h, w, dtype, device):
        if (
            self._coords is None
            or self._coords.shape[0] != b
            or self._coords.shape[2] != h
            or self._coords.shape[3] != w
            or self._coords.dtype != dtype
            or self._coords.device != device
        ):
            ys = torch.arange(h, device=device, dtype=dtype)
            xs = torch.arange(w, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            # Planar layout expected by LocalCorrCudaPyramid:
            # channel 0 = x, channel 1 = y.
            coords = torch.stack([gx, gy], dim=0)
            self._coords = coords[None].repeat(b, 1, 1, 1).contiguous()
        return self._coords

    def _cuda_pyramid(self, input1, input2):
        """Fused AOT CUDA kernel from localcorr/cuda_pyramid/.

        Returns None when the kernel cannot be used (library not built,
        unsupported dtype, or an inference-only build while gradients
        are required), so that forward() falls back to the Triton kernel.
        """
        # Imported here: blocks.py imports this module at load time.
        from .blocks import _cuda_pyramid_auto_available, _localcorr_cuda_pyramid_forward

        if not _cuda_pyramid_auto_available(input1, input2):
            return None
        b, c, h, w = input1.shape
        fmap1 = input1.permute(0, 2, 3, 1).contiguous()
        fmap2_packed = input2.permute(0, 2, 3, 1).contiguous().view(b, -1)
        coords = self._identity_coords(b, h, w, input1.dtype, input1.device)
        # ymajor=1: d = dy_idx * patch_size + dx_idx, the channel order
        # of the other samplers; single level, dilation 1. The autograd
        # function adds the fused backward pass when the library
        # provides it (the availability check above already rejected
        # inference-only builds when gradients are required).
        return _localcorr_cuda_pyramid_forward(
            fmap1, fmap2_packed, coords, self.md, 1, 1, True
        )

    def _sycl_xpu(self, input1, input2):
        """Fused SYCL kernel on torch.xpu (localcorr/xpu extension)."""
        ext = get_sycl_xpu_extension()
        if ext is None:
            return None
        b, c, h, w = input1.shape
        fmap1 = input1.permute(0, 2, 3, 1).contiguous()
        fmap2_packed = input2.permute(0, 2, 3, 1).contiguous().view(b, -1)
        coords = self._identity_coords(b, h, w, input1.dtype, input1.device)
        # ymajor=1: d = dy_idx * patch_size + dx_idx, the channel order of
        # the other samplers. The autograd wrapper adds the backward pass
        # when the extension provides it (older builds fall back to an
        # inference-only error via the grad guard in forward()).
        if (
            torch.is_grad_enabled()
            and (input1.requires_grad or input2.requires_grad)
            and not hasattr(ext, "backward")
        ):
            raise RuntimeError(
                'xpu_backend="sycl" is inference-only with this build of the '
                "localcorr SYCL extension (no backward pass). Rebuild it "
                "with: python -m localcorr.build_native xpu"
            )
        corr = LocalCorrSyclFunction.apply(
            fmap1, fmap2_packed, coords, self.md, 1, 1, True
        )
        return corr

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input1.shape

        if use_onnx_generic_ops() or (
            self.onnx_generic and torch.onnx.is_in_onnx_export()
        ):
            # Standard-ops ONNX export (e.g. for OpenVINO or plugin-free
            # TensorRT): computes the correlation with regular PyTorch
            # ops, traced into the graph.
            return spatial_corr_onnx_generic(input1, input2, self.md)

        if torch.onnx.is_in_onnx_export():
            fmap1 = input1.permute(0, 2, 3, 1).contiguous()
            fmap2 = input2.permute(0, 2, 3, 1).contiguous()
            coords = self._identity_coords(b, h, w, input1.dtype, input1.device)
            corr = LocalCorrCudaPyramidSpatialOnnx.apply(fmap1, fmap2, coords, self.md)
        elif input1.is_cuda:
            # NVIDIA GPUs: prefer the fused AOT CUDA kernel
            # (localcorr_cuda); fall back to the Triton kernel when its
            # library is not usable.
            corr = self._cuda_pyramid(input1, input2)
            if corr is None:
                corr = (
                    _get_triton_spatial_function()
                    .apply(input1, input2, self.md)
                    .unsqueeze(1)
                )
        elif input1.is_xpu and self.xpu_backend == "sycl":
            corr = self._sycl_xpu(input1, input2)
            if corr is None:
                # Extension not built: fall back to the Triton kernel.
                corr = (
                    _get_triton_spatial_function()
                    .apply(input1, input2, self.md)
                    .unsqueeze(1)
                )
        else:
            # The Triton kernel (CUDA, and XPU via triton-xpu) produces the
            # same y-major channel order
            # (channel = dy_idx * patch_size + dx_idx).
            corr = (
                _get_triton_spatial_function()
                .apply(input1, input2, self.md)
                .unsqueeze(1)
            )

        # [B, 1, patch_size * patch_size, H, W] -> [B, patch_size, patch_size, H, W]
        # with [b, dy_idx, dx_idx, h, w] ordering, matching the other samplers.
        return corr.view(b, self.patch_size, self.patch_size, h, w)
