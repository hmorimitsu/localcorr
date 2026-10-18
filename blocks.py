"""RAFT-style local correlation blocks.

The available corr modes are:

- localcorr_triton: local correlation with the fused Triton kernel: all
  pyramid levels are computed in one kernel launch. Eager forward and
  backward (training support). The simpler per-level LocalCorrTritonBlock
  (one kernel launch per pyramid level) is kept as an internal reference
  implementation, used by the tests to cross-check the fused kernel.
- localcorr_cuda: eager inference and training with the AOT-compiled
  fused CUDA kernel from localcorr/cuda_pyramid/ (built with
  python -m localcorr.build_native cuda), called through ctypes on CUDA
  tensors. The kernel provides forward and backward (gradients for the
  feature maps; the sampling coordinates stay detached, like the Triton
  backward). During ONNX export, the whole pyramid is emitted as one
  LocalCorrCudaPyramid TensorRT plugin node computed by the same CUDA
  kernel (the fastest inference path on NVIDIA).
- localcorr_onnx: eager execution is the fused Triton kernel (same as
  localcorr_triton, with training support), but ONNX export always
  emits the standard-ops (matmul + grid_sample) correlation instead of
  the TensorRT plugin node, regardless of the backend selected with
  set_onnx_backend(). Useful to build plugin-free TensorRT engines.
- localcorr_sycl: the Intel counterpart of localcorr_cuda. Eager
  execution on torch.xpu devices with the fused SYCL kernel from
  localcorr/xpu/ (built with python -m localcorr.build_native xpu), all
  pyramid levels in one launch, with the backward pass for training.
  During ONNX export, the LocalCorrCudaPyramid node is emitted, resolved by the
  OpenVINO custom-op extension from localcorr/sycl_pyramid/ (or by the
  TensorRT plugin on NVIDIA).
- localcorr_auto: picks the engine automatically instead of committing
  to one at model-definition time. The device of the feature maps (known
  at construction in the RAFT-style API) selects the eager engine: on
  NVIDIA GPUs the fused AOT CUDA kernel (localcorr_cuda) when its
  library has been built (trainable), with the fused Triton kernel as
  fallback, and, on torch.xpu (Intel GPUs), the fused SYCL kernel when
  the localcorr/xpu extension has been built (trainable), with the
  Triton kernel (triton-xpu) as fallback (xpu_backend="triton" forces
  it). During ONNX export with the default "tensorrt" backend, the
  LocalCorrCudaPyramid plugin node is emitted device-independently (the
  fused Triton kernel itself cannot be exported); with the "onnx"
  backend, the standard ops are emitted as for every other mode.

The ONNX export backend can be switched with set_onnx_backend() (see
localcorr/backend.py): with the "onnx" backend, the correlation is
exported with standard ONNX ops (matmul + grid_sample) so the model can
be converted to runtimes such as OpenVINO without the TensorRT plugin.
"""

import ctypes
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .backend import use_onnx_generic_ops
from .onnx_generic import localcorr_pyramid_onnx_generic, spatial_corr_onnx_generic
from .spatial_corr import (
    _get_triton_spatial_function,
    get_sycl_xpu_extension,
    LocalCorrCudaPyramidSampler,
    LocalCorrSyclFunction,
)

_TRITON_KERNELS = None

_CUDA_PYRAMID_LIB = None
_CUDA_PYRAMID_LIB_LOADED = False

logger = logging.getLogger(__name__)


def get_cuda_pyramid_library():
    """Return the built CUDA pyramid kernel library, or None.

    The library (cuda_pyramid/build/liblocalcorr_cuda_pyramid_plugin.so,
    built with python -m localcorr.build_native cuda) contains both the
    TensorRT plugin and the plain extern-C kernel launchers
    launchLocalCorrCudaPyramidForward and
    launchLocalCorrCudaPyramidBackward. Eager (PyTorch) execution calls
    those launchers through ctypes on torch's current CUDA stream, so no
    torch extension compilation is needed. The import is attempted once
    and the result cached. Returns None when the library has not been
    built or cannot be loaded (e.g. missing TensorRT runtime libraries).
    """
    global _CUDA_PYRAMID_LIB, _CUDA_PYRAMID_LIB_LOADED
    if _CUDA_PYRAMID_LIB_LOADED:
        return _CUDA_PYRAMID_LIB
    _CUDA_PYRAMID_LIB_LOADED = True
    lib_path = (
        Path(__file__).parent
        / "cuda_pyramid"
        / "build"
        / "liblocalcorr_cuda_pyramid_plugin.so"
    )
    if not lib_path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(lib_path))
        launch = lib.launchLocalCorrCudaPyramidForward
        launch.restype = ctypes.c_bool
        launch.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 7
            + [ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p]
        )
        # The backward launcher is resolved separately so libraries from
        # before its introduction still load (with backward disabled).
        try:
            backward = lib.launchLocalCorrCudaPyramidBackward
            backward.restype = ctypes.c_bool
            backward.argtypes = (
                [ctypes.c_void_p] * 5
                + [ctypes.c_void_p]
                + [ctypes.c_int] * 7
                + [ctypes.c_bool, ctypes.c_bool, ctypes.c_void_p]
            )
            lib.launchLocalCorrCudaPyramidBackward = backward
        except AttributeError:
            lib.launchLocalCorrCudaPyramidBackward = None
        _CUDA_PYRAMID_LIB = lib
    except OSError as exc:
        logger.warning(
            "Could not load the localcorr CUDA pyramid library from %s (%s); "
            "corr_mode=localcorr_cuda will not work in eager mode. "
            "Rebuild it with: python -m localcorr.build_native cuda",
            lib_path,
            exc,
        )
    return _CUDA_PYRAMID_LIB


def _get_triton_kernels():
    """Import the Triton pyramid kernels lazily and cache them.

    triton publishes Linux-only wheels on PyPI, so it is an optional
    dependency: without it the package still imports and the pure-PyTorch
    and ONNX-export paths work. The eager kernels raise this clear error
    on first use instead.
    """
    global _TRITON_KERNELS
    if _TRITON_KERNELS is None:
        try:
            from .triton import LocalCorrTriton, LocalCorrTritonFused
        except (ImportError, OSError) as exc:
            raise ImportError(
                "The localcorr eager kernels require the triton package, "
                "which could not be imported. Install it with: "
                "pip install triton"
            ) from exc
        _TRITON_KERNELS = (LocalCorrTriton, LocalCorrTritonFused)
    return _TRITON_KERNELS


class LocalCorrCudaPyramidOnnx(torch.autograd.Function):
    """ONNX-export wrapper of the fused multi-level local correlation.

    Only used during ONNX export (legacy tracer): forward() returns an
    empty tensor with the correct shape and symbolic() records a
    "custom_ops::LocalCorrCudaPyramid" node, which the TensorRT ONNX parser
    maps to the LocalCorrCudaPyramid IPluginV3 plugin registered by
    localcorr/cuda_pyramid/build/liblocalcorr_cuda_pyramid_plugin.so.
    Eager execution goes through the Triton kernels instead.
    """

    @staticmethod
    def forward(ctx, fmap1, fmap2_packed, coords, radius, num_levels, dilation=1):
        radius = int(radius)
        num_levels = int(num_levels)
        batch_size, height_1, width_1, _ = fmap1.shape
        rd = 2 * radius + 1
        return fmap1.new_empty((batch_size, num_levels, rd * rd, height_1, width_1))

    @staticmethod
    def symbolic(g, fmap1, fmap2_packed, coords, radius, num_levels, dilation=1):
        # The "radius", "levels", "dilation" and "ymajor" attributes are
        # forwarded to the TensorRT plugin creator as PluginFields with the
        # same names (ymajor=0 is the RAFT x-major channel order). The
        # OpenVINO custom-op extension of localcorr/sycl_pyramid also
        # requires all four attributes to be present.
        node = g.op(
            "custom_ops::LocalCorrCudaPyramid",
            fmap1,
            fmap2_packed,
            coords,
            radius_i=int(radius),
            levels_i=int(num_levels),
            dilation_i=int(dilation),
            ymajor_i=0,
        )
        fmap1_sizes = fmap1.type().sizes()
        coords_sizes = coords.type().sizes()
        if fmap1_sizes is not None and coords_sizes is not None:
            rd = 2 * int(radius) + 1
            output_sizes = [
                coords_sizes[0],
                int(num_levels),
                rd * rd,
                fmap1_sizes[1],
                fmap1_sizes[2],
            ]
            node.setType(fmap1.type().with_sizes(output_sizes))
        return node


class LocalCorrTritonBlock:
    """Per-level local correlation with the Triton kernels.

    The feature maps are converted to NHWC (and the fmap2 pyramid built)
    only once because they are constant throughout RAFT's recurrent update
    iterations.
    """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.dim = fmap1.shape[1]

        if use_onnx_generic_ops():
            # Standard-ops ONNX export (e.g. for OpenVINO): keep the
            # feature maps in NCHW and compute the correlation with
            # grid_sample ops when the block is called.
            self.fmap1_nchw = fmap1
            self.fmap2_nchw = fmap2
            return

        self.fmap1 = fmap1.permute(0, 2, 3, 1).contiguous()
        self.fmap2_pyramid = []
        fmap2_curr = fmap2
        for i in range(num_levels):
            if i > 0:
                fmap2_curr = F.avg_pool2d(fmap2_curr, 2, stride=2)
            self.fmap2_pyramid.append(fmap2_curr.permute(0, 2, 3, 1).contiguous())

    def __call__(self, coords):
        if use_onnx_generic_ops():
            B, _, H, W = coords.shape
            corr = localcorr_pyramid_onnx_generic(
                self.fmap1_nchw,
                self.fmap2_nchw,
                coords,
                num_levels=self.num_levels,
                radius=self.radius,
            )
            return corr.reshape(B, -1, H, W) / math.sqrt(self.dim)

        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape

        localcorr_triton, _ = _get_triton_kernels()
        corr_list = []
        for i in range(self.num_levels):
            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr = localcorr_triton.apply(
                self.fmap1, self.fmap2_pyramid[i], coords_i, self.radius
            )
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / math.sqrt(self.dim)


class LocalCorrTritonFusedBlock:
    """Local correlation with all pyramid levels in one Triton launch."""

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.dim = fmap1.shape[1]

        if use_onnx_generic_ops():
            # Standard-ops ONNX export (e.g. for OpenVINO): keep the
            # feature maps in NCHW and compute the correlation with
            # grid_sample ops when the block is called.
            self.fmap1_nchw = fmap1
            self.fmap2_nchw = fmap2
            return

        self.fmap1 = fmap1.permute(0, 2, 3, 1).contiguous()
        fmap2_pyramid = [fmap2]
        for _ in range(1, num_levels):
            fmap2_pyramid.append(F.avg_pool2d(fmap2_pyramid[-1], 2, stride=2))
        fmap2_pyramid = [
            fmap.permute(0, 2, 3, 1).contiguous() for fmap in fmap2_pyramid
        ]
        self.level_shapes = tuple(
            (fmap.shape[1], fmap.shape[2]) for fmap in fmap2_pyramid
        )
        self.fmap2_packed = torch.cat([fmap.reshape(-1) for fmap in fmap2_pyramid])
        self.level_offsets = []
        offset = 0
        for fmap in fmap2_pyramid:
            self.level_offsets.append(offset)
            offset += fmap.numel()
        self.level_offsets = tuple(self.level_offsets)
        self.fmap2_offsets = torch.tensor(
            self.level_offsets, dtype=torch.int64, device=fmap1.device
        )
        self.fmap2_shapes = torch.tensor(
            self.level_shapes, dtype=torch.int32, device=fmap1.device
        ).reshape(-1)
        self.level_scales = torch.tensor(
            [1 / 2**level for level in range(num_levels)],
            dtype=torch.float32,
            device=fmap1.device,
        )

    def __call__(self, coords):
        B, _, H, W = coords.shape
        if use_onnx_generic_ops():
            corr = localcorr_pyramid_onnx_generic(
                self.fmap1_nchw,
                self.fmap2_nchw,
                coords,
                num_levels=self.num_levels,
                radius=self.radius,
            )
            return corr.reshape(B, -1, H, W) / math.sqrt(self.dim)

        _, localcorr_triton_fused = _get_triton_kernels()
        corr = localcorr_triton_fused.apply(
            self.fmap1,
            self.fmap2_packed,
            coords,
            self.fmap2_offsets,
            self.fmap2_shapes,
            self.level_scales,
            self.level_offsets,
            self.level_shapes,
            self.radius,
        )
        return corr.reshape(B, -1, H, W) / math.sqrt(self.dim)


class LocalCorrCudaPyramidBlock:
    """Local correlation with all pyramid levels evaluated in one CUDA kernel launch.

    Eager execution calls the AOT-compiled fused CUDA kernel from
    localcorr/cuda_pyramid/ (built with python -m localcorr.build_native
    cuda) through ctypes, on torch's current CUDA stream. The kernel
    provides a fused backward as well, so this block is trainable
    (gradients for the feature maps; the sampling coordinates stay
    detached, like the Triton backward). During ONNX export, the same
    computation is emitted as one LocalCorrCudaPyramid TensorRT plugin
    node per call, computed by that same CUDA kernel.

    With onnx_generic=True, ONNX export emits the standard-ops
    (matmul + grid_sample) correlation instead of the plugin node, without
    changing the global backend selected with set_onnx_backend().
    """

    def __init__(
        self, fmap1, fmap2, num_levels=4, radius=4, onnx_generic: bool = False
    ):
        self.num_levels = num_levels
        self.radius = radius
        self.dim = fmap1.shape[1]
        self.onnx_generic = onnx_generic

        if self.use_generic_export_ops():
            # Standard-ops ONNX export (e.g. for OpenVINO): keep the
            # feature maps in NCHW; the correlation is computed with
            # grid_sample ops when the block is called.
            self.fmap1_nchw = fmap1
            self.fmap2_nchw = fmap2
            return

        self.fmap1 = fmap1.permute(0, 2, 3, 1).contiguous()
        fmap2_pyramid = [fmap2]
        for _ in range(1, num_levels):
            fmap2_pyramid.append(F.avg_pool2d(fmap2_pyramid[-1], 2, stride=2))
        self.fmap2_pyramid = [
            fmap.permute(0, 2, 3, 1).contiguous() for fmap in fmap2_pyramid
        ]
        # Batch-major packed pyramid [B, C*sum_i h_i*w_i] (each batch row
        # holds all its levels contiguously), the layout expected by the
        # CUDA kernel (and by the LocalCorrCudaPyramid TensorRT plugin).
        self.fmap2_packed = torch.cat(
            [fmap.reshape(fmap.shape[0], -1) for fmap in self.fmap2_pyramid],
            dim=1,
        )

    def use_generic_export_ops(self) -> bool:
        """Whether the ONNX export should use standard ops instead of the plugin node."""
        return use_onnx_generic_ops() or (
            self.onnx_generic and torch.onnx.is_in_onnx_export()
        )

    def __call__(self, coords):
        # coords: [B, 2, H, W] (x, y), level-0 scale.
        B, _, H, W = coords.shape

        if self.use_generic_export_ops():
            corr = localcorr_pyramid_onnx_generic(
                self.fmap1_nchw,
                self.fmap2_nchw,
                coords,
                num_levels=self.num_levels,
                radius=self.radius,
            )
        elif torch.onnx.is_in_onnx_export():
            corr = LocalCorrCudaPyramidOnnx.apply(
                self.fmap1, self.fmap2_packed, coords, self.radius, self.num_levels
            )
        else:
            corr = _localcorr_cuda_pyramid_forward(
                self.fmap1, self.fmap2_packed, coords, self.radius, self.num_levels
            )
        return corr.reshape(B, -1, H, W) / math.sqrt(self.dim)


class LocalCorrCudaPyramidFunction(torch.autograd.Function):
    """Autograd wrapper of the fused CUDA pyramid kernel.

    forward() launches the fused correlation kernel through ctypes on
    torch's current CUDA stream; backward() launches its backward kernel
    (gradients for fmap1 and fmap2_packed; zeros for coords, matching
    the Triton/SYCL backward and RAFT's detached coordinates). Inputs
    must be contiguous CUDA tensors in the layout documented by the
    kernel header (fmap1 NHWC, fmap2_packed batch-major, coords planar
    x/y). For fp16 inputs the scattered fmap2 gradient is accumulated
    in an fp32 shadow buffer and cast to half afterwards (fp32
    accumulation, no fp16 atomics).
    """

    @staticmethod
    def forward(ctx, fmap1, fmap2_packed, coords, radius, num_levels, dilation,
                ymajor):
        lib = get_cuda_pyramid_library()
        coords = coords.contiguous()
        batch_size, height_1, width_1, channels = fmap1.shape
        rd = 2 * radius + 1
        corr = fmap1.new_empty(
            (batch_size, num_levels, rd * rd, height_1, width_1)
        )
        # Launched on torch's current stream, so it is ordered with the
        # surrounding torch operations without extra synchronization.
        stream = torch.cuda.current_stream(fmap1.device).cuda_stream
        ok = lib.launchLocalCorrCudaPyramidForward(
            fmap1.data_ptr(),
            fmap2_packed.data_ptr(),
            coords.data_ptr(),
            corr.data_ptr(),
            batch_size,
            height_1,
            width_1,
            channels,
            radius,
            num_levels,
            dilation,
            fmap1.dtype == torch.float16,  # isHalf
            ymajor,
            stream,
        )
        if not ok:
            raise RuntimeError(
                "localcorr CUDA kernel launch failed (e.g. shared-memory "
                "overflow for this radius/channel count)."
            )
        ctx.save_for_backward(fmap1, fmap2_packed, coords)
        ctx.kernel_params = (
            int(radius), int(num_levels), int(dilation), bool(ymajor))
        return corr

    @staticmethod
    def backward(ctx, corr_grad):
        fmap1, fmap2_packed, coords = ctx.saved_tensors
        radius, num_levels, dilation, ymajor = ctx.kernel_params
        lib = get_cuda_pyramid_library()
        if getattr(lib, "launchLocalCorrCudaPyramidBackward", None) is None:
            raise RuntimeError(
                'corr_mode="localcorr_cuda" needs a backward pass, but this '
                "build of the CUDA kernel library does not provide one "
                "(the library predates it). Rebuild it with: "
                "python -m localcorr.build_native cuda"
            )
        corr_grad = corr_grad.contiguous()
        batch_size, height_1, width_1, channels = fmap1.shape
        stream = torch.cuda.current_stream(fmap1.device).cuda_stream
        fmap1_grad = torch.empty_like(fmap1)
        if fmap1.dtype == torch.float16:
            # fp32 shadow accumulator, cast to half afterwards.
            f2g32 = torch.zeros(
                fmap2_packed.shape,
                dtype=torch.float32,
                device=fmap2_packed.device,
            )
            ok = lib.launchLocalCorrCudaPyramidBackward(
                fmap1.data_ptr(),
                fmap2_packed.data_ptr(),
                coords.data_ptr(),
                corr_grad.data_ptr(),
                fmap1_grad.data_ptr(),
                f2g32.data_ptr(),
                batch_size,
                height_1,
                width_1,
                channels,
                radius,
                num_levels,
                dilation,
                True,  # isHalf
                ymajor,
                stream,
            )
            fmap2_grad = f2g32.to(torch.float16)
        else:
            fmap2_grad = torch.zeros_like(fmap2_packed)
            ok = lib.launchLocalCorrCudaPyramidBackward(
                fmap1.data_ptr(),
                fmap2_packed.data_ptr(),
                coords.data_ptr(),
                corr_grad.data_ptr(),
                fmap1_grad.data_ptr(),
                fmap2_grad.data_ptr(),
                batch_size,
                height_1,
                width_1,
                channels,
                radius,
                num_levels,
                dilation,
                False,  # isHalf
                ymajor,
                stream,
            )
        if not ok:
            raise RuntimeError(
                "localcorr CUDA backward kernel launch failed (e.g. "
                "shared-memory overflow for this channel count)."
            )
        # RAFT detaches the sampling coordinates; align with the Triton
        # and SYCL backward, which also return zeros here.
        coords_grad = torch.zeros_like(coords)
        return fmap1_grad, fmap2_grad, coords_grad, None, None, None, None


def _localcorr_cuda_pyramid_forward(
    fmap1, fmap2_packed, coords, radius, num_levels, dilation=1, ymajor=False
):
    """Eager forward/backward of the fused CUDA pyramid kernel (CUDA tensors).

    Runs the kernel through LocalCorrCudaPyramidFunction, so gradients
    with respect to fmap1 and fmap2_packed are computed by the fused
    backward kernel when autograd requires them. Raises clear errors
    when the tensors are not on a CUDA device, when the CUDA kernel
    library has not been built, or when the library is too old to
    provide the backward pass.
    """
    if not fmap1.is_cuda:
        raise RuntimeError(
            'corr_mode="localcorr_cuda" requires the feature maps to be on '
            f"a CUDA device, but they are on {fmap1.device}. Use "
            "localcorr_triton on other devices."
        )
    lib = get_cuda_pyramid_library()
    if lib is None:
        raise ImportError(
            'corr_mode="localcorr_cuda" requires the CUDA kernel library, '
            "but it has not been built (or could not be loaded). Build it with: "
            "python -m localcorr.build_native cuda"
        )
    if (
        torch.is_grad_enabled()
        and (fmap1.requires_grad or fmap2_packed.requires_grad or coords.requires_grad)
        and getattr(lib, "launchLocalCorrCudaPyramidBackward", None) is None
    ):
        raise RuntimeError(
            'corr_mode="localcorr_cuda" is inference-only with this build of '
            "the CUDA kernel library (no backward pass). Rebuild it with: "
            "python -m localcorr.build_native cuda"
        )
    if fmap1.dtype not in (torch.float32, torch.float16):
        raise RuntimeError(
            'corr_mode="localcorr_cuda" requires float32 or float16 '
            f"feature maps, but got {fmap1.dtype}."
        )
    if coords.dtype != fmap1.dtype:
        raise RuntimeError(
            'corr_mode="localcorr_cuda" requires the coords dtype to match '
            f"the feature maps dtype, but got {coords.dtype} and {fmap1.dtype}."
        )
    return LocalCorrCudaPyramidFunction.apply(
        fmap1, fmap2_packed, coords, int(radius), int(num_levels),
        int(dilation), bool(ymajor)
    )


def _cuda_pyramid_auto_available(input1, input2=None):
    """Whether corr_mode=localcorr_auto can use the AOT CUDA kernel.

    localcorr_auto prefers the localcorr_cuda engine on NVIDIA GPUs and
    falls back to the Triton kernel when this returns False: tensors not
    on a CUDA device, the CUDA kernel library not built (or not
    loadable), a dtype the kernel does not support, or a library too old
    to provide the backward pass while the inputs may need gradients
    (the Triton kernel always provides one).
    """
    if not input1.is_cuda:
        return False
    lib = get_cuda_pyramid_library()
    if lib is None:
        return False
    if input1.dtype not in (torch.float32, torch.float16):
        return False
    if (
        input1.requires_grad or (input2 is not None and input2.requires_grad)
    ) and getattr(lib, "launchLocalCorrCudaPyramidBackward", None) is None:
        return False
    return True


def _spatial_identity_coords(b, h, w, dtype, device):
    """Identity coordinate grid [B, 2, H, W] (x in channel 0), as expected by the kernels."""
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=0)[None].repeat(b, 1, 1, 1).contiguous()


class _LocalCorrNativeSpatialSampler:
    """Common logic of the AOT-kernel PWCNet-style spatial samplers.

    Subclasses provide the eager engine; ONNX export emits the
    LocalCorrCudaPyramid plugin node with levels=1, the y-major
    displacement order of the spatial correlation samplers and the given
    patch dilation, and the generic backend exports standard ops.
    """

    def __init__(
        self,
        kernel_size=1,
        patch_size=9,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=1,
    ):
        assert kernel_size == 1, "Only kernel_size=1 is supported."
        assert stride == 1, "Only stride=1 is supported."
        assert padding == 0, "Only padding=0 is supported."
        assert dilation == 1, "Only dilation=1 is supported."
        self.md = patch_size // 2
        self.patch_size = patch_size
        self.dilation_patch = dilation_patch

    def _nhwc_inputs(self, input1, input2):
        b = input1.shape[0]
        fmap1 = input1.permute(0, 2, 3, 1).contiguous()
        fmap2_packed = input2.permute(0, 2, 3, 1).contiguous().view(b, -1)
        coords = _spatial_identity_coords(
            b, input1.shape[2], input1.shape[3], input1.dtype, input1.device
        )
        return fmap1, fmap2_packed, coords

    def __call__(self, input1, input2):
        b, _, h, w = input1.shape
        if use_onnx_generic_ops():
            return spatial_corr_onnx_generic(input1, input2, self.md)
        if torch.onnx.is_in_onnx_export():
            fmap1, fmap2_packed, coords = self._nhwc_inputs(input1, input2)
            corr = LocalCorrCudaPyramidSpatialOnnx.apply(
                fmap1, fmap2_packed, coords, self.md, self.dilation_patch, None
            )
            return corr.view(b, self.patch_size, self.patch_size, h, w)
        return self._eager(input1, input2, *self._nhwc_inputs(input1, input2))


class LocalCorrCudaSpatialSampler(_LocalCorrNativeSpatialSampler):
    """corr_mode=localcorr_cuda, PWCNet style: AOT CUDA kernel, trainable.

    Eager execution computes the single-level dilated spatial correlation
    with the fused CUDA kernel from localcorr/cuda_pyramid/ (built with
    python -m localcorr.build_native cuda), called through ctypes on
    torch's current CUDA stream, with the fused backward kernel for
    training. ONNX export emits the LocalCorrCudaPyramid plugin node
    computed by the same kernel.
    """

    def _eager(self, input1, input2, fmap1, fmap2_packed, coords):
        # ymajor=True: SpatialCorrelationSampler channel order
        # (channel = dy_idx * patch_size + dx_idx).
        corr = _localcorr_cuda_pyramid_forward(
            fmap1, fmap2_packed, coords, self.md, 1, self.dilation_patch, True
        )
        return corr.view(
            input1.shape[0],
            self.patch_size,
            self.patch_size,
            input1.shape[2],
            input1.shape[3],
        )


class LocalCorrSyclSpatialSampler(_LocalCorrNativeSpatialSampler):
    """corr_mode=localcorr_sycl, PWCNet style: SYCL kernel on torch.xpu.

    Eager execution computes the correlation with the fused SYCL kernel
    of the localcorr/xpu extension (built with
    python -m localcorr.build_native xpu) on torch.xpu (Intel GPU)
    devices, with the backward pass for training when the extension
    provides it (older builds raise a clear inference-only error).
    ONNX export emits the LocalCorrCudaPyramid node, resolved by the
    OpenVINO SYCL custom-op extension (or the TensorRT plugin).
    """

    def _eager(self, input1, input2, fmap1, fmap2_packed, coords):
        if not input1.is_xpu:
            raise RuntimeError(
                'corr_mode="localcorr_sycl" requires the inputs to be on a '
                f"torch.xpu (Intel GPU) device for eager execution, but they "
                f"are on {input1.device}. Use localcorr_cuda on NVIDIA GPUs "
                "or localcorr_triton elsewhere; this mode can also be run "
                "through OpenVINO after ONNX export."
            )
        ext = get_sycl_xpu_extension()
        if ext is None:
            raise ImportError(
                'corr_mode="localcorr_sycl" requires the localcorr SYCL XPU '
                "extension for eager execution, but it could not be loaded. "
                "Build it with: python -m localcorr.build_native xpu"
            )
        if (
            torch.is_grad_enabled()
            and (input1.requires_grad or input2.requires_grad)
            and not hasattr(ext, "backward")
        ):
            raise RuntimeError(
                'corr_mode="localcorr_sycl" is inference-only with this '
                "build of the localcorr SYCL extension (no backward pass). "
                "Rebuild it with: python -m localcorr.build_native xpu"
            )
        # ymajor=True: SpatialCorrelationSampler channel order.
        corr = LocalCorrSyclFunction.apply(
            fmap1, fmap2_packed, coords, self.md, 1, self.dilation_patch, True
        )
        return corr.view(
            input1.shape[0],
            self.patch_size,
            self.patch_size,
            input1.shape[2],
            input1.shape[3],
        )


class LocalCorrOnnxSpatialSampler:
    """corr_mode=localcorr_onnx, PWCNet style: plugin-free export.

    Eager execution is the Triton spatial kernel (with backward, so
    training works). During ONNX export the correlation is always emitted
    with standard ops, regardless of the backend selected with
    set_onnx_backend().
    """

    def __init__(
        self,
        kernel_size=1,
        patch_size=9,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=1,
    ):
        assert kernel_size == 1, "Only kernel_size=1 is supported."
        assert stride == 1, "Only stride=1 is supported."
        assert padding == 0, "Only padding=0 is supported."
        assert dilation == 1, "Only dilation=1 is supported."
        self.md = patch_size // 2
        self.patch_size = patch_size
        self.dilation_patch = dilation_patch

    def __call__(self, input1, input2):
        if torch.onnx.is_in_onnx_export():
            if self.dilation_patch != 1:
                raise RuntimeError(
                    "dilation_patch != 1 cannot be exported with standard ops."
                )
            return spatial_corr_onnx_generic(input1, input2, self.md)
        triton_fn = _get_triton_spatial_function()
        return triton_fn.apply(input1, input2, self.md, self.dilation_patch)


def get_localcorr_sampler(
    patch_size: int = 9,
    dilation_patch: int = 1,
    corr_mode: str = "localcorr_triton",
    xpu_backend: str = "sycl",
):
    """Create a PWCNet-style spatial correlation sampler for the given mode.

    Modes: localcorr_triton (Triton kernel, eager forward + backward),
    localcorr_onnx (Triton kernel eagerly; ONNX export always emits
    standard ops), localcorr_cuda (eager execution with the AOT CUDA
    kernel, with the fused backward for training; ONNX export as the
    LocalCorrCudaPyramid plugin node) and localcorr_sycl (the Intel
    counterpart of localcorr_cuda: eager execution on torch.xpu with the
    SYCL kernel from localcorr/xpu, with the backward pass for training;
    ONNX export as the node resolved by the OpenVINO SYCL custom-op
    extension). localcorr_auto returns
    the device-dispatching LocalCorrCudaPyramidSampler: the AOT CUDA
    kernel on CUDA tensors when its library is built (trainable) and the
    SYCL kernel on torch.xpu when the
    xpu extension is built (Triton fallback otherwise); xpu_backend is
    only used by that mode. The samplers are equivalent to
    SpatialCorrelationSampler(kernel_size=1, patch_size, padding=0,
    dilation_patch=dilation_patch).
    """
    mode = corr_mode.lower()
    kwargs = dict(
        kernel_size=1,
        patch_size=patch_size,
        padding=0,
        dilation_patch=dilation_patch,
    )
    if mode == "localcorr_triton":
        try:
            from .triton import TritonSpatialCorrelationSampler
        except (ImportError, OSError) as exc:
            raise ImportError(
                "corr_mode='localcorr_triton' requires the triton package, "
                "which could not be imported. Install it with: pip install "
                "triton"
            ) from exc
        return TritonSpatialCorrelationSampler(**kwargs)
    if mode == "localcorr_auto":
        return LocalCorrCudaPyramidSampler(xpu_backend=xpu_backend, **kwargs)
    if mode == "localcorr_onnx":
        return LocalCorrOnnxSpatialSampler(**kwargs)
    if mode == "localcorr_cuda":
        return LocalCorrCudaSpatialSampler(**kwargs)
    if mode == "localcorr_sycl":
        return LocalCorrSyclSpatialSampler(**kwargs)
    raise ValueError(f"Unknown local corr_mode: {corr_mode}")


class LocalCorrSyclBlock(LocalCorrCudaPyramidBlock):
    """localcorr_sycl: the Intel counterpart of localcorr_cuda.

    Eager execution calls the fused SYCL kernel of the localcorr/xpu
    extension (built with python -m localcorr.build_native xpu): all
    pyramid levels in one kernel launch on torch's current XPU stream,
    for tensors on torch.xpu (Intel GPU) devices. The extension also
    provides the backward pass, so this mode is trainable (older
    extension builds without backward raise a clear inference-only
    error). During ONNX export, the LocalCorrCudaPyramid node is
    emitted, resolved by the TensorRT plugin or by the
    localcorr/sycl_pyramid OpenVINO custom-op extension.
    """

    def __call__(self, coords):
        # coords: [B, 2, H, W] (x, y), level-0 scale.
        B, _, H, W = coords.shape

        if self.use_generic_export_ops():
            corr = localcorr_pyramid_onnx_generic(
                self.fmap1_nchw,
                self.fmap2_nchw,
                coords,
                num_levels=self.num_levels,
                radius=self.radius,
            )
        elif torch.onnx.is_in_onnx_export():
            corr = LocalCorrCudaPyramidOnnx.apply(
                self.fmap1, self.fmap2_packed, coords, self.radius, self.num_levels
            )
        else:
            corr = _localcorr_sycl_forward(
                self.fmap1,
                self.fmap2_packed,
                coords.contiguous(),
                self.radius,
                self.num_levels,
            )
        return corr.reshape(B, -1, H, W) / math.sqrt(self.dim)


def _localcorr_sycl_forward(fmap1, fmap2_packed, coords, radius, num_levels):
    """Eager fused SYCL pyramid correlation (torch.xpu), with backward.

    Raises clear errors when the tensors are not on an XPU device or
    when the localcorr/xpu extension has not been built. Gradients are
    supported when the extension provides the backward kernel; older
    builds fall back to a clear inference-only error instead.
    """
    if not fmap1.is_xpu:
        raise RuntimeError(
            'corr_mode="localcorr_sycl" requires the feature maps to be on a '
            f"torch.xpu (Intel GPU) device for eager execution, but they are on "
            f"{fmap1.device}. Use localcorr_cuda on NVIDIA GPUs or "
            "localcorr_triton elsewhere; this mode can also be run "
            "through OpenVINO after ONNX export."
        )
    ext = get_sycl_xpu_extension()
    if ext is None:
        raise ImportError(
            'corr_mode="localcorr_sycl" requires the localcorr SYCL XPU '
            "extension for eager execution, but it could not be loaded. Build "
            "it with: python -m localcorr.build_native xpu"
        )
    if torch.is_grad_enabled() and (
        fmap1.requires_grad or fmap2_packed.requires_grad or coords.requires_grad
    ) and not hasattr(ext, "backward"):
        raise RuntimeError(
            'corr_mode="localcorr_sycl" is inference-only with this build of '
            "the localcorr SYCL extension (no backward pass). Rebuild it "
            "with: python -m localcorr.build_native xpu"
        )
    # ymajor=False: RAFT x-major displacement order (d = iy + rd * ix).
    return LocalCorrSyclFunction.apply(
        fmap1, fmap2_packed, coords, radius, num_levels, 1, False
    )


class LocalCorrOnnxBlock(LocalCorrTritonFusedBlock):
    """localcorr_onnx: plugin-free correlation.

    Eager execution is identical to LocalCorrTritonFusedBlock (the fused
    Triton kernel, with backward for training). During ONNX export, the
    correlation is always emitted with standard ops (matmul +
    grid_sample), regardless of the backend selected with
    set_onnx_backend(), producing graphs that any ONNX runtime (and
    plugin-free TensorRT engines) can convert.
    """

    onnx_generic = True

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__(fmap1, fmap2, num_levels=num_levels, radius=radius)
        # Always keep the NCHW feature maps, so the forced generic export
        # works even when the global backend was "tensorrt" when this
        # block was created.
        self.fmap1_nchw = fmap1
        self.fmap2_nchw = fmap2

    def __call__(self, coords):
        if torch.onnx.is_in_onnx_export():
            B, _, H, W = coords.shape
            corr = localcorr_pyramid_onnx_generic(
                self.fmap1_nchw,
                self.fmap2_nchw,
                coords,
                num_levels=self.num_levels,
                radius=self.radius,
            )
            return corr.reshape(B, -1, H, W) / math.sqrt(self.dim)
        return super().__call__(coords)


class LocalCorrCudaPyramidSpatialOnnx(torch.autograd.Function):
    """ONNX-export wrapper of the single-level dilated spatial correlation.

    forward() returns an empty tensor with the correct shape so tracing
    can continue; the output is produced by symbolic(), which records a
    "custom_ops::LocalCorrCudaPyramid" node with levels=1 and the given
    dilation. Eager execution goes through the Triton kernels instead.
    """

    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, radius, dilation, normalize):
        # fmap1/fmap2: [B, H, W, C] NHWC; coords: [B, 2, H, W] planar
        # (x in channel 0). The output is squeezed to [B, rd*rd, H, W] to
        # match the eager shape of LocalCorrCudaPyramidSpatialBlock.
        radius = int(radius)
        dilation = int(dilation)
        batch_size, height_1, width_1, _ = fmap1.shape
        rd = 2 * radius + 1
        return fmap1.new_empty((batch_size, rd * rd, height_1, width_1))

    @staticmethod
    def symbolic(g, fmap1, fmap2, coords, radius, dilation, normalize):
        # Single level (fmap2 has the same resolution as fmap1). The
        # dilated window (dilation = dilation_patch of the spatial
        # correlation samplers) is computed by the plugin kernel.
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
            dilation_i=int(dilation),
            # The eager block's channel order is d = ix + rd*iy (the
            # SpatialCorrelationSampler order), which the plugin produces
            # with ymajor=1.
            ymajor_i=1,
        )
        coords_sizes = coords.type().sizes() if coords.type() is not None else None
        if fmap1_sizes is not None and coords_sizes is not None:
            rd = 2 * int(radius) + 1
            # The plugin produces [B, 1, rd*rd, H, W]; squeeze the levels
            # dim so the graph type matches the eager [B, rd*rd, H, W].
            output_sizes = [
                coords_sizes[0],
                1,
                rd * rd,
                fmap1_sizes[1],
                fmap1_sizes[2],
            ]
            node.setType(fmap1.type().with_sizes(output_sizes))
            node = g.op(
                "Reshape",
                node,
                g.op(
                    "Constant",
                    value_t=torch.tensor(
                        [coords_sizes[0], rd * rd, fmap1_sizes[1], fmap1_sizes[2]],
                        dtype=torch.int64,
                    ),
                ),
            )
        # NOTE: the normalization (/sqrt(dim) or /dim) is applied by the
        # block's Python code AFTER this wrapper returns, and the legacy
        # tracer records that division as its own ONNX Div node on this
        # node's output. Do NOT emit it here (it would be applied twice).
        return node


class LocalCorrCudaPyramidSpatialBlock:
    """Single-level dilated spatial correlation (FlowNetC style).

    Equivalent to a spatial correlation sampler with kernel_size=1,
    patch_size=2*radius+1 and dilation_patch=dilation around the identity
    coordinate grid: corr[b, iy, ix, h, w] = sum_c fmap1[b,h,w,c] *
    fmap2[b, h+dilation*(iy-radius), w+dilation*(ix-radius)] (zero padded).

    Eager forward and backward are computed with pure PyTorch (shifted
    slicing; no custom kernel needed). During ONNX export, the same
    computation is emitted as one LocalCorrCudaPyramid plugin node with
    levels=1 and the given dilation, computed by the CUDA kernel from
    localcorr/cuda_pyramid/.
    """

    def __init__(self, fmap1, fmap2, radius=10, dilation=2, normalize="sqrt"):
        self.radius = radius
        self.dilation = dilation
        self.dim = fmap1.shape[1]
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        # "sqrt": divide by sqrt(C) (RAFT-style); "dim": divide by C
        # (FlowNetC correlate-style); None: no normalization.
        self.normalize = normalize
        self._coords = None

    def _normalize_out(self, corr):
        if self.normalize == "sqrt":
            return corr / math.sqrt(self.dim)
        if self.normalize == "dim":
            return corr / self.dim
        return corr

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

    def __call__(self, coords=None):
        # coords is accepted for API compatibility but ignored: this block
        # always correlates around the identity grid.
        if torch.onnx.is_in_onnx_export():
            fmap1 = self.fmap1.permute(0, 2, 3, 1).contiguous()
            fmap2 = self.fmap2.permute(0, 2, 3, 1).contiguous()
            b, c, h, w = self.fmap1.shape
            grid = self._identity_coords(b, h, w, self.fmap1.dtype, self.fmap1.device)
            corr = LocalCorrCudaPyramidSpatialOnnx.apply(
                fmap1, fmap2, grid, self.radius, self.dilation, self.normalize
            )
        else:
            b, c, h, w = self.fmap1.shape
            rd = 2 * self.radius + 1
            corr = []
            # Same sign convention as spatial_correlation_sample /
            # iter_spatial_correlation_sample:
            # corr[b, iy, ix, h, w] = sum_c x1[b,c,h,w] *
            #     x2[b, c, h + dil*(iy-r), w + dil*(ix-r)]  (zero padded).
            for iy in range(rd):
                for ix in range(rd):
                    yy = self.dilation * (iy - self.radius)
                    xx = self.dilation * (ix - self.radius)
                    # x2 sampled at (h + yy, w + xx): x2_lo = max(0, -yy) etc.
                    s_h_lo, s_h_hi = max(0, -yy), min(h, h - yy)
                    s_w_lo, s_w_hi = max(0, -xx), min(w, w - xx)
                    val = torch.zeros(
                        b, h, w, dtype=self.fmap1.dtype, device=self.fmap1.device
                    )
                    if s_h_lo < s_h_hi and s_w_lo < s_w_hi:
                        val[:, s_h_lo:s_h_hi, s_w_lo:s_w_hi] = (
                            self.fmap1[:, :, s_h_lo:s_h_hi, s_w_lo:s_w_hi]
                            * self.fmap2[
                                :,
                                :,
                                s_h_lo + yy : s_h_hi + yy,
                                s_w_lo + xx : s_w_hi + xx,
                            ]
                        ).sum(dim=1)
                    corr.append(val)
            corr = torch.stack(corr, dim=1).reshape(b, rd * rd, h, w)

        return self._normalize_out(corr)


def _make_localcorr_auto_block(
    fmap1, fmap2, num_levels=4, radius=4, xpu_backend="sycl"
):
    """Construct the corr_mode=localcorr_auto block for the feature maps.

    The device of the feature maps (known at construction in the
    RAFT-style API) selects the eager engine: on NVIDIA GPUs the fused
    AOT CUDA kernel (localcorr_cuda) when its library has been built
    (with backward, training support), with the fused Triton kernel as
    fallback; and, on torch.xpu (Intel GPUs), the fused SYCL kernel when
    the localcorr/xpu extension has been built (with backward, training
    support), with the Triton kernel (triton-xpu) as fallback;
    xpu_backend="triton" forces the fallback. During ONNX
    export with the default "tensorrt" backend, the
    LocalCorrCudaPyramid plugin node is emitted device-independently
    (the fused Triton kernel itself cannot be exported); with the "onnx"
    backend, the standard ops are emitted as for every other mode.
    """
    if torch.onnx.is_in_onnx_export() and not use_onnx_generic_ops():
        # Plugin-node export: LocalCorrCudaPyramidBlock emits the node
        # from tensors on any device (the CUDA kernel is only touched by
        # eager execution, which never runs on this path).
        return LocalCorrCudaPyramidBlock(
            fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels
        )
    if _cuda_pyramid_auto_available(fmap1, fmap2):
        # NVIDIA GPUs: prefer the fused AOT CUDA kernel (the
        # localcorr_cuda engine); the Triton kernel is the fallback.
        return LocalCorrCudaPyramidBlock(
            fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels
        )
    if fmap1.is_xpu and xpu_backend == "sycl" and get_sycl_xpu_extension() is not None:
        return LocalCorrSyclBlock(
            fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels
        )
    return LocalCorrTritonFusedBlock(
        fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels
    )


def get_localcorr_block(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    num_levels: int = 4,
    radius: int = 4,
    corr_mode: str = "localcorr_triton",
    xpu_backend: str = "sycl",
):
    """Create a local correlation block for the given mode.

    Modes: localcorr_triton (Triton kernels, eager forward + backward;
    all pyramid levels in one kernel launch), localcorr_cuda (eager
    execution with the AOT CUDA kernel, with the fused backward for
    training, ONNX export as the LocalCorrCudaPyramid TensorRT plugin
    node), localcorr_onnx (eager
    fused Triton kernel; ONNX export always emits standard ops instead
    of the plugin node) and localcorr_sycl (the Intel counterpart of
    localcorr_cuda: eager execution on torch.xpu with the fused SYCL
    kernel from localcorr/xpu, with the backward pass for training; ONNX
    export as the node resolved by the OpenVINO SYCL custom-op
    extension). localcorr_auto
    picks the engine automatically from the device of the feature maps
    (the AOT CUDA kernel on NVIDIA GPUs when its library is built, the
    SYCL kernel on torch.xpu when the xpu extension is built, Triton
    fallback otherwise; xpu_backend is only
    used by that mode). The allpairs
    mode is not a local correlation and is handled by the callers (it
    needs no kernels from this package).
    """
    mode = corr_mode.lower()
    if mode == "localcorr_triton":
        return LocalCorrTritonFusedBlock(
            fmap1=fmap1, fmap2=fmap2, radius=radius, num_levels=num_levels
        )
    if mode == "localcorr_auto":
        return _make_localcorr_auto_block(
            fmap1,
            fmap2,
            num_levels=num_levels,
            radius=radius,
            xpu_backend=xpu_backend,
        )
    if mode == "localcorr_cuda":
        return LocalCorrCudaPyramidBlock(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=radius,
            num_levels=num_levels,
        )
    if mode == "localcorr_onnx":
        return LocalCorrOnnxBlock(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=radius,
            num_levels=num_levels,
        )
    if mode == "localcorr_sycl":
        return LocalCorrSyclBlock(
            fmap1=fmap1,
            fmap2=fmap2,
            radius=radius,
            num_levels=num_levels,
        )
    raise ValueError(f"Unknown local corr_mode: {corr_mode}")
