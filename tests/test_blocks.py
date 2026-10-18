"""Tests for localcorr.blocks (RAFT-style correlation blocks)."""

import math
import sys

import pytest
import torch

import localcorr
import localcorr.blocks as blocks_mod
from localcorr.blocks import (
    LocalCorrCudaPyramidBlock,
    LocalCorrCudaPyramidOnnx,
    LocalCorrCudaPyramidSpatialBlock,
    LocalCorrCudaPyramidSpatialOnnx,
    LocalCorrOnnxBlock,
    LocalCorrSyclBlock,
    LocalCorrTritonBlock,
    LocalCorrTritonFusedBlock,
    get_localcorr_block,
)

from _reference import (
    make_coords,
    make_fmaps,
    pyramid_localcorr_reference,
    shifted_spatial_reference,
    spatial_corr_reference,
)

B, C, H, W = 2, 8, 16, 24
LEVELS, RADIUS = 3, 2


@pytest.fixture
def fmaps():
    return make_fmaps(batch=B, channels=C, height=H, width=W, seed=20)


@pytest.fixture
def coords():
    return make_coords(batch=B, height=H, width=W, max_disp=3.0, seed=21)


class TestGetLocalCorrBlock:
    @pytest.mark.parametrize(
        "mode,cls",
        [
            ("localcorr_triton", LocalCorrTritonFusedBlock),
            ("localcorr_cuda", LocalCorrCudaPyramidBlock),
            ("localcorr_onnx", LocalCorrOnnxBlock),
            ("localcorr_sycl", LocalCorrSyclBlock),
            # On CPU tensors the auto mode falls through to the Triton
            # block (construction only; running it needs a GPU).
            ("localcorr_auto", LocalCorrTritonFusedBlock),
        ],
    )
    def test_dispatch(self, fmaps, mode, cls):
        block = get_localcorr_block(
            *fmaps, num_levels=LEVELS, radius=RADIUS, corr_mode=mode
        )
        assert type(block) is cls
        assert block.num_levels == LEVELS
        assert block.radius == RADIUS
        assert block.dim == C

    def test_auto_mode_dispatches_plugin_export(self, fmaps, monkeypatch):
        # During ONNX export with the default backend, the auto mode
        # must emit the plugin node, which the Triton block cannot do.
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        block = get_localcorr_block(*fmaps, corr_mode="localcorr_auto")
        assert type(block) is LocalCorrCudaPyramidBlock

    def test_auto_mode_respects_generic_backend(self, fmaps, monkeypatch):
        # With the "onnx" backend selected, the auto mode keeps the
        # generic-export layout (NCHW feature maps kept, no packing).
        localcorr.set_onnx_backend("onnx")
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        try:
            block = get_localcorr_block(*fmaps, corr_mode="localcorr_auto")
            assert type(block) is LocalCorrTritonFusedBlock
            assert block.fmap1_nchw is not None  # generic-export layout
        finally:
            localcorr.set_onnx_backend("tensorrt")

    def test_mode_is_case_insensitive(self, fmaps):
        block = get_localcorr_block(*fmaps, corr_mode="LOCALCORR_TRITON")
        assert type(block) is LocalCorrTritonFusedBlock

    def test_unknown_mode_raises(self, fmaps):
        with pytest.raises(ValueError, match="Unknown local corr_mode"):
            get_localcorr_block(*fmaps, corr_mode="allpairs_local")

    def test_onnx_mode_sets_generic_flag(self, fmaps):
        block = get_localcorr_block(*fmaps, corr_mode="localcorr_onnx")
        assert block.onnx_generic is True
        block = get_localcorr_block(*fmaps, corr_mode="localcorr_cuda")
        assert block.onnx_generic is False
        block = get_localcorr_block(*fmaps, corr_mode="localcorr_sycl")
        assert block.onnx_generic is False


class TestAutoModeCudaPreference:
    def test_cuda_availability_false_on_cpu_tensors(self, fmaps):
        f1, f2 = fmaps
        assert blocks_mod._cuda_pyramid_auto_available(f1, f2) is False
        assert blocks_mod._cuda_pyramid_auto_available(f1) is False

    def test_prefers_cuda_block_when_available(self, fmaps, monkeypatch):
        # localcorr_auto must pick the localcorr_cuda engine whenever the
        # AOT CUDA kernel is usable for the tensors (simulated here; the
        # real check needs a CUDA device with the library built).
        monkeypatch.setattr(
            blocks_mod, "_cuda_pyramid_auto_available", lambda f1, f2=None: True
        )
        block = get_localcorr_block(
            *fmaps, num_levels=LEVELS, radius=RADIUS, corr_mode="localcorr_auto"
        )
        assert type(block) is LocalCorrCudaPyramidBlock

    def test_falls_back_to_triton_when_unavailable(self, fmaps, monkeypatch):
        monkeypatch.setattr(
            blocks_mod, "_cuda_pyramid_auto_available", lambda f1, f2=None: False
        )
        block = get_localcorr_block(
            *fmaps, num_levels=LEVELS, radius=RADIUS, corr_mode="localcorr_auto"
        )
        assert type(block) is LocalCorrTritonFusedBlock

    def test_availability_requires_cuda_dtype_and_library(self, monkeypatch):
        class FakeTensor:
            def __init__(self, is_cuda, dtype, requires_grad=False):
                self.is_cuda = is_cuda
                self.dtype = dtype
                self.requires_grad = requires_grad

        fake_cuda = FakeTensor(True, torch.float32)
        fake_cpu = FakeTensor(False, torch.float32)
        fake_bf16 = FakeTensor(True, torch.bfloat16)

        class FakeLib:
            launchLocalCorrCudaPyramidBackward = None

        monkeypatch.setattr(
            blocks_mod, "get_cuda_pyramid_library", lambda: None
        )
        assert blocks_mod._cuda_pyramid_auto_available(fake_cuda) is False

        monkeypatch.setattr(
            blocks_mod, "get_cuda_pyramid_library", lambda: FakeLib()
        )
        assert blocks_mod._cuda_pyramid_auto_available(fake_cpu) is False
        assert blocks_mod._cuda_pyramid_auto_available(fake_bf16) is False
        # No gradients required: an inference-only library is fine.
        assert blocks_mod._cuda_pyramid_auto_available(fake_cuda) is True
        # Gradients required: without a backward pass the auto mode must
        # fall back to the (always trainable) Triton kernel.
        assert blocks_mod._cuda_pyramid_auto_available(
            FakeTensor(True, torch.float32, requires_grad=True)
        ) is False


class TestBlockConstruction:
    def test_triton_block_builds_pyramid(self, fmaps):
        f1, f2 = fmaps
        block = LocalCorrTritonBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        assert block.fmap1.shape == (B, H, W, C)
        assert len(block.fmap2_pyramid) == LEVELS
        for level in range(LEVELS):
            expected = (B, H // 2**level, W // 2**level, C)
            assert block.fmap2_pyramid[level].shape == expected

    def test_fused_block_packing(self, fmaps):
        f1, f2 = fmaps
        block = LocalCorrTritonFusedBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        numels = [B * C * (H // 2**lvl) * (W // 2**lvl) for lvl in range(LEVELS)]
        assert block.fmap2_packed.numel() == sum(numels)
        assert block.level_offsets == tuple(sum(numels[:lvl]) for lvl in range(LEVELS))
        assert block.level_shapes == tuple(
            (H // 2**lvl, W // 2**lvl) for lvl in range(LEVELS)
        )
        assert block.fmap2_offsets.dtype == torch.int64
        assert block.fmap2_shapes.dtype == torch.int32
        assert block.level_scales.tolist() == [1 / 2**lvl for lvl in range(LEVELS)]

    def test_cuda_pyramid_block_packing_eager(self, fmaps):
        f1, f2 = fmaps
        block = LocalCorrCudaPyramidBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        # Batch-major packed pyramid [B, C*sum_i h_i*w_i] (the layout of
        # the CUDA kernel and of the ONNX plugin node): batch row 0
        # starts with level 0, which is exactly fmap2 in NHWC.
        per_row = sum(C * (H // 2**lvl) * (W // 2**lvl) for lvl in range(LEVELS))
        assert block.fmap2_packed.shape == (B, per_row)
        assert torch.equal(
            block.fmap2_packed[0, : f2[0].numel()], f2[0].permute(1, 2, 0).reshape(-1)
        )
        assert not hasattr(block, "fmap2_offsets")

    def test_cuda_pyramid_block_packing_batch_major_under_export(
        self, fmaps, onnx_export_mode
    ):
        f1, f2 = fmaps
        block = LocalCorrCudaPyramidBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        assert block.fmap2_packed.shape[0] == B
        assert block.fmap2_packed.ndim == 2
        assert not hasattr(block, "fmap2_offsets")

    def test_generic_export_keeps_nchw(self, fmaps, onnx_export_mode):
        localcorr.set_onnx_backend("onnx")
        f1, f2 = fmaps
        block = LocalCorrTritonBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        assert block.fmap1_nchw is f1
        assert block.fmap2_nchw is f2
        assert not hasattr(block, "fmap1")

    def test_use_generic_export_ops_matrix(self, fmaps, onnx_export_mode):
        f1, f2 = fmaps
        plain = LocalCorrCudaPyramidBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        forced = LocalCorrCudaPyramidBlock(
            f1, f2, num_levels=LEVELS, radius=RADIUS, onnx_generic=True
        )
        localcorr.set_onnx_backend("tensorrt")
        assert plain.use_generic_export_ops() is False
        assert forced.use_generic_export_ops() is True
        localcorr.set_onnx_backend("onnx")
        assert plain.use_generic_export_ops() is True
        onnx_export_mode(False)
        assert plain.use_generic_export_ops() is False
        assert forced.use_generic_export_ops() is False


class TestSyclEagerGuards:
    def test_packing_is_batch_major_in_eager(self, fmaps):
        f1, f2 = fmaps
        block = LocalCorrSyclBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        # Batch-major packed pyramid [B, C*sum_i h_i*w_i] (the layout of the
        # XPU extension and of the ONNX plugin node): batch row 0 starts
        # with level 0, which is exactly fmap2 in NHWC.
        per_row = sum(C * (H // 2**lvl) * (W // 2**lvl) for lvl in range(LEVELS))
        assert block.fmap2_packed.shape == (B, per_row)
        assert torch.equal(
            block.fmap2_packed[0, : f2[0].numel()], f2[0].permute(1, 2, 0).reshape(-1)
        )

    def test_eager_requires_xpu_tensors(self, fmaps, coords):
        f1, f2 = fmaps
        block = LocalCorrSyclBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        if f1.is_xpu:
            pytest.skip("requires a non-XPU device")
        with pytest.raises(RuntimeError, match="torch.xpu"):
            block(coords)

    def test_plugin_node_path_shape(self, fmaps, coords, onnx_export_mode):
        # Under export with the tensorrt backend the block records the
        # plugin node, whose eager forward only materializes the shape.
        localcorr.set_onnx_backend("tensorrt")
        f1, f2 = fmaps
        block = LocalCorrSyclBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        out = block(coords)
        assert out.shape == (B, LEVELS * (2 * RADIUS + 1) ** 2, H, W)
        assert out.dtype == torch.float32


class TestGenericExportCalls:
    @pytest.mark.parametrize(
        "mode",
        [
            "localcorr_triton",
            "localcorr_cuda",
            "localcorr_onnx",
            "localcorr_sycl",
        ],
    )
    def test_call_matches_reference(self, fmaps, coords, onnx_export_mode, mode):
        localcorr.set_onnx_backend("onnx")
        f1, f2 = fmaps
        block = get_localcorr_block(
            *fmaps, num_levels=LEVELS, radius=RADIUS, corr_mode=mode
        )
        out = block(coords)
        ref = pyramid_localcorr_reference(f1, f2, coords, LEVELS, RADIUS)
        ref = ref.reshape(B, LEVELS * (2 * RADIUS + 1) ** 2, H, W) / math.sqrt(C)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-3)

    def test_onnx_mode_forces_generic_ops_with_tensorrt_backend(
        self, fmaps, coords, onnx_export_mode
    ):
        # corr_mode=localcorr_onnx must use standard ops even when the
        # global backend is "tensorrt".
        localcorr.set_onnx_backend("tensorrt")
        f1, f2 = fmaps
        block = get_localcorr_block(
            *fmaps, num_levels=LEVELS, radius=RADIUS, corr_mode="localcorr_onnx"
        )
        out = block(coords)
        ref = pyramid_localcorr_reference(f1, f2, coords, LEVELS, RADIUS)
        ref = ref.reshape(B, -1, H, W) / math.sqrt(C)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_plugin_node_path_shape(self, fmaps, coords, onnx_export_mode):
        # Under export with the tensorrt backend the block records the
        # plugin node, whose eager forward only materializes the shape.
        localcorr.set_onnx_backend("tensorrt")
        block = get_localcorr_block(
            *fmaps,
            num_levels=LEVELS,
            radius=RADIUS,
            corr_mode="localcorr_cuda",
        )
        out = block(coords)
        assert out.shape == (B, LEVELS * (2 * RADIUS + 1) ** 2, H, W)
        assert out.dtype == torch.float32


class TestOnnxWrappers:
    def test_pyramid_wrapper_forward_shape(self):
        fmap1 = torch.zeros(B, H, W, C)
        fmap2_packed = torch.zeros(B, C * H * W)
        coords = torch.zeros(B, 2, H, W)
        out = LocalCorrCudaPyramidOnnx.apply(
            fmap1, fmap2_packed, coords, RADIUS, LEVELS
        )
        rd = 2 * RADIUS + 1
        assert out.shape == (B, LEVELS, rd * rd, H, W)
        # During tracing the wrapper only materializes the output shape;
        # the values come from the recorded symbolic node.
        assert out.dtype == fmap1.dtype

    def test_spatial_wrapper_forward_shape(self):
        fmap1 = torch.zeros(B, H, W, C)
        coords = torch.zeros(B, 2, H, W)
        out = LocalCorrCudaPyramidSpatialOnnx.apply(
            fmap1, fmap1, coords, RADIUS, 2, "sqrt"
        )
        rd = 2 * RADIUS + 1
        assert out.shape == (B, rd * rd, H, W)
        assert out.dtype == fmap1.dtype


class TestSpatialBlock:
    def test_matches_shifted_reference(self):
        f1, f2 = make_fmaps(batch=2, channels=6, height=10, width=14, seed=22)
        block = LocalCorrCudaPyramidSpatialBlock(f1, f2, radius=3, dilation=1)
        out = block()
        ref = shifted_spatial_reference(f1, f2, radius=3, dilation=1, normalize="sqrt")
        assert out.shape == (2, 49, 10, 14)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_dilation(self):
        f1, f2 = make_fmaps(batch=2, channels=4, height=12, width=12, seed=23)
        block = LocalCorrCudaPyramidSpatialBlock(f1, f2, radius=2, dilation=3)
        ref = shifted_spatial_reference(f1, f2, radius=2, dilation=3, normalize="sqrt")
        assert torch.allclose(block(), ref, atol=1e-4)

    @pytest.mark.parametrize("normalize", ["sqrt", "dim", None])
    def test_normalize_modes(self, normalize):
        f1, f2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=24)
        block = LocalCorrCudaPyramidSpatialBlock(
            f1, f2, radius=1, dilation=1, normalize=normalize
        )
        out = block()
        ref = shifted_spatial_reference(
            f1, f2, radius=1, dilation=1, normalize=normalize
        )
        assert torch.allclose(out, ref, atol=1e-4)

    def test_matches_pwcnet_reference_dilation1(self):
        # With dilation=1 the block equals the flattened PWCNet sampler
        # (same y-major channel order, sqrt normalization).
        f1, f2 = make_fmaps(batch=2, channels=5, height=9, width=11, seed=25)
        out = LocalCorrCudaPyramidSpatialBlock(f1, f2, radius=2, dilation=1)()
        ref = spatial_corr_reference(f1, f2, 2).reshape(2, 25, 9, 11) / math.sqrt(5)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_coords_argument_is_ignored(self):
        f1, f2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=26)
        block = LocalCorrCudaPyramidSpatialBlock(f1, f2, radius=1)
        out_none = block()
        out_coords = block(torch.full((1, 2, 8, 8), 123.0))
        assert torch.equal(out_none, out_coords)

    def test_identity_coords_cache(self):
        f1, f2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=27)
        block = LocalCorrCudaPyramidSpatialBlock(f1, f2, radius=1)
        assert block._coords is None
        grid = block._identity_coords(1, 8, 8, torch.float32, torch.device("cpu"))
        assert (
            block._identity_coords(1, 8, 8, torch.float32, torch.device("cpu")) is grid
        )
        # Channel 0 must be x.
        assert torch.equal(grid[0, 0], torch.arange(8.0).repeat(8, 1))
        assert torch.equal(grid[0, 1], torch.arange(8.0).repeat(8, 1).t())
        # A different height rebuilds the cache.
        assert block._identity_coords(
            1, 6, 8, torch.float32, torch.device("cpu")
        ).shape == (
            1,
            2,
            6,
            8,
        )
        assert block._coords.shape == (1, 2, 6, 8)

    def test_export_branch_shape(self, onnx_export_mode):
        f1, f2 = make_fmaps(batch=2, channels=4, height=8, width=10, seed=28)
        block = LocalCorrCudaPyramidSpatialBlock(f1, f2, radius=2, dilation=2)
        out = block()
        assert out.shape == (2, 25, 8, 10)


class TestTritonAvailability:
    @pytest.mark.parametrize(
        "mode",
        ["localcorr_triton", "localcorr_onnx"],
    )
    def test_missing_triton_raises_clear_error(self, monkeypatch, coords, mode):
        # Simulate an environment without triton (no Linux wheels there):
        # importing localcorr.triton fails and the eager kernels raise a
        # clear ImportError on first use.
        monkeypatch.setitem(sys.modules, "localcorr.triton", None)
        blocks_mod._TRITON_KERNELS = None
        f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=29)
        block = get_localcorr_block(
            f1, f2, num_levels=LEVELS, radius=RADIUS, corr_mode=mode
        )
        with pytest.raises(ImportError, match="pip install triton"):
            block(coords)


class TestCudaPyramidEager:
    """Guards of the AOT CUDA kernel eager path (localcorr_cuda)."""

    def test_eager_requires_cuda_tensors(self, fmaps, coords):
        f1, f2 = fmaps
        if f1.is_cuda:
            pytest.skip("requires non-CUDA tensors")
        block = LocalCorrCudaPyramidBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        with pytest.raises(RuntimeError, match="CUDA device"):
            block(coords)

    def test_missing_library_raises_clear_error(self, fmaps, coords):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        f1, f2 = make_fmaps(
            batch=B, channels=C, height=H, width=W, seed=30, device="cuda"
        )
        block = LocalCorrCudaPyramidBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        saved = (blocks_mod._CUDA_PYRAMID_LIB, blocks_mod._CUDA_PYRAMID_LIB_LOADED)
        blocks_mod._CUDA_PYRAMID_LIB = None
        blocks_mod._CUDA_PYRAMID_LIB_LOADED = True
        try:
            with pytest.raises(ImportError, match="build_native cuda"):
                block(make_coords(batch=B, height=H, width=W, seed=31, device="cuda"))
        finally:
            blocks_mod._CUDA_PYRAMID_LIB, blocks_mod._CUDA_PYRAMID_LIB_LOADED = saved

    def test_stale_library_without_backward_raises(self, fmaps, coords):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        f1, f2 = make_fmaps(
            batch=B, channels=C, height=H, width=W, seed=32, device="cuda"
        )
        f1.requires_grad_(True)
        block = LocalCorrCudaPyramidBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        lib = blocks_mod.get_cuda_pyramid_library()
        if lib is None or not hasattr(lib, "launchLocalCorrCudaPyramidBackward"):
            pytest.skip("requires the built CUDA kernel library")
        saved = lib.launchLocalCorrCudaPyramidBackward
        lib.launchLocalCorrCudaPyramidBackward = None
        try:
            with pytest.raises(RuntimeError, match="no backward pass"):
                block(make_coords(batch=B, height=H, width=W, seed=33, device="cuda"))
        finally:
            lib.launchLocalCorrCudaPyramidBackward = saved

    def test_matches_triton_fused(self, coords):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        f1, f2 = make_fmaps(
            batch=B, channels=C, height=H, width=W, seed=34, device="cuda"
        )
        coords = make_coords(batch=B, height=H, width=W, seed=35, device="cuda")
        cuda_block = LocalCorrCudaPyramidBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
        triton_block = LocalCorrTritonFusedBlock(
            f1, f2, num_levels=LEVELS, radius=RADIUS
        )
        with torch.no_grad():
            out_cuda = cuda_block(coords)
            out_triton = triton_block(coords)
        assert out_cuda.shape == out_triton.shape
        assert torch.allclose(out_cuda, out_triton, atol=1e-4)


class TestSyclBlockExport:
    def test_export_path_matches_cuda(self, fmaps, coords, onnx_export_mode):
        # Under export the sycl block emits the same node as
        # localcorr_cuda (resolved by the SYCL OpenVINO extension).
        localcorr.set_onnx_backend("tensorrt")
        f1, f2 = fmaps
        block = get_localcorr_block(
            *fmaps, num_levels=LEVELS, radius=RADIUS, corr_mode="localcorr_sycl"
        )
        out = block(coords)
        assert out.shape == (B, LEVELS * (2 * RADIUS + 1) ** 2, H, W)
        assert out.dtype == torch.float32
