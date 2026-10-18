"""Tests for the strict PWCNet-style corr modes (get_localcorr_sampler)."""

import math

import pytest
import torch

import localcorr
import localcorr.blocks as blocks_mod
from localcorr.blocks import (
    get_localcorr_sampler,
    LocalCorrCudaSpatialSampler,
    LocalCorrOnnxSpatialSampler,
    LocalCorrSyclSpatialSampler,
)
from localcorr import LocalCorrCudaPyramidSampler

from _reference import make_fmaps, spatial_corr_reference

B, C, H, W = 2, 16, 20, 28
PATCH = 9


def make_inputs(seed=50, device="cpu"):
    f1, f2 = make_fmaps(
        batch=B, channels=C, height=H, width=W, seed=seed, device=device
    )
    return f1, f2


class TestDispatch:
    @pytest.mark.parametrize(
        "mode,cls",
        [
            ("localcorr_cuda", LocalCorrCudaSpatialSampler),
            ("localcorr_onnx", LocalCorrOnnxSpatialSampler),
            ("localcorr_sycl", LocalCorrSyclSpatialSampler),
            ("localcorr_auto", LocalCorrCudaPyramidSampler),
        ],
    )
    def test_dispatch(self, mode, cls):
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode=mode)
        assert type(sampler) is cls
        assert sampler.md == PATCH // 2

    def test_triton_dispatch(self):
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_triton")
        from localcorr.triton import TritonSpatialCorrelationSampler

        assert type(sampler) is TritonSpatialCorrelationSampler

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown local corr_mode"):
            get_localcorr_sampler(corr_mode="allpairs_spatial")


class TestCudaSpatialSampler:
    def test_matches_reference_and_triton(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        f1, f2 = make_inputs(seed=51, device="cuda")
        ref = spatial_corr_reference(f1.cpu(), f2.cpu(), PATCH // 2)
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_cuda")
        with torch.no_grad():
            out = sampler(f1, f2)
        assert out.shape == (B, PATCH, PATCH, H, W)
        assert torch.allclose(out.cpu(), ref, atol=1e-3)

    def test_fp16(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        f1, f2 = make_inputs(seed=52, device="cuda")
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_cuda")
        with torch.no_grad():
            out = sampler(f1.half(), f2.half())
        assert out.shape == (B, PATCH, PATCH, H, W)
        assert out.dtype == torch.float16

    def test_requires_cuda_tensors(self):
        f1, f2 = make_inputs(seed=53)
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_cuda")
        with pytest.raises(RuntimeError, match="CUDA device"):
            sampler(f1, f2)

    def test_gradients_match_triton(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        from localcorr.spatial_corr import _get_triton_spatial_function

        f1, f2 = make_inputs(seed=54, device="cuda")
        f1t, f2t = f1.clone().requires_grad_(True), f2.clone().requires_grad_(True)
        f1c, f2c = f1.clone().requires_grad_(True), f2.clone().requires_grad_(True)
        triton_fn = _get_triton_spatial_function()
        triton_fn.apply(f1t, f2t, PATCH // 2).square().mean().backward()
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_cuda")
        sampler(f1c, f2c).square().mean().backward()
        scale1 = f1t.grad.abs().max().clamp_min(1e-12)
        scale2 = f2t.grad.abs().max().clamp_min(1e-12)
        assert (f1c.grad - f1t.grad).abs().max() / scale1 < 1e-4
        assert (f2c.grad - f2t.grad).abs().max() / scale2 < 1e-4

    def test_stale_library_without_backward_raises(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        f1, f2 = make_inputs(seed=57, device="cuda")
        f1.requires_grad_(True)
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_cuda")
        lib = blocks_mod.get_cuda_pyramid_library()
        saved = lib.launchLocalCorrCudaPyramidBackward
        lib.launchLocalCorrCudaPyramidBackward = None
        try:
            with pytest.raises(RuntimeError, match="no backward pass"):
                sampler(f1, f2)
        finally:
            lib.launchLocalCorrCudaPyramidBackward = saved

    def test_missing_library_raises(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA device")
        f1, f2 = make_inputs(seed=55, device="cuda")
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_cuda")
        saved = (blocks_mod._CUDA_PYRAMID_LIB, blocks_mod._CUDA_PYRAMID_LIB_LOADED)
        blocks_mod._CUDA_PYRAMID_LIB = None
        blocks_mod._CUDA_PYRAMID_LIB_LOADED = True
        try:
            with pytest.raises(ImportError, match="build_native cuda"):
                sampler(f1, f2)
        finally:
            blocks_mod._CUDA_PYRAMID_LIB, blocks_mod._CUDA_PYRAMID_LIB_LOADED = saved

    def test_plugin_node_path_shape(self, onnx_export_mode):
        localcorr.set_onnx_backend("tensorrt")
        f1, f2 = make_inputs(seed=56)
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_cuda")
        out = sampler(f1, f2)
        assert out.shape == (B, PATCH, PATCH, H, W)


class TestSyclSpatialSampler:
    def test_eager_requires_xpu_tensors(self):
        f1, f2 = make_inputs(seed=57)
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_sycl")
        with pytest.raises(RuntimeError, match="torch.xpu"):
            sampler(f1, f2)

    def test_plugin_node_path_shape(self, onnx_export_mode):
        localcorr.set_onnx_backend("tensorrt")
        f1, f2 = make_inputs(seed=58)
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_sycl")
        out = sampler(f1, f2)
        assert out.shape == (B, PATCH, PATCH, H, W)


class TestOnnxSpatialSampler:
    def test_generic_export_matches_reference(self, onnx_export_mode):
        localcorr.set_onnx_backend("tensorrt")
        f1, f2 = make_inputs(seed=59)
        sampler = get_localcorr_sampler(patch_size=PATCH, corr_mode="localcorr_onnx")
        out = sampler(f1, f2)
        ref = spatial_corr_reference(f1, f2, PATCH // 2)
        assert out.shape == (B, PATCH, PATCH, H, W)
        assert torch.allclose(out, ref, atol=1e-4)
