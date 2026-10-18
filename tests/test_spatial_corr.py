"""Tests for localcorr.spatial_corr (PWCNet-style sampler)."""

import sys

import pytest
import torch

import localcorr
import localcorr.spatial_corr as spatial_mod
from localcorr.spatial_corr import (
    LocalCorrCudaPyramidSampler,
    get_sycl_xpu_extension,
)

from _reference import make_fmaps, spatial_corr_reference


class TestConstructor:
    def test_defaults(self):
        sampler = LocalCorrCudaPyramidSampler(patch_size=9)
        assert sampler.md == 4
        assert sampler.patch_size == 9
        assert sampler.onnx_generic is False
        assert sampler.xpu_backend == "sycl"
        assert sampler._coords is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kernel_size": 3},
            {"stride": 2},
            {"padding": 1},
            {"dilation": 2},
            {"dilation_patch": 2},
            {"xpu_backend": "cuda"},
        ],
    )
    def test_unsupported_arguments_raise(self, kwargs):
        with pytest.raises(AssertionError):
            LocalCorrCudaPyramidSampler(**kwargs)


class TestIdentityCoords:
    def test_layout(self):
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        grid = sampler._identity_coords(2, 6, 8, torch.float32, torch.device("cpu"))
        assert grid.shape == (2, 2, 6, 8)
        # Channel 0 = x, channel 1 = y.
        assert torch.equal(grid[0, 0], torch.arange(8.0).repeat(6, 1))
        assert torch.equal(grid[0, 1], torch.arange(6.0).repeat(8, 1).t())

    def test_cached_until_shape_changes(self):
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        first = sampler._identity_coords(1, 6, 8, torch.float32, torch.device("cpu"))
        assert (
            sampler._identity_coords(1, 6, 8, torch.float32, torch.device("cpu"))
            is first
        )
        assert (
            sampler._identity_coords(2, 6, 8, torch.float32, torch.device("cpu"))
            is not first
        )
        other = sampler._identity_coords(1, 5, 8, torch.float32, torch.device("cpu"))
        assert other.shape == (1, 2, 5, 8)
        assert (
            sampler._identity_coords(1, 5, 8, torch.float64, torch.device("cpu"))
            is not other
        )


class TestForwardPaths:
    def test_generic_export_matches_reference(self, onnx_export_mode):
        i1, i2 = make_fmaps(batch=2, channels=4, height=10, width=12, seed=30)
        sampler = LocalCorrCudaPyramidSampler(patch_size=7, onnx_generic=True)
        out = sampler(i1, i2)
        ref = spatial_corr_reference(i1, i2, 3)
        assert out.shape == (2, 7, 7, 10, 12)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_generic_backend_without_flag(self, onnx_export_mode):
        # Global "onnx" backend also selects the standard-ops path.
        localcorr.set_onnx_backend("onnx")
        i1, i2 = make_fmaps(batch=1, channels=3, height=8, width=8, seed=31)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        out = sampler(i1, i2)
        ref = spatial_corr_reference(i1, i2, 2)
        assert torch.allclose(out, ref, atol=1e-4)

    def test_plugin_export_shape(self, onnx_export_mode):
        # TensorRT backend under export: the plugin wrapper returns an
        # empty tensor with the sampler's output shape.
        localcorr.set_onnx_backend("tensorrt")
        i1, i2 = make_fmaps(batch=2, channels=4, height=8, width=10, seed=32)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        out = sampler(i1, i2)
        assert out.shape == (2, 5, 5, 8, 10)

    def test_missing_triton_raises_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "localcorr.triton", None)
        spatial_mod._TRITON_SPATIAL_FN = None
        i1, i2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=33)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        with pytest.raises(ImportError, match="pip install triton"):
            sampler(i1, i2)


class TestSyclExtension:
    def test_extension_lookup_is_cached_and_safe(self):
        ext = get_sycl_xpu_extension()
        assert get_sycl_xpu_extension() is ext

    def test_sycl_xpu_returns_none_without_extension(self):
        if get_sycl_xpu_extension() is not None:
            pytest.skip("SYCL XPU extension is built on this machine")
        i1, i2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=34)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        assert sampler._sycl_xpu(i1, i2) is None


class TestCudaPyramidPreference:
    """localcorr_auto prefers the AOT CUDA kernel, Triton as fallback."""

    def test_cuda_pyramid_returns_none_without_library(self):
        from localcorr.blocks import get_cuda_pyramid_library

        if get_cuda_pyramid_library() is not None:
            pytest.skip("CUDA kernel library is built on this machine")
        i1, i2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=36)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        assert sampler._cuda_pyramid(i1, i2) is None

    def test_forward_prefers_cuda_pyramid(self, monkeypatch):
        # With a usable CUDA engine, forward() must dispatch to it instead
        # of the Triton kernel. The inputs fake CUDA tensors (a real CUDA
        # device is not needed: the engine is simulated).
        class CudaFlagTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        i1, i2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=37)
        i1 = i1.as_subclass(CudaFlagTensor)
        i2 = i2.as_subclass(CudaFlagTensor)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        corr = torch.zeros(1, 1, 25, 8, 8)  # the kernel's [B, 1, rd*rd, H, W]
        monkeypatch.setattr(sampler, "_cuda_pyramid", lambda a, b: corr)
        assert torch.equal(sampler(i1, i2), corr.view(1, 5, 5, 8, 8))

    def test_cuda_branch_falls_back_within_forward(self, monkeypatch):
        # A CUDA tensor whose engine is unusable must not raise from the
        # Triton kernel being skipped: forward() consults _cuda_pyramid
        # and falls back to Triton (simulated) when it returns None.
        class CudaFlagTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        class FakeTritonFunction:
            @staticmethod
            def apply(f1, f2, md):
                return spatial_corr_reference(f1, f2, md)

        i1, i2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=39)
        i1 = i1.as_subclass(CudaFlagTensor)
        i2 = i2.as_subclass(CudaFlagTensor)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        monkeypatch.setattr(sampler, "_cuda_pyramid", lambda a, b: None)
        saved = spatial_mod._TRITON_SPATIAL_FN
        spatial_mod._TRITON_SPATIAL_FN = FakeTritonFunction
        try:
            out = sampler(i1, i2)
        finally:
            spatial_mod._TRITON_SPATIAL_FN = saved
        ref = spatial_corr_reference(i1, i2, 2)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-4)

    def test_forward_falls_back_to_triton(self, monkeypatch):
        # When the CUDA engine is unusable, forward() must fall back to
        # the Triton kernel (simulated here; running it needs a GPU).
        class FakeTritonFunction:
            @staticmethod
            def apply(f1, f2, md):
                return spatial_corr_reference(f1, f2, md)

        i1, i2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=38)
        sampler = LocalCorrCudaPyramidSampler(patch_size=5)
        monkeypatch.setattr(sampler, "_cuda_pyramid", lambda a, b: None)
        saved = spatial_mod._TRITON_SPATIAL_FN
        spatial_mod._TRITON_SPATIAL_FN = FakeTritonFunction
        try:
            out = sampler(i1, i2)
        finally:
            spatial_mod._TRITON_SPATIAL_FN = saved
        ref = spatial_corr_reference(i1, i2, 2)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-4)
