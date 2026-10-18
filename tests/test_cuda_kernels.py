"""GPU-only tests: the real Triton kernels.

Skipped automatically on machines without a CUDA device (e.g. the plain
CPU runners used by CI).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from localcorr import LocalCorrCudaPyramidSampler, get_localcorr_block, get_localcorr_sampler
from localcorr.blocks import LocalCorrCudaPyramidFunction

from _reference import (
    make_coords,
    make_fmaps,
    pyramid_localcorr_reference,
    spatial_corr_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

B, C, H, W = 2, 64, 40, 52
LEVELS, RADIUS = 4, 4


def _cuda_backward_available():
    from localcorr.blocks import get_cuda_pyramid_library

    lib = get_cuda_pyramid_library()
    return (
        lib is not None
        and getattr(lib, "launchLocalCorrCudaPyramidBackward", None) is not None
    )


requires_cuda_backward = pytest.mark.skipif(
    not _cuda_backward_available(),
    reason="CUDA device, the built CUDA kernel library (with backward) required",
)


def _pyramid_grads_reference(f1, f2, coords, levels, radius, grad_out):
    """Autograd of the reference pyramid correlation (unnormalized).

    The block normalizes its output by sqrt(C), so the block's gradients
    are the reference gradients divided by sqrt(C).
    """
    f1r = f1.detach().clone().requires_grad_(True)
    f2r = f2.detach().clone().requires_grad_(True)
    out = pyramid_localcorr_reference(f1r, f2r, coords, levels, radius)
    (g1, g2) = torch.autograd.grad(out, [f1r, f2r], grad_out.reshape(out.shape))
    return g1 / math.sqrt(C), g2 / math.sqrt(C)


@pytest.fixture
def cuda_fmaps():
    return make_fmaps(batch=B, channels=C, height=H, width=W, seed=40, device="cuda")


@pytest.fixture
def cuda_coords():
    return make_coords(batch=B, height=H, width=W, max_disp=8.0, seed=41, device="cuda")


@pytest.mark.parametrize(
    "mode",
    [
        "localcorr_triton",
        "localcorr_cuda",
        "localcorr_onnx",
    ],
)
def test_pyramid_modes_match_reference(cuda_fmaps, cuda_coords, mode):
    f1, f2 = cuda_fmaps
    coords = cuda_coords
    block = get_localcorr_block(
        f1, f2, num_levels=LEVELS, radius=RADIUS, corr_mode=mode
    )
    out = block(coords)
    ref = pyramid_localcorr_reference(f1, f2, coords, LEVELS, RADIUS)
    ref = ref.reshape(B, -1, H, W) / math.sqrt(C)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-2)


def test_spatial_sampler_matches_reference():
    i1, i2 = make_fmaps(
        batch=2, channels=32, height=24, width=30, seed=42, device="cuda"
    )
    sampler = LocalCorrCudaPyramidSampler(patch_size=9)
    out = sampler(i1, i2)
    ref = spatial_corr_reference(i1, i2, 4)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-2)


def test_fused_backward_grads():
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=43, device="cuda")
    coords = make_coords(
        batch=B, height=H, width=W, max_disp=8.0, seed=44, device="cuda"
    )
    f1 = f1.clone().requires_grad_(True)
    f2 = f2.clone().requires_grad_(True)
    block = get_localcorr_block(
        f1, f2, num_levels=LEVELS, radius=RADIUS, corr_mode="localcorr_triton"
    )
    loss = block(coords).square().mean()
    loss.backward()
    assert torch.isfinite(f1.grad).all() and f1.grad.abs().sum() > 0
    assert torch.isfinite(f2.grad).all() and f2.grad.abs().sum() > 0


def test_per_level_block_matches_reference(cuda_fmaps, cuda_coords):
    # LocalCorrTritonBlock (one launch per level) is no longer a corr mode;
    # it is kept as the internal reference for the fused kernel.
    from localcorr.blocks import LocalCorrTritonBlock

    f1, f2 = cuda_fmaps
    coords = cuda_coords
    block = LocalCorrTritonBlock(f1, f2, num_levels=LEVELS, radius=RADIUS)
    out = block(coords)
    ref = pyramid_localcorr_reference(f1, f2, coords, LEVELS, RADIUS)
    ref = ref.reshape(B, -1, H, W) / math.sqrt(C)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-2)


@pytest.mark.parametrize(
    "levels,radius",
    [(1, 4), (2, 2), (3, 3), (4, 4)],
)
@requires_cuda_backward
def test_cuda_pyramid_backward_matches_reference(levels, radius):
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=60, device="cuda")
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=61, device="cuda")
    f1 = f1.clone().requires_grad_(True)
    f2 = f2.clone().requires_grad_(True)
    block = get_localcorr_block(
        f1, f2, num_levels=levels, radius=radius, corr_mode="localcorr_cuda"
    )
    out = block(coords)
    grad_out = torch.randn_like(out)
    (out * grad_out).sum().backward()

    # The block normalizes by sqrt(C); scale the reference grads alike.
    ref_out = pyramid_localcorr_reference(
        f1.detach(), f2.detach(), coords, levels, radius
    ).reshape(B, -1, H, W) / math.sqrt(C)
    assert torch.allclose(out.detach(), ref_out, atol=1e-3)
    g1, g2 = _pyramid_grads_reference(f1, f2, coords, levels, radius, grad_out)
    torch.cuda.synchronize()
    assert torch.allclose(f1.grad, g1, atol=2e-3, rtol=2e-3)
    assert torch.allclose(f2.grad, g2, atol=2e-3, rtol=2e-3)


@requires_cuda_backward
def test_cuda_pyramid_backward_matches_triton():
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=62, device="cuda")
    coords = make_coords(batch=B, height=H, width=W, max_disp=4.0, seed=63, device="cuda")
    grads = {}
    for mode in ("localcorr_triton", "localcorr_cuda"):
        fa = f1.clone().requires_grad_(True)
        fb = f2.clone().requires_grad_(True)
        block = get_localcorr_block(
            fa, fb, num_levels=LEVELS, radius=RADIUS, corr_mode=mode
        )
        loss = block(coords).square().mean()
        loss.backward()
        grads[mode] = (fa.grad, fb.grad)
    scale1 = grads["localcorr_triton"][0].abs().max().clamp_min(1e-12)
    scale2 = grads["localcorr_triton"][1].abs().max().clamp_min(1e-12)
    assert (grads["localcorr_cuda"][0] - grads["localcorr_triton"][0]).abs().max() / scale1 < 1e-4
    assert (grads["localcorr_cuda"][1] - grads["localcorr_triton"][1]).abs().max() / scale2 < 1e-4


@requires_cuda_backward
def test_cuda_pyramid_backward_half():
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=64, device="cuda")
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=65, device="cuda")
    f1h = f1.half().clone().requires_grad_(True)
    f2h = f2.half().clone().requires_grad_(True)
    block = get_localcorr_block(
        f1h, f2h, num_levels=3, radius=3, corr_mode="localcorr_cuda"
    )
    out = block(coords.half())
    grad_out = torch.randn_like(out)
    (out * grad_out).sum().backward()
    torch.cuda.synchronize()
    assert f1h.grad is not None and torch.isfinite(f1h.grad).all()
    assert f2h.grad is not None and torch.isfinite(f2h.grad).all()
    assert f1h.grad.abs().sum() > 0 and f2h.grad.abs().sum() > 0
    assert f1h.grad.dtype == torch.float16 and f2h.grad.dtype == torch.float16


@requires_cuda_backward
def test_cuda_pyramid_backward_dilation_and_ymajor():
    # dilation > 1 and ymajor go through the raw autograd function, since
    # the RAFT-style block API does not expose them.
    levels, radius, dilation = 2, 2, 2
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=66, device="cuda")
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=67, device="cuda")
    fmap1 = f1.permute(0, 2, 3, 1).contiguous().requires_grad_(True)
    pyr = [f2]
    for _ in range(1, levels):
        pyr.append(F.avg_pool2d(pyr[-1], 2, stride=2))
    pyr = [f.permute(0, 2, 3, 1).contiguous() for f in pyr]
    packed = torch.stack(
        [
            torch.cat([f.permute(0, 2, 3, 1).contiguous()[b].reshape(-1) for f in pyr])
            for b in range(B)
        ]
    ).contiguous().requires_grad_(True)
    coords = coords.contiguous().requires_grad_(True)

    corr = LocalCorrCudaPyramidFunction.apply(
        fmap1, packed, coords, radius, levels, dilation, True
    )
    corr_grad = torch.randn_like(corr)
    f1g, f2g, cg = torch.autograd.grad(corr, [fmap1, packed, coords], corr_grad)
    torch.cuda.synchronize()
    assert f1g.shape == fmap1.shape and f2g.shape == packed.shape
    assert cg.shape == coords.shape and cg.abs().sum().item() == 0.0
    assert torch.isfinite(f1g.float()).all() and torch.isfinite(f2g.float()).all()
    assert f1g.float().abs().sum() > 0 and f2g.float().abs().sum() > 0


@requires_cuda_backward
def test_cuda_pyramid_coords_grad_is_zero():
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=68, device="cuda")
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=69, device="cuda")
    coords = coords.clone().requires_grad_(True)
    block = get_localcorr_block(
        f1, f2, num_levels=2, radius=2, corr_mode="localcorr_cuda"
    )
    loss = block(coords).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert coords.grad is not None
    assert coords.grad.abs().sum().item() == 0.0


@requires_cuda_backward
def test_cuda_sampler_backward_grads():
    patch = 9
    i1, i2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=70, device="cuda")
    i1 = i1.clone().requires_grad_(True)
    i2 = i2.clone().requires_grad_(True)
    sampler = get_localcorr_sampler(patch_size=patch, corr_mode="localcorr_cuda")
    out = sampler(i1, i2)
    ref = spatial_corr_reference(i1.detach(), i2.detach(), patch // 2)
    assert torch.allclose(out.detach(), ref, atol=1e-2)
    grad_out = torch.randn_like(out)
    (out * grad_out).sum().backward()
    torch.cuda.synchronize()
    assert i1.grad is not None and torch.isfinite(i1.grad).all()
    assert i2.grad is not None and torch.isfinite(i2.grad).all()
    assert i1.grad.abs().sum() > 0 and i2.grad.abs().sum() > 0


def _clear_cuda_library_cache():
    """Force get_cuda_pyramid_library() to report 'not built'."""
    from localcorr import blocks as blocks_mod

    saved = (blocks_mod._CUDA_PYRAMID_LIB, blocks_mod._CUDA_PYRAMID_LIB_LOADED)
    blocks_mod._CUDA_PYRAMID_LIB = None
    blocks_mod._CUDA_PYRAMID_LIB_LOADED = True
    return saved


def _restore_cuda_library_cache(saved):
    from localcorr import blocks as blocks_mod

    blocks_mod._CUDA_PYRAMID_LIB, blocks_mod._CUDA_PYRAMID_LIB_LOADED = saved


class TestAutoModeCudaPreference:
    """localcorr_auto prefers localcorr_cuda, Triton as fallback."""

    def test_block_prefers_cuda_engine(self):
        from localcorr.blocks import (
            LocalCorrCudaPyramidBlock,
            LocalCorrTritonFusedBlock,
            get_cuda_pyramid_library,
        )

        f1, f2 = make_fmaps(
            batch=B, channels=C, height=H, width=W, seed=71, device="cuda"
        )
        block = get_localcorr_block(
            f1, f2, num_levels=LEVELS, radius=RADIUS, corr_mode="localcorr_auto"
        )
        if get_cuda_pyramid_library() is None:
            # Library not built on this machine: Triton fallback.
            assert type(block) is LocalCorrTritonFusedBlock
            return
        assert type(block) is LocalCorrCudaPyramidBlock
        coords = make_coords(
            batch=B, height=H, width=W, max_disp=4.0, seed=72, device="cuda"
        )
        with torch.no_grad():
            out = block(coords)
        ref = pyramid_localcorr_reference(f1, f2, coords, LEVELS, RADIUS)
        ref = ref.reshape(B, -1, H, W) / math.sqrt(C)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-2)

    def test_block_falls_back_to_triton_without_library(self):
        from localcorr.blocks import LocalCorrTritonFusedBlock

        f1, f2 = make_fmaps(
            batch=B, channels=C, height=H, width=W, seed=73, device="cuda"
        )
        saved = _clear_cuda_library_cache()
        try:
            block = get_localcorr_block(
                f1, f2, num_levels=LEVELS, radius=RADIUS, corr_mode="localcorr_auto"
            )
        finally:
            _restore_cuda_library_cache(saved)
        assert type(block) is LocalCorrTritonFusedBlock
        coords = make_coords(
            batch=B, height=H, width=W, max_disp=4.0, seed=74, device="cuda"
        )
        with torch.no_grad():
            out = block(coords)
        ref = pyramid_localcorr_reference(f1, f2, coords, LEVELS, RADIUS)
        ref = ref.reshape(B, -1, H, W) / math.sqrt(C)
        assert torch.allclose(out, ref, atol=1e-2)

    def test_sampler_prefers_cuda_engine(self):
        from localcorr.blocks import get_cuda_pyramid_library

        i1, i2 = make_fmaps(
            batch=2, channels=32, height=24, width=30, seed=75, device="cuda"
        )
        sampler = LocalCorrCudaPyramidSampler(patch_size=9)
        corr = sampler._cuda_pyramid(i1, i2)
        if get_cuda_pyramid_library() is None:
            assert corr is None
            return
        assert corr is not None and corr.shape == (2, 1, 81, 24, 30)
        with torch.no_grad():
            out = sampler(i1, i2)
        ref = spatial_corr_reference(i1, i2, 4)
        assert torch.allclose(out, ref, atol=1e-2)

    def test_sampler_falls_back_to_triton_without_library(self):
        i1, i2 = make_fmaps(
            batch=2, channels=32, height=24, width=30, seed=76, device="cuda"
        )
        sampler = LocalCorrCudaPyramidSampler(patch_size=9)
        saved = _clear_cuda_library_cache()
        try:
            assert sampler._cuda_pyramid(i1, i2) is None
            with torch.no_grad():
                out = sampler(i1, i2)
        finally:
            _restore_cuda_library_cache(saved)
        ref = spatial_corr_reference(i1, i2, 4)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-2)
