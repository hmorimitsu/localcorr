"""XPU-only tests: the real SYCL kernel (forward + backward) on torch.xpu.

Skipped automatically on machines without an Intel GPU (torch.xpu) or
when the localcorr/xpu extension has not been built.
"""

import math
import types

import pytest
import torch
import torch.nn.functional as F

import localcorr.blocks as blocks_mod
from localcorr import (
    get_localcorr_block,
    get_localcorr_sampler,
)
from localcorr.spatial_corr import get_sycl_xpu_extension

from _reference import (
    make_coords,
    make_fmaps,
    pyramid_localcorr_reference,
    spatial_corr_reference,
)


def _xpu_available():
    return (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and get_sycl_xpu_extension() is not None
        and hasattr(get_sycl_xpu_extension(), "backward")
    )


pytestmark = pytest.mark.skipif(
    not _xpu_available(),
    reason="Intel GPU (torch.xpu) and the localcorr SYCL extension "
    "(with backward) required",
)

B, C, H, W = 2, 64, 20, 26


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


@pytest.mark.parametrize(
    "levels,radius",
    [(1, 4), (2, 2), (3, 3), (4, 4)],
)
def test_pyramid_backward_matches_reference(levels, radius):
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=60, device="xpu")
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=61, device="xpu")
    f1 = f1.clone().requires_grad_(True)
    f2 = f2.clone().requires_grad_(True)
    block = get_localcorr_block(
        f1, f2, num_levels=levels, radius=radius, corr_mode="localcorr_sycl"
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
    torch.xpu.synchronize()
    assert torch.allclose(f1.grad, g1, atol=2e-3, rtol=2e-3)
    assert torch.allclose(f2.grad, g2, atol=2e-3, rtol=2e-3)


def test_pyramid_backward_half():
    f1, f2 = make_fmaps(
        batch=B, channels=C, height=H, width=W, seed=62, device="xpu"
    )
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=63, device="xpu")
    f1h = f1.half().clone().requires_grad_(True)
    f2h = f2.half().clone().requires_grad_(True)
    block = get_localcorr_block(
        f1h, f2h, num_levels=3, radius=3, corr_mode="localcorr_sycl"
    )
    out = block(coords.half())
    grad_out = torch.randn_like(out)
    (out * grad_out).sum().backward()
    torch.xpu.synchronize()
    assert f1h.grad is not None and torch.isfinite(f1h.grad).all()
    assert f2h.grad is not None and torch.isfinite(f2h.grad).all()
    assert f1h.grad.abs().sum() > 0 and f2h.grad.abs().sum() > 0
    assert f1h.grad.dtype == torch.float16 and f2h.grad.dtype == torch.float16


def test_extension_backward_dilation_and_ymajor():
    # dilation > 1 and ymajor go through the raw extension, since the
    # RAFT-style block API does not expose them.
    ext = get_sycl_xpu_extension()
    levels, radius, dilation = 2, 2, 2
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=64, device="xpu")
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=65, device="xpu")
    fmap1 = f1.permute(0, 2, 3, 1).contiguous()
    pyr = [f2]
    for _ in range(1, levels):
        pyr.append(F.avg_pool2d(pyr[-1], 2, stride=2))
    pyr = [f.permute(0, 2, 3, 1).contiguous() for f in pyr]
    packed = torch.stack(
        [
            torch.cat([f.permute(0, 2, 3, 1).contiguous()[b].reshape(-1) for f in pyr])
            for b in range(B)
        ]
    ).contiguous()
    rd = 2 * radius + 1

    corr = ext.forward(fmap1, packed, coords, radius, levels, dilation, True)
    corr_grad = torch.randn_like(corr)
    f1g, f2g, cg = ext.backward(
        fmap1, packed, coords, corr_grad, radius, levels, dilation, True
    )
    torch.xpu.synchronize()
    assert f1g.shape == fmap1.shape and f2g.shape == packed.shape
    assert cg.shape == coords.shape and cg.abs().sum().item() == 0.0
    assert torch.isfinite(f1g.float()).all() and torch.isfinite(f2g.float()).all()
    assert f1g.float().abs().sum() > 0 and f2g.float().abs().sum() > 0


def test_sampler_backward_grads():
    patch = 9
    i1, i2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=66, device="xpu")
    i1 = i1.clone().requires_grad_(True)
    i2 = i2.clone().requires_grad_(True)
    for mode in ("localcorr_sycl", "localcorr_auto"):
        sampler = get_localcorr_sampler(
            patch_size=patch, corr_mode=mode, xpu_backend="sycl"
        )
        out = sampler(i1, i2)
        ref = spatial_corr_reference(i1.detach(), i2.detach(), patch // 2)
        assert torch.allclose(out.detach(), ref, atol=1e-2)
        grad_out = torch.randn_like(out)
        (out * grad_out).sum().backward()
        torch.xpu.synchronize()
        assert i1.grad is not None and torch.isfinite(i1.grad).all()
        assert i2.grad is not None and torch.isfinite(i2.grad).all()
        assert i1.grad.abs().sum() > 0 and i2.grad.abs().sum() > 0
        i1.grad = i2.grad = None


def test_coords_grad_is_zero():
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=67, device="xpu")
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=68, device="xpu")
    coords = coords.clone().requires_grad_(True)
    block = get_localcorr_block(
        f1, f2, num_levels=2, radius=2, corr_mode="localcorr_sycl"
    )
    loss = block(coords).square().mean()
    loss.backward()
    torch.xpu.synchronize()
    assert coords.grad is not None
    assert coords.grad.abs().sum().item() == 0.0


def test_stale_extension_raises_clear_error():
    # An extension .so without the backward binding must fail clearly
    # when gradients are required, and still work under no_grad.
    ext = get_sycl_xpu_extension()
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=69, device="xpu")
    f1 = f1.clone().requires_grad_(True)
    stale = types.SimpleNamespace(forward=ext.forward)
    saved = (blocks_mod.get_sycl_xpu_extension,)
    blocks_mod.get_sycl_xpu_extension = lambda: stale
    try:
        block = get_localcorr_block(
            f1, f2, num_levels=2, radius=2, corr_mode="localcorr_sycl"
        )
        coords = make_coords(batch=B, height=H, width=W, seed=70, device="xpu")
        with pytest.raises(RuntimeError, match="backward"):
            block(coords)
    finally:
        blocks_mod.get_sycl_xpu_extension = saved[0]
