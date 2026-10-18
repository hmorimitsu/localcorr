"""Independent reference implementations used by the unit tests.

These re-derive the localcorr semantics with plain PyTorch (double
loops, shifts, per-displacement grid_sample), so they are intentionally
written differently from the package code under test.
"""

import math

import torch
import torch.nn.functional as F


def make_fmaps(batch=2, channels=8, height=16, width=24, seed=0, device="cpu"):
    """Return two random feature maps [B, C, H, W] (deterministic)."""
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {"generator": generator, "device": device}
    f1 = torch.randn(batch, channels, height, width, **kwargs)
    f2 = torch.randn(batch, channels, height, width, **kwargs)
    return f1, f2


def make_coords(batch=2, height=16, width=24, max_disp=3.0, seed=1, device="cpu"):
    """Random pixel coordinates [B, 2, H, W] (x in channel 0) around identity."""
    generator = torch.Generator(device=device).manual_seed(seed)
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    identity = torch.stack([xs, ys], dim=0)[None].repeat(batch, 1, 1, 1)
    flow = (
        torch.rand(batch, 2, height, width, generator=generator, device=device) * 2 - 1
    )
    return identity + flow * max_disp


def pixel_to_grid(coords, size):
    """Normalize pixel coordinates for grid_sample(align_corners=False)."""
    return (coords + 0.5) / size * 2.0 - 1.0


def pyramid_localcorr_reference(fmap1, fmap2, coords, num_levels, radius):
    """RAFT-style multi-level local correlation reference.

    Computes each displacement channel independently with a per-channel
    grid_sample over the (pooled) fmap2, in contrast to the matmul
    formulation under test. Returns [B, L, (2r+1)^2, H, W] (not
    normalized), with the RAFT flatten order d = ix * rd + iy (verified
    against the kernels and the ptlflow RAFT CorrBlock).
    """
    batch, channels, height, width = fmap1.shape
    rd = 2 * radius + 1
    f2_pyramid = [fmap2]
    for _ in range(1, num_levels):
        f2_pyramid.append(F.avg_pool2d(f2_pyramid[-1], 2, stride=2))
    levels = []
    for level, f2 in enumerate(f2_pyramid):
        h2, w2 = f2.shape[-2:]
        cx = coords[:, 0] / (2**level)
        cy = coords[:, 1] / (2**level)
        plane = torch.zeros(
            batch, height, width, rd * rd, dtype=fmap1.dtype, device=fmap1.device
        )
        for iy in range(rd):
            for ix in range(rd):
                gx = pixel_to_grid(cx + (ix - radius), w2)
                gy = pixel_to_grid(cy + (iy - radius), h2)
                grid = torch.stack([gx, gy], dim=-1)
                sampled = F.grid_sample(
                    f2, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                plane[..., ix * rd + iy] = (fmap1 * sampled).sum(dim=1)
        levels.append(plane.permute(0, 3, 1, 2))
    return torch.stack(levels, dim=1)


def shifted_spatial_reference(fmap1, fmap2, radius, dilation=1, normalize="sqrt"):
    """Single-level dilated spatial correlation, flattened [B, rd*rd, H, W].

    Matches LocalCorrCudaPyramidSpatialBlock:
    corr[b, iy*rd+ix, h, w] = sum_c fmap1[b,c,h,w] *
        fmap2[b, c, h + dil*(iy-r), w + dil*(ix-r)]  (zero padded).
    """
    batch, channels, height, width = fmap1.shape
    rd = 2 * radius + 1
    out = torch.zeros(
        batch, rd * rd, height, width, dtype=fmap1.dtype, device=fmap1.device
    )
    for iy in range(rd):
        for ix in range(rd):
            yy = dilation * (iy - radius)
            xx = dilation * (ix - radius)
            sh_lo, sh_hi = max(0, -yy), min(height, height - yy)
            sw_lo, sw_hi = max(0, -xx), min(width, width - xx)
            if sh_lo < sh_hi and sw_lo < sw_hi:
                out[:, iy * rd + ix, sh_lo:sh_hi, sw_lo:sw_hi] = (
                    fmap1[:, :, sh_lo:sh_hi, sw_lo:sw_hi]
                    * fmap2[:, :, sh_lo + yy : sh_hi + yy, sw_lo + xx : sw_hi + xx]
                ).sum(dim=1)
    if normalize == "sqrt":
        out = out / math.sqrt(channels)
    elif normalize == "dim":
        out = out / channels
    return out


def spatial_corr_reference(input1, input2, radius):
    """PWCNet-style spatial correlation reference: [B, ps, ps, H, W].

    out[b, dy_idx, dx_idx, h, w] = sum_c x1[b,c,h,w] *
    x2[b, c, h + (dy_idx-r), w + (dx_idx-r)] (zero padded).
    """
    batch, channels, height, width = input1.shape
    ps = 2 * radius + 1
    padded = F.pad(input2, (radius, radius, radius, radius))
    out = torch.zeros(
        batch, ps, ps, height, width, dtype=input1.dtype, device=input1.device
    )
    for iy in range(ps):
        for ix in range(ps):
            out[:, iy, ix] = (
                input1 * padded[:, :, iy : iy + height, ix : ix + width]
            ).sum(dim=1)
    return out
