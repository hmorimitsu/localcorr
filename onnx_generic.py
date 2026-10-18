"""Standard-ops implementations of the local correlation for ONNX export.

These functions reproduce the semantics of the Triton kernels (and of the
LocalCorrCudaPyramid TensorRT plugin) using only regular PyTorch ops, so
the exported ONNX graphs can be converted and executed by any ONNX
runtime (e.g. OpenVINO) without custom plugins.

The implementations only run during ONNX export (traced). Eager execution
always goes through the Triton kernels.

The pyramid correlation is computed with a matmul formulation: by the
linearity of the dot product, correlating fmap1 with bilinearly sampled
fmap2 values equals bilinearly sampling the full correlation volume
M[p, q] = sum_c fmap1[c, p] * fmap2[c, q] (gathered on single-channel
planes, so the gathers are cheap and the heavy channel contraction runs
on tensor cores). M is computed in row tiles whose lifetimes are chained
so runtimes can reuse one buffer for all tiles, keeping the memory
bounded (unlike the all-pairs RAFT formulation, which keeps the whole
volume alive). The sampling uses grid_sample with align_corners=False
and zeros padding, which numerically matches the floor-based bilinear
interpolation with per-corner zero masking used by the kernels.

A direct grid_sample over the 81 displacement windows of fmap2 (the
previous formulation) is ~6x slower on TensorRT/Jetson: each output pixel
must gather 4 corners for every channel of fmap2, while in the matmul
formulation the channel dimension is contracted by a GEMM and the
grid_sample gathers only one channel.
"""

import torch
import torch.nn.functional as F


def _pixel_to_grid(coords: torch.Tensor, size: float) -> torch.Tensor:
    """Convert pixel coordinates to grid_sample normalized coordinates.

    Uses the align_corners=False convention (pixel centers at integers,
    image edges at -/+ 0.5), which reproduces the kernels' floor-based
    bilinear interpolation with per-corner zero masking for out-of-bounds
    coordinates.
    """
    return (coords.to(torch.float32) + 0.5) / size * 2.0 - 1.0


# Target size (in bytes, fp16) of one correlation-volume row tile. Larger
# tiles need fewer (larger) GEMMs; smaller tiles bound the peak memory.
_CORR_TILE_MAX_BYTES = 48 * 1024 * 1024
# Maximum number of fmap1 rows per tile (upper bound for the tile height).
_CORR_TILE_MAX_ROWS = 8


def _corr_tile_rows(width_1: int, h2: int, w2: int) -> int:
    """Number of fmap1 rows per correlation tile for the given shapes."""
    bytes_per_row = 2 * width_1 * h2 * w2
    return max(1, min(_CORR_TILE_MAX_ROWS, _CORR_TILE_MAX_BYTES // bytes_per_row))


def localcorr_pyramid_onnx_generic(
    fmap1_nchw: torch.Tensor,
    fmap2_nchw: torch.Tensor,
    coords: torch.Tensor,
    num_levels: int,
    radius: int,
) -> torch.Tensor:
    """RAFT-style multi-level local correlation with standard ops.

    Args:
        fmap1_nchw: [B, C, H1, W1] feature map (first image).
        fmap2_nchw: [B, C, H1, W1] feature map (second image); its pyramid
            is built internally with 2x average pooling, like the kernels.
        coords: [B, 2, H1, W1] pixel coordinates (x in channel 0), level-0
            scale.
        num_levels: number of pyramid levels.
        radius: local window radius.

    Returns:
        [B, num_levels, (2*radius+1)^2, H1, W1] correlation volume with the
        RAFT x-major displacement order (channel = ix + rd * iy), matching
        the Triton kernels and the LocalCorrCudaPyramid plugin.

    The correlation volume M[p, q] = sum_c fmap1[c, p] * fmap2[c, q] is
    computed per pyramid level in tiles of fmap1 rows (p). Each tile is
    immediately sampled at the (2*radius+1)^2 displacement positions with a
    grid_sample over single-channel planes, and a bit-exact no-op
    dependency (the previous tile's first sample, clamped and multiplied by
    zero, added to the next tile's matmul input) is threaded between
    consecutive tiles so that the tile lifetimes are serialized and the
    memory planner can reuse one buffer for all tiles of a call.
    """
    batch_size, channels, height_1, width_1 = fmap1_nchw.shape
    rd = 2 * radius + 1
    num_disp = rd * rd
    axis = torch.arange(rd, device=fmap1_nchw.device, dtype=torch.float32) - radius

    fmap2_pyramid = [fmap2_nchw]
    for _ in range(1, num_levels):
        fmap2_pyramid.append(F.avg_pool2d(fmap2_pyramid[-1], 2, stride=2))

    f1 = fmap1_nchw.reshape(batch_size, channels, -1).permute(0, 2, 1)  # [B, P, C]

    corr_levels = []
    for level, fmap2_lvl in enumerate(fmap2_pyramid):
        h2 = fmap2_lvl.shape[2]
        w2 = fmap2_lvl.shape[3]
        f2 = fmap2_lvl.reshape(batch_size, channels, -1)  # [B, C, Q]

        coords_lvl = coords * (1.0 / (2**level))
        bx = _pixel_to_grid(coords_lvl[:, 0], w2)  # [B, H1, W1]
        by = _pixel_to_grid(coords_lvl[:, 1], h2)

        # Normalized displacement deltas along the two grid axes; the grid
        # is laid out [.., ix, iy] so the sampled channel is ix*rd + iy
        # (the RAFT x-major order of the kernels).
        dxn = (axis * 2.0 / w2).view(1, 1, 1, rd, 1)
        dyn = (axis * 2.0 / h2).view(1, 1, 1, 1, rd)

        tile_rows = _corr_tile_rows(width_1, h2, w2)
        corr_tiles = []
        prev_head = None
        for r0 in range(0, height_1, tile_rows):
            r1 = min(r0 + tile_rows, height_1)
            pt = (r1 - r0) * width_1
            g_x = bx[:, r0:r1].unsqueeze(-1).unsqueeze(-1) + dxn
            g_y = by[:, r0:r1].unsqueeze(-1).unsqueeze(-1) + dyn
            g_x = g_x.expand(batch_size, r1 - r0, width_1, rd, rd)
            g_y = g_y.expand(batch_size, r1 - r0, width_1, rd, rd)
            grid = torch.stack([g_x, g_y], dim=-1).reshape(batch_size * pt, rd, rd, 2)

            f1_tile = f1[:, r0 * width_1 : r1 * width_1]
            if prev_head is not None:
                # Serializes the tile lifetimes without changing the math:
                # the next matmul consumes an arithmetic function of the
                # previous tile's output, so it cannot be hoisted before
                # the previous tile's grid_sample. The chain value is a
                # single sample clamped to [-1, 1] (no reduction, so it
                # cannot overflow to inf in fp16), and adding it times
                # zero is bit-exact.
                f1_tile = f1_tile + prev_head * 0.0
            m_tile = torch.matmul(f1_tile, f2)  # [B, Pt, Q]
            m_tile = m_tile.reshape(batch_size * pt, 1, h2, w2)
            sampled = F.grid_sample(
                m_tile,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # [B*Pt, 1, rd, rd]
            sampled = sampled.reshape(batch_size, pt, num_disp)
            corr_tiles.append(sampled)
            prev_head = sampled[:, :1, :1].clamp(-1.0, 1.0)  # [B, 1, 1]

        corr = torch.cat(corr_tiles, dim=1)  # [B, P, D]
        corr = corr.reshape(batch_size, height_1, width_1, num_disp)
        corr_levels.append(corr.permute(0, 3, 1, 2))  # [B, D, H1, W1]

    return torch.stack(corr_levels, dim=1)


def spatial_corr_onnx_generic(
    input1: torch.Tensor, input2: torch.Tensor, radius: int
) -> torch.Tensor:
    """PWCNet-style single-level spatial correlation with standard ops.

    Args:
        input1: [B, C, H, W] feature map (first image).
        input2: [B, C, H, W] feature map (second image, same resolution).
        radius: half of the patch size (patch_size = 2*radius + 1).

    Returns:
        [B, patch_size, patch_size, H, W] correlation volume with the
        y-major displacement order ([b, dy_idx, dx_idx, h, w]), zero
        padding at the borders, matching the Triton spatial kernel.
    """
    batch_size, _, height, width = input1.shape
    patch_size = 2 * radius + 1

    f1 = input1.to(torch.float32)
    f2 = input2.to(torch.float32)

    # Zero-padded fmap2, shifted by each integer displacement: the sample
    # for displacement (dy, dx) at output pixel (h, w) is f2[h + dy, w + dx].
    f2_padded = F.pad(f2, (radius, radius, radius, radius), mode="constant", value=0.0)

    corr_disp = []
    for dy_idx in range(patch_size):
        for dx_idx in range(patch_size):
            shifted = f2_padded[
                :,
                :,
                dy_idx : dy_idx + height,
                dx_idx : dx_idx + width,
            ]
            corr_disp.append((f1 * shifted).sum(dim=1))

    # [B, patch_size * patch_size, H, W] with channel = dy_idx * ps + dx_idx.
    corr = torch.stack(corr_disp, dim=1)
    corr = corr.to(input1.dtype)
    return corr.view(batch_size, patch_size, patch_size, height, width)
