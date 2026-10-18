"""Tests for localcorr.onnx_generic (standard-ops ONNX formulations)."""

import pytest
import torch
import torch.nn.functional as F

import localcorr.onnx_generic as onnx_generic
from localcorr.onnx_generic import (
    _pixel_to_grid,
    localcorr_pyramid_onnx_generic,
    spatial_corr_onnx_generic,
)

from _reference import (
    make_coords,
    make_fmaps,
    pixel_to_grid,
    pyramid_localcorr_reference,
    spatial_corr_reference,
)

B, C, H, W = 2, 8, 16, 24
LEVELS, RADIUS = 3, 2


class TestPixelToGrid:
    def test_pixel_centers_sample_identity(self):
        image = torch.randn(2, 3, 8, 10)
        ys, xs = torch.meshgrid(
            torch.arange(8, dtype=torch.float32),
            torch.arange(10, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack(
            [pixel_to_grid(xs, 10), pixel_to_grid(ys, 8)], dim=-1
        ).repeat(2, 1, 1, 1)
        out = F.grid_sample(image, grid, align_corners=False)
        assert torch.allclose(out, image, atol=1e-5)

    def test_half_pixel_interpolates_neighbors(self):
        image = torch.zeros(1, 1, 1, 2)
        image[..., 0] = 1.0
        image[..., 1] = 3.0
        x_grid = pixel_to_grid(torch.tensor([0.5]), 2)  # halfway 0 and 1
        y_grid = pixel_to_grid(torch.tensor([0.0]), 1)  # center of H=1
        grid = torch.stack([x_grid.view(1, 1, 1), y_grid.view(1, 1, 1)], dim=-1)
        assert grid.shape == (1, 1, 1, 2)
        out = F.grid_sample(image, grid, align_corners=False)
        assert torch.allclose(out, torch.tensor([[[[2.0]]]]), atol=1e-6)

    def test_out_of_bounds_maps_beyond_edge(self):
        assert _pixel_to_grid(torch.tensor(-1.0), 10) < -1.0
        assert _pixel_to_grid(torch.tensor(10.0), 10) > 1.0
        # Border coordinates stay inside the [-1, 1] domain.
        assert -1.0 <= _pixel_to_grid(torch.tensor(0.0), 10) <= 1.0


class TestSpatialCorrGeneric:
    def test_matches_reference(self):
        i1, i2 = make_fmaps(batch=2, channels=3, height=10, width=12, seed=3)
        out = spatial_corr_onnx_generic(i1, i2, 3)
        ref = spatial_corr_reference(i1, i2, 3)
        assert out.shape == (2, 7, 7, 10, 12)
        assert out.dtype == i1.dtype
        assert torch.allclose(out, ref, atol=1e-4)

    def test_channel_order(self):
        # f1 has a single nonzero pixel at (2, 4); f2 a single nonzero at
        # (3, 4) = one row below. The response must light up at
        # [dy_idx=radius+1, dx_idx=radius] of the zero displacement.
        radius = 2
        f1 = torch.zeros(1, 1, 6, 8)
        f2 = torch.zeros(1, 1, 6, 8)
        f1[0, 0, 2, 4] = 1.0
        f2[0, 0, 3, 4] = 2.0
        out = spatial_corr_onnx_generic(f1, f2, radius)
        assert out.shape == (1, 5, 5, 6, 8)
        assert torch.allclose(out[0, 3, 2, 2, 4], torch.tensor(2.0))
        out[0, 3, 2, 2, 4] = 0.0
        assert out.abs().sum() == 0.0

    def test_preserves_dtype(self):
        i1, i2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=4)
        out = spatial_corr_onnx_generic(i1.half(), i2.half(), 1)
        assert out.dtype == torch.float16
        ref = spatial_corr_reference(i1, i2, 1)
        assert torch.allclose(out.float(), ref, atol=2e-2)


class TestPyramidCorrGeneric:
    def test_matches_reference(self):
        f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=5)
        coords = make_coords(batch=B, height=H, width=W, max_disp=3.0, seed=6)
        out = localcorr_pyramid_onnx_generic(f1, f2, coords, LEVELS, RADIUS)
        ref = pyramid_localcorr_reference(f1, f2, coords, LEVELS, RADIUS)
        assert out.shape == (B, LEVELS, 25, H, W)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_single_level_radius_one(self):
        f1, f2 = make_fmaps(batch=1, channels=4, height=8, width=8, seed=7)
        coords = make_coords(batch=1, height=8, width=8, max_disp=1.5, seed=8)
        out = localcorr_pyramid_onnx_generic(f1, f2, coords, 1, 1)
        ref = pyramid_localcorr_reference(f1, f2, coords, 1, 1)
        assert out.shape == (1, 1, 9, 8, 8)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_zero_displacement_channel_level0(self):
        # With identity coordinates the center channel of level 0 is the
        # per-pixel dot product f1.f2, computed with exact gathers.
        f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=9)
        coords = make_coords(batch=B, height=H, width=W, max_disp=0.0, seed=10)
        out = localcorr_pyramid_onnx_generic(f1, f2, coords, LEVELS, RADIUS)
        center = RADIUS * (2 * RADIUS + 1) + RADIUS
        ref = (f1 * f2).sum(dim=1)
        assert torch.allclose(out[:, 0, center], ref, atol=1e-3)

    def test_shifted_channel_level0(self):
        # Displacement (dy=+1, dx=0) at identity coords must equal the
        # product of f1 with f2 shifted one pixel down (zero padded).
        # RAFT channel order: d = ix * rd + iy.
        f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=11)
        coords = make_coords(batch=B, height=H, width=W, max_disp=0.0, seed=12)
        out = localcorr_pyramid_onnx_generic(f1, f2, coords, LEVELS, RADIUS)
        rd = 2 * RADIUS + 1
        channel = RADIUS * rd + (RADIUS + 1)
        # Sampling f2 at y+1: rows 0..H-2 pair f1[y] with f2[y+1]; the
        # last row falls outside and is zero padded.
        ref = torch.zeros(B, H, W)
        ref[:, :-1] = (f1[:, :, :-1] * f2[:, :, 1:]).sum(dim=1)
        assert torch.allclose(out[:, 0, channel], ref, atol=1e-3)

    def test_channel_order_one_hot(self):
        # f1 has a single nonzero pixel at (2, 4); f2 a single nonzero at
        # (3, 4), one row below. With identity coords the response must
        # appear at d = ix*rd + iy = radius*rd + (radius+1) and nowhere else.
        radius = 2
        f1 = torch.zeros(1, 1, 8, 10)
        f2 = torch.zeros(1, 1, 8, 10)
        f1[0, 0, 2, 4] = 1.0
        f2[0, 0, 3, 4] = 2.0
        coords = make_coords(batch=1, height=8, width=10, max_disp=0.0, seed=99)
        out = localcorr_pyramid_onnx_generic(f1, f2, coords, 1, radius)
        rd = 2 * radius + 1
        expected = radius * rd + (radius + 1)
        assert torch.allclose(out[0, 0, expected, 2, 4], torch.tensor(2.0))
        probe = out.clone()
        probe[0, 0, expected, 2, 4] = 0.0
        assert probe.abs().sum() == 0.0

    def test_far_outside_coords_give_zero(self):
        f1, f2 = make_fmaps(batch=1, channels=C, height=H, width=W, seed=13)
        coords = torch.full((1, 2, H, W), 1000.0)
        out = localcorr_pyramid_onnx_generic(f1, f2, coords, LEVELS, RADIUS)
        assert torch.count_nonzero(out) == 0

    def test_tiling_does_not_change_result(self, monkeypatch):
        # Force one fmap1 row per tile so the tile-chaining dependency is
        # exercised; the result must not change.
        f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=14)
        coords = make_coords(batch=B, height=H, width=W, max_disp=3.0, seed=15)
        full = localcorr_pyramid_onnx_generic(f1, f2, coords, LEVELS, RADIUS)
        monkeypatch.setattr(onnx_generic, "_CORR_TILE_MAX_ROWS", 1)
        monkeypatch.setattr(onnx_generic, "_CORR_TILE_MAX_BYTES", 1)
        tiled = localcorr_pyramid_onnx_generic(f1, f2, coords, LEVELS, RADIUS)
        assert torch.allclose(full, tiled, atol=1e-5)


class TestTileRows:
    def test_small_shapes_use_max_rows(self):
        assert (
            onnx_generic._corr_tile_rows(24, 8, 12) == onnx_generic._CORR_TILE_MAX_ROWS
        )

    def test_rows_are_bounded(self):
        rows = onnx_generic._corr_tile_rows(1280, 90, 160)
        assert 1 <= rows <= onnx_generic._CORR_TILE_MAX_ROWS

    def test_huge_rows_fall_back_to_one(self):
        assert onnx_generic._corr_tile_rows(100000, 4096, 4096) == 1
