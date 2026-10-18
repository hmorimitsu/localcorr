"""Triton-based local correlation kernels (localcorr_triton).

These kernels are JIT-compiled by Triton at first use, so no build step is
required - only `pip install triton`. They provide fast forward passes and
backward passes (training support) for:

- pyramid_corr: RAFT-style multi-level local correlation around sampled
  coordinates (per-level and fused-all-levels variants).
- spatial_corr: PWCNet-style single-level spatial correlation
  (equivalent to SpatialCorrelationSampler with kernel_size=1 and zero
  padding).
"""

from .pyramid_corr import LocalCorrTriton, LocalCorrTritonFused
from .spatial_corr import (
    TritonSpatialCorrelationFunction,
    TritonSpatialCorrelationSampler,
)

__all__ = [
    "LocalCorrTriton",
    "LocalCorrTritonFused",
    "TritonSpatialCorrelationFunction",
    "TritonSpatialCorrelationSampler",
]
