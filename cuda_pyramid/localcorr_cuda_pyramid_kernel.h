#ifndef LOCALCORR_CUDA_PYRAMID_KERNEL_H
#define LOCALCORR_CUDA_PYRAMID_KERNEL_H

#include <cuda_runtime.h>

// Fused multi-level variant of launchAltCudaCorrForward: all pyramid levels
// of RAFT's AlternateCorrBlock are evaluated in a single kernel launch.
//
//   corr[b, lvl, iy + rd*ix, h1, w1]
//     = sum_c fmap1[b, h1, w1, c] *
//       bilinear(fmap2_lvl[b, :, :, :], coords[b, :, h1, w1] / 2^lvl
//                + (iy - r, ix - r))
//
// fmap1:        [B, H1, W1, C] (contiguous, NHWC)
// fmap2_packed: [B, P] (contiguous, batch-major) with
//               P = C * sum_{i=0}^{levels-1} h_i * w_i, h_i = H1 >> i,
//               w_i = W1 >> i (true integer-halved dims). Batch b holds all
//               its levels contiguously: level i starts at element
//               b * P + C * sum_{j<i} h_j * w_j (NHWC per level).
// coords:       [B, 2, H1, W1] (contiguous, NCHW, (x, y) order, level-0 scale)
// corr:         [B, levels, RD*RD, H1, W1] with RD = 2*radius+1
//               (pre-allocated)
//
// isHalf selects the half-precision kernels; all pointers must then be
// half-precision too.
//
// ymajor selects the ordering of the RD*RD displacement channels of the
// output (see alt_cuda_corr_kernel.h for details).
//
// levels must be in [1, 8] and H1 >> (levels-1) and W1 >> (levels-1) must
// both be >= 1.
//
// dilation >= 1 is the step between consecutive displacement positions of
// the correlation window (dilation_patch of the spatial correlation
// samplers; 1 = unit steps). dilation > 1 uses the generic-radius kernel.
//
// Returns true on success, false if the launch is impossible (e.g. shared
// memory overflow for the given radius/channel count).
extern "C" bool launchLocalCorrCudaPyramidForward(
    const void* fmap1,
    const void* fmap2_packed,
    const void* coords,
    void* corr,
    int B,
    int H1,
    int W1,
    int C,
    int radius,
    int levels,
    int dilation,
    bool isHalf,
    bool ymajor,
    cudaStream_t stream);

// Backward of launchLocalCorrCudaPyramidForward for the same tensor
// contract (fmap1/fmap2_packed/coords as above, corr_grad shaped like
// corr). Computes
//   fmap1_grad[b, h1, w1, c]
//     = sum_{lvl, iy, ix} corr_grad[b, lvl, d, h1, w1] *
//       bilinear(fmap2_lvl, coords/2^lvl + (iy-r, ix-r))[c]
//   fmap2_grad_packed[corner, c]
//     = sum over (b, lvl, p, iy, ix) whose bilinear window covers the
//       corner: corr_grad * fmap1[b, p, c] * corner_weight
// (d indexed with the same ymajor convention as the forward) via one
// fused kernel launch: one warp per output pixel, lanes striding the
// channels, all levels in one pass, and global fp32 atomic adds for the
// scattered fmap2 gradients.
//
// fmap1_grad is written in the input dtype (scalar_t, one exclusive
// owner per pixel, no atomics). fmap2_grad MUST point to a
// zero-initialized float32 buffer with fmap2_packed's element count:
// for f32 inputs it is the final gradient of the same dtype, for f16
// inputs it is an fp32 shadow accumulator that the caller casts to
// half afterwards (fp32 accumulation, no reliance on native fp16
// atomics).
//
// Gradients for coords are not computed (RAFT detaches the sampling
// coordinates); the caller returns zeros.
//
// Returns true on success, false if the launch is impossible (e.g.
// shared-memory overflow for the given channel count).
extern "C" bool launchLocalCorrCudaPyramidBackward(
    const void* fmap1,
    const void* fmap2_packed,
    const void* coords,
    const void* corr_grad,
    void* fmap1_grad,
    float* fmap2_grad,
    int B,
    int H1,
    int W1,
    int C,
    int radius,
    int levels,
    int dilation,
    bool isHalf,
    bool ymajor,
    cudaStream_t stream);

#endif  // LOCALCORR_CUDA_PYRAMID_KERNEL_H
