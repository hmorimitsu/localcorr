// Fused multi-level variant of alt_cuda_corr_kernel.cu (forward path).
//
// All pyramid levels of RAFT's AlternateCorrBlock are evaluated in one
// kernel launch: one block iterates over the levels for its 32 consecutive
// output pixels. The level-l pyramid of fmap2 is packed in one buffer
// [B, C*sum_i h_i*w_i] (NHWC per level), which the ONNX graph produces with
// Slice + Concat nodes, so no plugin-side packing kernel is needed.
//
// Per-warp layout (fast path): a warp owns 4 consecutive pixels; lanes
// split the feature channels exactly as in the original per-level kernel
// (fmap1 and the sliding bilinear corner window live in registers, only 2
// corner vectors are loaded per displacement). The displacement window is
// further split by rows across... no: each warp processes its pixels one
// at a time, but a whole window ROW is computed with one batched
// butterfly reduction (rd independent partials reduced together, hiding
// the shuffle latency), and the outputs are staged in shared memory so the
// final global stores are coalesced (32 consecutive pixels per
// displacement channel).
//
// The fast-path kernel is templated on RADIUS so the ix loop fully unrolls
// and the compiler can pipeline the corner loads; radii other than the
// instantiated ones fall back to a generic-radius kernel with the same
// layout.

#include "localcorr_cuda_pyramid_kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define LCORR_WARP_SIZE 32
#define LCORR_WARPS_PER_BLOCK 4
#define LCORR_THREADS_PER_BLOCK (LCORR_WARPS_PER_BLOCK * LCORR_WARP_SIZE)
#define LCORR_PIXELS_PER_BLOCK (LCORR_WARPS_PER_BLOCK * LCORR_PIXELS_PER_WARP)
#define LCORR_PIXELS_PER_WARP 1

#define LCORR_MAX_LEVELS 8
#define LCORR_MAX_RADIUS 4


template <typename scalar_t, int VEC>
struct alignas(sizeof(scalar_t) * VEC) FusedChannelVec {
  scalar_t v[VEC];
};


// Fast path: C == 32*VEC*CITER, RADIUS compile-time. Triton-style layout:
// one warp per pixel (4 warps = 4 pixels per block), lanes split channels,
// predicated corner loads with 1-deep prefetch, whole displacement rows
// reduced with one interleaved butterfly, direct stores from the lanes that
// own each displacement (no shared memory, no __syncthreads).
template <typename scalar_t, int VEC, int CITER, int RADIUS, bool YMAJOR = false>
__global__ void __launch_bounds__(LCORR_THREADS_PER_BLOCK, 4)
localcorr_pyramid_kernel(
    const scalar_t* __restrict__ fmap1,        // [B, H1, W1, C]
    const scalar_t* __restrict__ fmap2_packed, // [B, C*sum_i h_i*w_i]
    const scalar_t* __restrict__ coords,       // [B, 2, H1, W1] (x, y)
    scalar_t* __restrict__ corr,               // [B, levels, RD*RD, H1, W1]
    int H1, int W1, int C, int levels)
{
  constexpr int RD = 2 * RADIUS + 1;
  constexpr int RD2 = RD * RD;
  constexpr int NELEM = VEC * CITER;   // channels per lane
  constexpr bool IS_HALF = (sizeof(scalar_t) == 2);
  constexpr int NH2 = IS_HALF ? (NELEM + 1) / 2 : 1;
  constexpr int NARR = IS_HALF ? NH2 : NELEM;  // corner/f1 array length
  using vec_t = FusedChannelVec<scalar_t, VEC>;

  const int b = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (LCORR_WARP_SIZE - 1);
  const int HW = H1 * W1;
  const int p = blockIdx.x * LCORR_PIXELS_PER_BLOCK + warp;
  const bool valid = p < HW;

  // Per-level sizes of the packed fmap2 buffer (batch-major layout:
  // [B, C*sum_i h_i*w_i], level i of batch b at b*P + C*sum_{j<i} h_j*w_j).
  int64_t f2_batch_elems = 0;
  {
    int lh = H1, lw = W1;
    for (int i = 0; i < levels; ++i) {
      f2_batch_elems += (int64_t)C * lh * lw;
      lh >>= 1;
      lw >>= 1;
    }
  }
  const scalar_t* f2_lvl = fmap2_packed + (int64_t)b * f2_batch_elems;
  int lvl_h = H1;
  int lvl_w = W1;

  // ---- Load fmap1 for this pixel into registers. ----
  float f1_reg[IS_HALF ? 1 : NELEM];
  __half2 f1_h2[IS_HALF ? NH2 : 1];
  if (valid) {
    const int h1 = p / W1;
    const int w1 = p - h1 * W1;
    const scalar_t* f1ptr =
        fmap1 + (((int64_t)b * H1 + h1) * W1 + w1) * C + lane * VEC;
    #pragma unroll
    for (int cit = 0; cit < CITER; ++cit) {
      vec_t t = *reinterpret_cast<const vec_t*>(
          f1ptr + cit * (LCORR_WARP_SIZE * VEC));
      if constexpr (IS_HALF) {
        #pragma unroll
        for (int k = 0; k < VEC / 2; ++k)
          f1_h2[cit * (VEC / 2) + k] =
              *reinterpret_cast<const __half2*>(&t.v[2 * k]);
      } else {
        #pragma unroll
        for (int k = 0; k < VEC; ++k)
          f1_reg[cit * VEC + k] = t.v[k];
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < NARR; ++i) {
      if constexpr (IS_HALF)
        f1_h2[i] = __float2half2_rn(0.f);
      else
        f1_reg[i] = 0.f;
    }
  }

  // Level-0 coordinates of this pixel.
  float cx = 0.f, cy = 0.f;
  if (valid) {
    const scalar_t* cptr = coords + ((int64_t)b * 2 * HW) + p;
    cx = static_cast<float>(cptr[0]);   // x
    cy = static_cast<float>(cptr[HW]);  // y
  }

  for (int lvl = 0; lvl < levels; ++lvl) {
    // Level coordinates are halved every level.
    const float clx = cx / (float)(1 << lvl);
    const float cly = cy / (float)(1 << lvl);

    const int H2 = lvl_h;
    const int W2 = lvl_w;

    const float x0f = floorf(clx);
    const float y0f = floorf(cly);
    const float dx = clx - x0f;
    const float dy = cly - y0f;
    const int xb = static_cast<int>(x0f) - RADIUS;
    const int yb = static_cast<int>(y0f) - RADIUS;

    const float wa = (1.f - dx) * (1.f - dy);
    const float wb = dx * (1.f - dy);
    const float wc = (1.f - dx) * dy;
    const float wd = dx * dy;
    const __half2 w2a = __float2half2_rn(wa);
    const __half2 w2b = __float2half2_rn(wb);
    const __half2 w2c = __float2half2_rn(wc);
    const __half2 w2d = __float2half2_rn(wd);

    scalar_t* out_lvl = corr + ((int64_t)b * levels + lvl) * RD2 * HW;

    // Predicated corner load: column x of row `row` (or zeros).
    // dst holds the CITER*VEC channels of this lane.
    auto loadCorner = [&](int64_t row, int x, bool vy, bool vx,
                          float (&fa)[IS_HALF ? 1 : NELEM],
                          __half2 (&ha)[IS_HALF ? NH2 : 1]) {
      if (vy && vx) {
        const scalar_t* q = f2_lvl + (row + x) * C + lane * VEC;
        #pragma unroll
        for (int cit = 0; cit < CITER; ++cit) {
          vec_t t = *reinterpret_cast<const vec_t*>(
              q + cit * (LCORR_WARP_SIZE * VEC));
          if constexpr (IS_HALF) {
            #pragma unroll
            for (int k = 0; k < VEC / 2; ++k)
              ha[cit * (VEC / 2) + k] =
                  *reinterpret_cast<const __half2*>(&t.v[2 * k]);
          } else {
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
              fa[cit * VEC + k] = t.v[k];
          }
        }
      } else {
        #pragma unroll
        for (int i = 0; i < NARR; ++i) {
          if constexpr (IS_HALF)
            ha[i] = __float2half2_rn(0.f);
          else
            fa[i] = 0.f;
        }
      }
    };

    #pragma unroll
    for (int iy = 0; iy < RD; ++iy) {
      const int y0 = yb + iy;
      const int y1 = y0 + 1;
      const bool vy0 = y0 >= 0 && y0 < H2;
      const bool vy1 = y1 >= 0 && y1 < H2;
      const int64_t row0 = vy0 ? (int64_t)y0 * W2 : 0;
      const int64_t row1 = vy1 ? (int64_t)y1 * W2 : 0;

      // Corner windows: a/c = column x0 (rows y0/y1), b/d = column x1.
      float ca_a[IS_HALF ? 1 : NELEM], cb_a[IS_HALF ? 1 : NELEM];
      float cc_a[IS_HALF ? 1 : NELEM], cd_a[IS_HALF ? 1 : NELEM];
      __half2 ha_a[IS_HALF ? NH2 : 1], hb_a[IS_HALF ? NH2 : 1];
      __half2 hc_a[IS_HALF ? NH2 : 1], hd_a[IS_HALF ? NH2 : 1];
      const bool vx0 = xb >= 0 && xb < W2;
      const bool vx1 = xb + 1 >= 0 && xb + 1 < W2;
      loadCorner(row0, xb, vy0, vx0, ca_a, ha_a);
      loadCorner(row1, xb, vy1, vx0, cc_a, hc_a);
      loadCorner(row0, xb + 1, vy0, vx1, cb_a, hb_a);
      loadCorner(row1, xb + 1, vy1, vx1, cd_a, hd_a);

      // Per-displacement partial dot products of this window row.
      float partial[RD];

      #pragma unroll
      for (int ix = 0; ix < RD; ++ix) {
        float sum = 0.f;
        if constexpr (IS_HALF) {
          #pragma unroll
          for (int i = 0; i < NH2; ++i) {
            __half2 f2v = __hfma2(ha_a[i], w2a, __hmul2(hb_a[i], w2b));
            f2v = __hfma2(hc_a[i], w2c, f2v);
            f2v = __hfma2(hd_a[i], w2d, f2v);
            __half2 pr = __hmul2(f1_h2[i], f2v);
            sum += __half2float(__low2half(pr)) +
                   __half2float(__high2half(pr));
          }
        } else {
          #pragma unroll
          for (int i = 0; i < NELEM; ++i) {
            const float f2v = wa * ca_a[i] + wb * cb_a[i] +
                              wc * cc_a[i] + wd * cd_a[i];
            sum = fmaf(f1_reg[i], f2v, sum);
          }
        }
        partial[ix] = sum;

        // Slide the window (the x1 corners become the next x0 corners) and
        // prefetch the next x1 column (1-deep prefetch).
        #pragma unroll
        for (int i = 0; i < NARR; ++i) {
          if constexpr (IS_HALF) {
            ha_a[i] = hb_a[i];
            hc_a[i] = hd_a[i];
          } else {
            ca_a[i] = cb_a[i];
            cc_a[i] = cd_a[i];
          }
        }
        const int xnext = xb + ix + 2;
        const bool vxn = xnext >= 0 && xnext < W2;
        loadCorner(row0, xnext, vy0, vxn, cb_a, hb_a);
        loadCorner(row1, xnext, vy1, vxn, cd_a, hd_a);
      }

      // Interleaved butterfly reduction of the RD partials (independent
      // shuffles, so the latency of one chain is paid once per row).
      #pragma unroll
      for (int off = LCORR_WARP_SIZE / 2; off > 0; off >>= 1) {
        #pragma unroll
        for (int ix = 0; ix < RD; ++ix)
          partial[ix] += __shfl_xor_sync(0xffffffffu, partial[ix], off);
      }

      // After the butterfly lane ix holds the reduced partial[ix]; store it
      // directly. Displacement index matches the original kernel:
      // d = iy + rd*ix (YMAJOR: d = ix + rd*iy).
      if (valid && lane < RD) {
        const int d = YMAJOR ? (lane + RD * iy) : (iy + RD * lane);
        out_lvl[(int64_t)d * HW + p] = static_cast<scalar_t>(partial[lane]);
      }
    }

    // Advance to the next level in the packed buffer.
    if (lvl + 1 < levels) {
      f2_lvl += (int64_t)C * lvl_h * lvl_w;
      lvl_h >>= 1;
      lvl_w >>= 1;
    }
  }
}



// Fast path A (8 lanes per pixel): lanes are split into groups of 8; each
// group owns one output pixel and each lane owns C/8 consecutive channels,
// so corner loads are wide (up to 32 B) and only 3 shuffle steps are needed
// per butterfly. Whole displacement rows are reduced with one interleaved
// butterfly; every lane of the group ends up with all reduced values, so
// the stores are strided by the group size. Requires C == 8*VEC*CITER with
// VEC*sizeof(scalar_t) <= 32 B and radius in [1, 4] (compile-time).
template <typename scalar_t, int VEC, int CITER, int RADIUS, bool YMAJOR = false>
__global__ void __launch_bounds__(LCORR_THREADS_PER_BLOCK, 4)
localcorr_pyramid_kernel_g8(
    const scalar_t* __restrict__ fmap1,        // [B, H1, W1, C]
    const scalar_t* __restrict__ fmap2_packed, // [B, C*sum_i h_i*w_i]
    const scalar_t* __restrict__ coords,       // [B, 2, H1, W1] (x, y)
    scalar_t* __restrict__ corr,               // [B, levels, RD*RD, H1, W1]
    int H1, int W1, int C, int levels)
{
  constexpr int LANES_PER_PIXEL = 8;
  constexpr int RD = 2 * RADIUS + 1;
  constexpr int RD2 = RD * RD;
  constexpr int NELEM = VEC * CITER;   // channels per lane
  constexpr bool IS_HALF = (sizeof(scalar_t) == 2);
  constexpr int NH2 = IS_HALF ? (NELEM + 1) / 2 : 1;
  constexpr int NARR = IS_HALF ? NH2 : NELEM;
  using vec_t = FusedChannelVec<scalar_t, VEC>;
  constexpr int PIXELS_PER_WARP = LCORR_WARP_SIZE / LANES_PER_PIXEL;

  const int b = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (LCORR_WARP_SIZE - 1);
  const int lgrp = lane / LANES_PER_PIXEL;       // pixel within the warp
  const int lhalf = lane % LANES_PER_PIXEL;      // lane within the group
  const int HW = H1 * W1;
  const int p =
      blockIdx.x * (LCORR_WARPS_PER_BLOCK * PIXELS_PER_WARP) +
      warp * PIXELS_PER_WARP + lgrp;
  const bool valid = p < HW;

  // Per-level sizes of the packed fmap2 buffer (batch-major layout).
  int64_t f2_batch_elems = 0;
  {
    int lh = H1, lw = W1;
    for (int i = 0; i < levels; ++i) {
      f2_batch_elems += (int64_t)C * lh * lw;
      lh >>= 1;
      lw >>= 1;
    }
  }
  const scalar_t* f2_lvl = fmap2_packed + (int64_t)b * f2_batch_elems;
  int lvl_h = H1;
  int lvl_w = W1;

  // ---- Load fmap1 for this pixel into registers. ----
  float f1_reg[IS_HALF ? 1 : NELEM];
  __half2 f1_h2[IS_HALF ? NH2 : 1];
  if (valid) {
    const int h1 = p / W1;
    const int w1 = p - h1 * W1;
    const scalar_t* f1ptr =
        fmap1 + (((int64_t)b * H1 + h1) * W1 + w1) * C + lhalf * NELEM;
    #pragma unroll
    for (int cit = 0; cit < CITER; ++cit) {
      vec_t t = *reinterpret_cast<const vec_t*>(
          f1ptr + cit * VEC);
      if constexpr (IS_HALF) {
        #pragma unroll
        for (int k = 0; k < VEC / 2; ++k)
          f1_h2[cit * (VEC / 2) + k] =
              *reinterpret_cast<const __half2*>(&t.v[2 * k]);
      } else {
        #pragma unroll
        for (int k = 0; k < VEC; ++k)
          f1_reg[cit * VEC + k] = t.v[k];
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < NARR; ++i) {
      if constexpr (IS_HALF)
        f1_h2[i] = __float2half2_rn(0.f);
      else
        f1_reg[i] = 0.f;
    }
  }

  // Level-0 coordinates of this pixel.
  float cx = 0.f, cy = 0.f;
  if (valid) {
    const scalar_t* cptr = coords + ((int64_t)b * 2 * HW) + p;
    cx = static_cast<float>(cptr[0]);   // x
    cy = static_cast<float>(cptr[HW]);  // y
  }

  for (int lvl = 0; lvl < levels; ++lvl) {
    const float clx = cx / (float)(1 << lvl);
    const float cly = cy / (float)(1 << lvl);

    const int H2 = lvl_h;
    const int W2 = lvl_w;

    const float x0f = floorf(clx);
    const float y0f = floorf(cly);
    const float dx = clx - x0f;
    const float dy = cly - y0f;
    const int xb = static_cast<int>(x0f) - RADIUS;
    const int yb = static_cast<int>(y0f) - RADIUS;

    const float wa = (1.f - dx) * (1.f - dy);
    const float wb = dx * (1.f - dy);
    const float wc = (1.f - dx) * dy;
    const float wd = dx * dy;
    const __half2 w2a = __float2half2_rn(wa);
    const __half2 w2b = __float2half2_rn(wb);
    const __half2 w2c = __float2half2_rn(wc);
    const __half2 w2d = __float2half2_rn(wd);

    scalar_t* out_lvl = corr + ((int64_t)b * levels + lvl) * RD2 * HW;

    // Predicated corner load: column x of row `row` (or zeros).
    auto loadCorner = [&](int64_t row, int x, bool vy, bool vx,
                          float (&fa)[IS_HALF ? 1 : NELEM],
                          __half2 (&ha)[IS_HALF ? NH2 : 1]) {
      if (vy && vx) {
        const scalar_t* q = f2_lvl + (row + x) * C + lhalf * NELEM;
        #pragma unroll
        for (int cit = 0; cit < CITER; ++cit) {
          vec_t t = *reinterpret_cast<const vec_t*>(
              q + cit * VEC);
          if constexpr (IS_HALF) {
            #pragma unroll
            for (int k = 0; k < VEC / 2; ++k)
              ha[cit * (VEC / 2) + k] =
                  *reinterpret_cast<const __half2*>(&t.v[2 * k]);
          } else {
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
              fa[cit * VEC + k] = t.v[k];
          }
        }
      } else {
        #pragma unroll
        for (int i = 0; i < NARR; ++i) {
          if constexpr (IS_HALF)
            ha[i] = __float2half2_rn(0.f);
          else
            fa[i] = 0.f;
        }
      }
    };

    #pragma unroll
    for (int iy = 0; iy < RD; ++iy) {
      const int y0 = yb + iy;
      const int y1 = y0 + 1;
      const bool vy0 = y0 >= 0 && y0 < H2;
      const bool vy1 = y1 >= 0 && y1 < H2;
      const int64_t row0 = vy0 ? (int64_t)y0 * W2 : 0;
      const int64_t row1 = vy1 ? (int64_t)y1 * W2 : 0;

      // Corner windows: a/c = column x0 (rows y0/y1), b/d = column x1.
      float ca_a[IS_HALF ? 1 : NELEM], cb_a[IS_HALF ? 1 : NELEM];
      float cc_a[IS_HALF ? 1 : NELEM], cd_a[IS_HALF ? 1 : NELEM];
      __half2 ha_a[IS_HALF ? NH2 : 1], hb_a[IS_HALF ? NH2 : 1];
      __half2 hc_a[IS_HALF ? NH2 : 1], hd_a[IS_HALF ? NH2 : 1];
      const bool vx0 = xb >= 0 && xb < W2;
      const bool vx1 = xb + 1 >= 0 && xb + 1 < W2;
      loadCorner(row0, xb, vy0, vx0, ca_a, ha_a);
      loadCorner(row1, xb, vy1, vx0, cc_a, hc_a);
      loadCorner(row0, xb + 1, vy0, vx1, cb_a, hb_a);
      loadCorner(row1, xb + 1, vy1, vx1, cd_a, hd_a);

      // Per-displacement partial dot products of this window row.
      float partial[RD];

      #pragma unroll
      for (int ix = 0; ix < RD; ++ix) {
        float sum = 0.f;
        if constexpr (IS_HALF) {
          #pragma unroll
          for (int i = 0; i < NH2; ++i) {
            __half2 f2v = __hfma2(ha_a[i], w2a, __hmul2(hb_a[i], w2b));
            f2v = __hfma2(hc_a[i], w2c, f2v);
            f2v = __hfma2(hd_a[i], w2d, f2v);
            __half2 pr = __hmul2(f1_h2[i], f2v);
            sum += __half2float(__low2half(pr)) +
                   __half2float(__high2half(pr));
          }
        } else {
          #pragma unroll
          for (int i = 0; i < NELEM; ++i) {
            const float f2v = wa * ca_a[i] + wb * cb_a[i] +
                              wc * cc_a[i] + wd * cd_a[i];
            sum = fmaf(f1_reg[i], f2v, sum);
          }
        }
        partial[ix] = sum;

        // Slide the window and prefetch the next x1 column.
        #pragma unroll
        for (int i = 0; i < NARR; ++i) {
          if constexpr (IS_HALF) {
            ha_a[i] = hb_a[i];
            hc_a[i] = hd_a[i];
          } else {
            ca_a[i] = cb_a[i];
            cc_a[i] = cd_a[i];
          }
        }
        const int xnext = xb + ix + 2;
        const bool vxn = xnext >= 0 && xnext < W2;
        loadCorner(row0, xnext, vy0, vxn, cb_a, hb_a);
        loadCorner(row1, xnext, vy1, vxn, cd_a, hd_a);
      }

      // Interleaved butterfly within the 8-lane group; afterwards every
      // lane of the group holds every reduced partial[ix].
      #pragma unroll
      for (int off = LANES_PER_PIXEL / 2; off > 0; off >>= 1) {
        #pragma unroll
        for (int ix = 0; ix < RD; ++ix)
          partial[ix] += __shfl_xor_sync(0xffffffffu, partial[ix], off);
      }

      // Strided direct stores (lane lhalf stores displacements
      // lhalf, lhalf+8, ...). Displacement index matches the original
      // kernel: d = iy + rd*ix (YMAJOR: d = ix + rd*iy).
      if (valid) {
        #pragma unroll
        for (int ix = lhalf; ix < RD; ix += LANES_PER_PIXEL) {
          const int d = YMAJOR ? (ix + RD * iy) : (iy + RD * ix);
          out_lvl[(int64_t)d * HW + p] = static_cast<scalar_t>(partial[ix]);
        }
      }
    }

    // Advance to the next level in the packed buffer.
    if (lvl + 1 < levels) {
      f2_lvl += (int64_t)C * lvl_h * lvl_w;
      lvl_h >>= 1;
      lvl_w >>= 1;
    }
  }
}


// ---------------------------------------------------------------------------
// Generic-radius fast path (same layout, runtime radius, ix loop not
// unrolled). Used for radii without a dedicated instantiation.
// ---------------------------------------------------------------------------
template <typename scalar_t, int VEC, int CITER, bool YMAJOR = false>
__global__ void __launch_bounds__(LCORR_THREADS_PER_BLOCK, 2)
localcorr_pyramid_kernel_genr(
    const scalar_t* __restrict__ fmap1,
    const scalar_t* __restrict__ fmap2_packed,
    const scalar_t* __restrict__ coords,
    scalar_t* __restrict__ corr,
    int H1, int W1, int C, int r, int levels, int dil)
{
  constexpr int NELEM = VEC * CITER;
  constexpr bool IS_HALF = (sizeof(scalar_t) == 2);
  constexpr int NH2 = IS_HALF ? (NELEM + 1) / 2 : 1;
  using vec_t = FusedChannelVec<scalar_t, VEC>;

  extern __shared__ float smem[];
  float* out_s = smem;

  const int rd = 2 * r + 1;
  const int RD2 = rd * rd;
  const int b = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (LCORR_WARP_SIZE - 1);
  const int HW = H1 * W1;
  const int p0 = blockIdx.x * LCORR_PIXELS_PER_BLOCK;

  int64_t f2_batch_elems = 0;
  {
    int lh = H1, lw = W1;
    for (int i = 0; i < levels; ++i) {
      f2_batch_elems += (int64_t)C * lh * lw;
      lh >>= 1;
      lw >>= 1;
    }
  }
  const scalar_t* f2_lvl = fmap2_packed + (int64_t)b * f2_batch_elems;
  int lvl_h = H1;
  int lvl_w = W1;

  for (int lvl = 0; lvl < levels; ++lvl) {
    const int H2 = lvl_h;
    const int W2 = lvl_w;
    const int pw0 = p0 + warp * LCORR_PIXELS_PER_WARP;

    for (int ip = 0; ip < LCORR_PIXELS_PER_WARP; ++ip) {
      const int p = pw0 + ip;
      if (p >= HW) break;
      const int h1 = p / W1;
      const int w1 = p - h1 * W1;

      float f1_reg[IS_HALF ? 1 : NELEM];
      __half2 f1_h2[IS_HALF ? NH2 : 1];
      {
        const scalar_t* f1ptr =
            fmap1 + (((int64_t)b * H1 + h1) * W1 + w1) * C + lane * VEC;
        #pragma unroll
        for (int cit = 0; cit < CITER; ++cit) {
          vec_t t = *reinterpret_cast<const vec_t*>(
              f1ptr + cit * (LCORR_WARP_SIZE * VEC));
          if constexpr (IS_HALF) {
            #pragma unroll
            for (int k = 0; k < VEC / 2; ++k)
              f1_h2[cit * (VEC / 2) + k] =
                  *reinterpret_cast<const __half2*>(&t.v[2 * k]);
          } else {
            #pragma unroll
            for (int k = 0; k < VEC; ++k)
              f1_reg[cit * VEC + k] = t.v[k];
          }
        }
      }

      const scalar_t* cptr = coords + ((int64_t)b * 2 * HW) + p;
      const float clx = static_cast<float>(cptr[0]) / (float)(1 << lvl);
      const float cly = static_cast<float>(cptr[HW]) / (float)(1 << lvl);

      const float x0f = floorf(clx);
      const float y0f = floorf(cly);
      const float dx = clx - x0f;
      const float dy = cly - y0f;
      const int xb = static_cast<int>(x0f) - r * dil;
      const int yb = static_cast<int>(y0f) - r * dil;

      const float wa = (1.f - dx) * (1.f - dy);
      const float wb = dx * (1.f - dy);
      const float wc = (1.f - dx) * dy;
      const float wd = dx * dy;
      const __half2 w2a = __float2half2_rn(wa);
      const __half2 w2b = __float2half2_rn(wb);
      const __half2 w2c = __float2half2_rn(wc);
      const __half2 w2d = __float2half2_rn(wd);

      for (int iy = 0; iy < rd; ++iy) {
        const int y0 = yb + dil * iy;
        const int y1 = y0 + 1;
        const bool vy0 = y0 >= 0 && y0 < H2;
        const bool vy1 = y1 >= 0 && y1 < H2;
        const int64_t row0 = vy0 ? (int64_t)y0 * W2 : 0;
        const int64_t row1 = vy1 ? (int64_t)y1 * W2 : 0;

        float col_a[IS_HALF ? 1 : NELEM], col_b[IS_HALF ? 1 : NELEM];
        float col_c[IS_HALF ? 1 : NELEM], col_d[IS_HALF ? 1 : NELEM];
        __half2 ha[IS_HALF ? NH2 : 1], hb[IS_HALF ? NH2 : 1];
        __half2 hc[IS_HALF ? NH2 : 1], hd[IS_HALF ? NH2 : 1];

        for (int ix = 0; ix < rd; ++ix) {
          const int x0 = xb + dil * ix;
          const int x1 = x0 + 1;
          const bool vx0 = x0 >= 0 && x0 < W2;
          const bool vx1 = x1 >= 0 && x1 < W2;

          // With dilation > 1 the x1 corners of one displacement are NOT
          // the x0 corners of the next (positions jump by dil), so the
          // sliding-window reuse only applies to unit dilation.
          if (ix == 0 || dil != 1) {
            #pragma unroll
            for (int cit = 0; cit < CITER; ++cit) {
              vec_t ta{}, tc{};
              if (vy0 && vx0)
                ta = *reinterpret_cast<const vec_t*>(
                    f2_lvl + (row0 + x0) * C + lane * VEC +
                    cit * (LCORR_WARP_SIZE * VEC));
              if (vy1 && vx0)
                tc = *reinterpret_cast<const vec_t*>(
                    f2_lvl + (row1 + x0) * C + lane * VEC +
                    cit * (LCORR_WARP_SIZE * VEC));
              if constexpr (IS_HALF) {
                #pragma unroll
                for (int k = 0; k < VEC / 2; ++k) {
                  ha[cit * (VEC / 2) + k] =
                      *reinterpret_cast<const __half2*>(&ta.v[2 * k]);
                  hc[cit * (VEC / 2) + k] =
                      *reinterpret_cast<const __half2*>(&tc.v[2 * k]);
                }
              } else {
                #pragma unroll
                for (int k = 0; k < VEC; ++k) {
                  col_a[cit * VEC + k] = ta.v[k];
                  col_c[cit * VEC + k] = tc.v[k];
                }
              }
            }
          } else {
            if constexpr (IS_HALF) {
              #pragma unroll
              for (int i = 0; i < NH2; ++i) {
                ha[i] = hb[i];
                hc[i] = hd[i];
              }
            } else {
              #pragma unroll
              for (int i = 0; i < NELEM; ++i) {
                col_a[i] = col_b[i];
                col_c[i] = col_d[i];
              }
            }
          }

          {
            #pragma unroll
            for (int cit = 0; cit < CITER; ++cit) {
              vec_t tb{}, td{};
              if (vy0 && vx1)
                tb = *reinterpret_cast<const vec_t*>(
                    f2_lvl + (row0 + x1) * C + lane * VEC +
                    cit * (LCORR_WARP_SIZE * VEC));
              if (vy1 && vx1)
                td = *reinterpret_cast<const vec_t*>(
                    f2_lvl + (row1 + x1) * C + lane * VEC +
                    cit * (LCORR_WARP_SIZE * VEC));
              if constexpr (IS_HALF) {
                #pragma unroll
                for (int k = 0; k < VEC / 2; ++k) {
                  hb[cit * (VEC / 2) + k] =
                      *reinterpret_cast<const __half2*>(&tb.v[2 * k]);
                  hd[cit * (VEC / 2) + k] =
                      *reinterpret_cast<const __half2*>(&td.v[2 * k]);
                }
              } else {
                #pragma unroll
                for (int k = 0; k < VEC; ++k) {
                  col_b[cit * VEC + k] = tb.v[k];
                  col_d[cit * VEC + k] = td.v[k];
                }
              }
            }
          }

          float sum = 0.f;
          if constexpr (IS_HALF) {
            #pragma unroll
            for (int i = 0; i < NH2; ++i) {
              __half2 f2v = __hfma2(ha[i], w2a, __hmul2(hb[i], w2b));
              f2v = __hfma2(hc[i], w2c, f2v);
              f2v = __hfma2(hd[i], w2d, f2v);
              __half2 pr = __hmul2(f1_h2[i], f2v);
              sum += __half2float(__low2half(pr)) +
                     __half2float(__high2half(pr));
            }
          } else {
            #pragma unroll
            for (int i = 0; i < NELEM; ++i) {
              const float f2v = wa * col_a[i] + wb * col_b[i] +
                                wc * col_c[i] + wd * col_d[i];
              sum = fmaf(f1_reg[i], f2v, sum);
            }
          }

          float partial = sum;
          #pragma unroll
          for (int off = LCORR_WARP_SIZE / 2; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, off);

          const int d = YMAJOR ? (ix + rd * iy) : (iy + rd * ix);
          if (lane == 0)
            out_s[d * LCORR_PIXELS_PER_BLOCK +
                  (warp * LCORR_PIXELS_PER_WARP + ip)] = partial;
        }
      }
    }
    __syncthreads();

    scalar_t* out_lvl = corr + ((int64_t)b * levels + lvl) * RD2 * HW;
    const int npix = min(LCORR_PIXELS_PER_BLOCK, HW - p0);
    for (int i = threadIdx.x; i < RD2 * LCORR_PIXELS_PER_BLOCK;
         i += LCORR_THREADS_PER_BLOCK) {
      const int d = i / LCORR_PIXELS_PER_BLOCK;
      const int q = i - d * LCORR_PIXELS_PER_BLOCK;
      if (q < npix)
        out_lvl[(int64_t)d * HW + p0 + q] =
            static_cast<scalar_t>(out_s[i]);
    }
    __syncthreads();

    if (lvl + 1 < levels) {
      f2_lvl += (int64_t)C * lvl_h * lvl_w;
      lvl_h >>= 1;
      lvl_w >>= 1;
    }
  }
}


// ---------------------------------------------------------------------------
// Backward kernel (any C / radius / dilation, all levels fused in one
// launch): one warp per output pixel, lanes stride the channels. fmap1
// and the fmap1 gradient accumulators are staged in shared memory as
// float; the scattered fmap2 gradients are accumulated with global fp32
// atomicAdd (the buffer must be zero-initialized; for f16 inputs it is
// an fp32 shadow buffer). Coordinates gradients are not computed (RAFT
// detaches coords). Port of the SYCL localcorr_pyramid_backward_kernel
// (sycl_pyramid/localcorr_cuda_pyramid_kernel.dp.cpp).
// ---------------------------------------------------------------------------
template <typename scalar_t, bool YMAJOR = false>
__global__ void __launch_bounds__(LCORR_THREADS_PER_BLOCK)
localcorr_pyramid_backward_kernel(
    const scalar_t* __restrict__ fmap1,        // [B, H1, W1, C]
    const scalar_t* __restrict__ fmap2_packed, // [B, C*sum_i h_i*w_i]
    const scalar_t* __restrict__ coords,       // [B, 2, H1, W1] (x, y)
    const scalar_t* __restrict__ corr_grad,    // [B, levels, RD*RD, H1, W1]
    scalar_t* __restrict__ fmap1_grad,         // [B, H1, W1, C]
    float* __restrict__ fmap2_grad,            // [B, C*sum_i h_i*w_i] f32
    int H1, int W1, int C, int r, int levels, int dil) {
  extern __shared__ float smem[];
  constexpr int PPB = LCORR_WARPS_PER_BLOCK;  // 1 pixel per warp
  float* f1_s = smem;                          // [PPB, C]
  float* f1g_s = smem + PPB * C;               // [PPB, C]

  const int rd = 2 * r + 1;
  const int RD2 = rd * rd;
  const int b = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (LCORR_WARP_SIZE - 1);
  const int HW = H1 * W1;
  const int p0 = blockIdx.x * PPB;
  const int npix = min(PPB, HW - p0);

  // Stage fmap1 for the pixels of this block (converted to float) and
  // zero the gradient accumulators.
  for (int i = threadIdx.x; i < PPB * C; i += LCORR_THREADS_PER_BLOCK) {
    const int q = i / C;
    const int cc = i - q * C;
    float v = 0.f;
    if (q < npix) {
      const int pq = p0 + q;
      const int hq = pq / W1;
      const int wq = pq - hq * W1;
      v = static_cast<float>(
          fmap1[(((int64_t)b * H1 + hq) * W1 + wq) * C + cc]);
    }
    f1_s[i] = v;
    f1g_s[i] = 0.f;
  }
  __syncthreads();

  int64_t f2_batch_elems = 0;
  {
    int lh = H1, lw = W1;
    for (int i = 0; i < levels; ++i) {
      f2_batch_elems += (int64_t)C * lh * lw;
      lh >>= 1;
      lw >>= 1;
    }
  }
  const scalar_t* f2_lvl = fmap2_packed + (int64_t)b * f2_batch_elems;
  float* f2g_lvl = fmap2_grad + (int64_t)b * f2_batch_elems;
  int lvl_h = H1;
  int lvl_w = W1;

  for (int lvl = 0; lvl < levels; ++lvl) {
    const int H2 = lvl_h;
    const int W2 = lvl_w;
    const int p = p0 + warp;  // one pixel per warp

    if (p < HW) {
      const int q = warp;
      const float* f1w = f1_s + q * C;
      float* f1gw = f1g_s + q * C;
      const scalar_t* lvl_base =
          corr_grad + ((int64_t)b * levels + lvl) * RD2 * HW;

      const scalar_t* cptr = coords + ((int64_t)b * 2 * HW) + p;
      const float clx = static_cast<float>(cptr[0]) / (float)(1 << lvl);
      const float cly = static_cast<float>(cptr[HW]) / (float)(1 << lvl);

      const float x0f = floorf(clx);
      const float y0f = floorf(cly);
      const float dx = clx - x0f;
      const float dy = cly - y0f;
      const int xb = static_cast<int>(x0f) - r * dil;
      const int yb = static_cast<int>(y0f) - r * dil;

      const float wa = (1.f - dx) * (1.f - dy);
      const float wb = dx * (1.f - dy);
      const float wc = (1.f - dx) * dy;
      const float wd = dx * dy;

      for (int iy = 0; iy < rd; ++iy) {
        const int y0 = yb + dil * iy;
        const int y1 = y0 + 1;
        const bool vy0 = y0 >= 0 && y0 < H2;
        const bool vy1 = y1 >= 0 && y1 < H2;
        const int64_t row0 = vy0 ? (int64_t)y0 * W2 : 0;
        const int64_t row1 = vy1 ? (int64_t)y1 * W2 : 0;

        for (int ix = 0; ix < rd; ++ix) {
          const int x0 = xb + dil * ix;
          const int x1 = x0 + 1;
          const bool vx0 = x0 >= 0 && x0 < W2;
          const bool vx1 = x1 >= 0 && x1 < W2;
          const bool la = vy0 && vx0;
          const bool lb = vy0 && vx1;
          const bool lc = vy1 && vx0;
          const bool ld = vy1 && vx1;

          // Displacement index: same channel order as the forward.
          const int d = YMAJOR ? (ix + rd * iy) : (iy + rd * ix);
          const float g =
              static_cast<float>(lvl_base[(int64_t)d * HW + p]);
          if (g == 0.f)
            continue;  // dense grads never skip; sparse ones do

          for (int c = lane; c < C; c += LCORR_WARP_SIZE) {
            const float f1v = f1w[c];
            float va = 0.f, vb = 0.f, vc = 0.f, vd = 0.f;
            if (la) va = static_cast<float>(f2_lvl[(row0 + x0) * C + c]);
            if (lb) vb = static_cast<float>(f2_lvl[(row0 + x1) * C + c]);
            if (lc) vc = static_cast<float>(f2_lvl[(row1 + x0) * C + c]);
            if (ld) vd = static_cast<float>(f2_lvl[(row1 + x1) * C + c]);

            // Gradient of this pixel's fmap1 row: sum_d g * f2_interp.
            const float f2v = wa * va + wb * vb + wc * vc + wd * vd;
            f1gw[c] += g * f2v;

            // Scatter g * f1 to the four bilinear corners (fp32 atomics).
            const float gf1 = g * f1v;
            if (la)
              atomicAdd(f2g_lvl + (row0 + x0) * C + c, gf1 * wa);
            if (lb)
              atomicAdd(f2g_lvl + (row0 + x1) * C + c, gf1 * wb);
            if (lc)
              atomicAdd(f2g_lvl + (row1 + x0) * C + c, gf1 * wc);
            if (ld)
              atomicAdd(f2g_lvl + (row1 + x1) * C + c, gf1 * wd);
          }
        }
      }
    }

    // Advance to the next level in the packed buffers.
    if (lvl + 1 < levels) {
      f2_lvl += (int64_t)C * lvl_h * lvl_w;
      f2g_lvl += (int64_t)C * lvl_h * lvl_w;
      lvl_h >>= 1;
      lvl_w >>= 1;
    }
  }
  // All warps finished accumulating before the cooperative store below.
  __syncthreads();

  // Store the fmap1 gradients (each accumulator element is written by
  // exactly one warp, so no atomics are needed).
  for (int i = threadIdx.x; i < PPB * C; i += LCORR_THREADS_PER_BLOCK) {
    const int q = i / C;
    const int cc = i - q * C;
    if (q < npix) {
      const int pq = p0 + q;
      const int hq = pq / W1;
      const int wq = pq - hq * W1;
      fmap1_grad[(((int64_t)b * H1 + hq) * W1 + wq) * C + cc] =
          static_cast<scalar_t>(f1g_s[i]);
    }
  }
}


template <typename scalar_t, bool YM>
static bool localcorr_pyramid_backward_dispatch(
    const void* fmap1,
    const void* fmap2_packed,
    const void* coords,
    const void* corr_grad,
    void* fmap1_grad,
    float* fmap2_grad,
    int H1,
    int W1,
    int C,
    int radius,
    int levels,
    int dil,
    int batch,
    cudaStream_t stream) {
  constexpr int PPB = LCORR_WARPS_PER_BLOCK;  // 1 pixel per warp
  const size_t HW = (size_t)H1 * W1;
  const dim3 blocks((HW + PPB - 1) / PPB, batch);
  const dim3 threads(LCORR_THREADS_PER_BLOCK);
  const size_t smem_bytes = (size_t)2 * PPB * C * sizeof(float);
  if (smem_bytes > 99 * 1024) {
    return false;
  }
  auto k = localcorr_pyramid_backward_kernel<scalar_t, YM>;
  if (smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(
        k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }
  k<<<blocks, threads, smem_bytes, stream>>>(
      static_cast<const scalar_t*>(fmap1),
      static_cast<const scalar_t*>(fmap2_packed),
      static_cast<const scalar_t*>(coords),
      static_cast<const scalar_t*>(corr_grad),
      static_cast<scalar_t*>(fmap1_grad),
      fmap2_grad,
      H1,
      W1,
      C,
      radius,
      levels,
      dil);
  return cudaGetLastError() == cudaSuccess;
}


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
    cudaStream_t stream) {
  const int HW = H1 * W1;
  if (HW <= 0 || B <= 0) {
    return true;
  }
  if (levels < 1 || levels > LCORR_MAX_LEVELS) {
    return false;
  }
  if (dilation < 1) {
    return false;
  }
  // Every level must have at least one pixel.
  for (int lvl = 0; lvl < levels; ++lvl) {
    if ((H1 >> lvl) < 1 || (W1 >> lvl) < 1) {
      return false;
    }
  }

  if (!isHalf) {
    if (ymajor) {
      return localcorr_pyramid_backward_dispatch<float, true>(
          fmap1, fmap2_packed, coords, corr_grad, fmap1_grad, fmap2_grad,
          H1, W1, C, radius, levels, dilation, B, stream);
    }
    return localcorr_pyramid_backward_dispatch<float, false>(
        fmap1, fmap2_packed, coords, corr_grad, fmap1_grad, fmap2_grad,
        H1, W1, C, radius, levels, dilation, B, stream);
  }
  if (ymajor) {
    return localcorr_pyramid_backward_dispatch<__half, true>(
        fmap1, fmap2_packed, coords, corr_grad, fmap1_grad, fmap2_grad,
        H1, W1, C, radius, levels, dilation, B, stream);
  }
  return localcorr_pyramid_backward_dispatch<__half, false>(
      fmap1, fmap2_packed, coords, corr_grad, fmap1_grad, fmap2_grad,
      H1, W1, C, radius, levels, dilation, B, stream);
}


// ---------------------------------------------------------------------------
// Generic fallback for any C: fmap1 staged in shared memory as float.
// ---------------------------------------------------------------------------
template <typename scalar_t, bool YMAJOR = false>
__global__ void __launch_bounds__(LCORR_THREADS_PER_BLOCK)
localcorr_pyramid_kernel_any(
    const scalar_t* __restrict__ fmap1,        // [B, H1, W1, C]
    const scalar_t* __restrict__ fmap2_packed, // [B, C*sum_i h_i*w_i]
    const scalar_t* __restrict__ coords,       // [B, 2, H1, W1] (x, y)
    scalar_t* __restrict__ corr,               // [B, levels, RD*RD, H1, W1]
    int H1, int W1, int C, int r, int levels, int dil)
{
  extern __shared__ float smem[];
  const int rd = 2 * r + 1;
  const int RD2 = rd * rd;
  float* out_s = smem;                          // [RD2, PIXELS_PER_BLOCK]
  float* f1_s = smem + RD2 * LCORR_PIXELS_PER_BLOCK;  // [PIXELS, C]

  const int b = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (LCORR_WARP_SIZE - 1);
  const int HW = H1 * W1;
  const int p0 = blockIdx.x * LCORR_PIXELS_PER_BLOCK;
  const int npix = min(LCORR_PIXELS_PER_BLOCK, HW - p0);

  // Stage fmap1 for the pixels of this block (converted to float).
  for (int i = threadIdx.x; i < LCORR_PIXELS_PER_BLOCK * C;
       i += LCORR_THREADS_PER_BLOCK) {
    const int q = i / C;
    const int cc = i - q * C;
    float v = 0.f;
    if (q < npix) {
      const int pq = p0 + q;
      const int hq = pq / W1;
      const int wq = pq - hq * W1;
      v = static_cast<float>(
          fmap1[(((int64_t)b * H1 + hq) * W1 + wq) * C + cc]);
    }
    f1_s[i] = v;
  }
  __syncthreads();

  int64_t f2_batch_elems = 0;
  {
    int lh = H1, lw = W1;
    for (int i = 0; i < levels; ++i) {
      f2_batch_elems += (int64_t)C * lh * lw;
      lh >>= 1;
      lw >>= 1;
    }
  }
  const scalar_t* f2_lvl = fmap2_packed + (int64_t)b * f2_batch_elems;
  int lvl_h = H1;
  int lvl_w = W1;

  for (int lvl = 0; lvl < levels; ++lvl) {
    const int H2 = lvl_h;
    const int W2 = lvl_w;
    const int pw0 = p0 + warp * LCORR_PIXELS_PER_WARP;

    for (int ip = 0; ip < LCORR_PIXELS_PER_WARP; ++ip) {
      const int p = pw0 + ip;
      if (p >= HW) break;
      const int q = warp * LCORR_PIXELS_PER_WARP + ip;
      const float* f1w = f1_s + q * C;

      const scalar_t* cptr = coords + ((int64_t)b * 2 * HW) + p;
      const float clx = static_cast<float>(cptr[0]) / (float)(1 << lvl);
      const float cly = static_cast<float>(cptr[HW]) / (float)(1 << lvl);

      const float x0f = floorf(clx);
      const float y0f = floorf(cly);
      const float dx = clx - x0f;
      const float dy = cly - y0f;
      const int xb = static_cast<int>(x0f) - r * dil;
      const int yb = static_cast<int>(y0f) - r * dil;

      const float wa = (1.f - dx) * (1.f - dy);
      const float wb = dx * (1.f - dy);
      const float wc = (1.f - dx) * dy;
      const float wd = dx * dy;

      for (int iy = 0; iy < rd; ++iy) {
        const int y0 = yb + dil * iy;
        const int y1 = y0 + 1;
        const bool vy0 = y0 >= 0 && y0 < H2;
        const bool vy1 = y1 >= 0 && y1 < H2;
        const int64_t row0 = vy0 ? (int64_t)y0 * W2 : 0;
        const int64_t row1 = vy1 ? (int64_t)y1 * W2 : 0;

        for (int ix = 0; ix < rd; ++ix) {
          const int x0 = xb + dil * ix;
          const int x1 = x0 + 1;
          const bool vx0 = x0 >= 0 && x0 < W2;
          const bool vx1 = x1 >= 0 && x1 < W2;
          const bool la = vy0 && vx0;
          const bool lb = vy0 && vx1;
          const bool lc = vy1 && vx0;
          const bool ld = vy1 && vx1;

          float partial = 0.f;
          for (int c = lane; c < C; c += LCORR_WARP_SIZE) {
            float va = 0.f, vb = 0.f, vc = 0.f, vd = 0.f;
            if (la) va = static_cast<float>(f2_lvl[(row0 + x0) * C + c]);
            if (lb) vb = static_cast<float>(f2_lvl[(row0 + x1) * C + c]);
            if (lc) vc = static_cast<float>(f2_lvl[(row1 + x0) * C + c]);
            if (ld) vd = static_cast<float>(f2_lvl[(row1 + x1) * C + c]);
            partial += f1w[c] * (wa * va + wb * vb + wc * vc + wd * vd);
          }

          #pragma unroll
          for (int off = LCORR_WARP_SIZE / 2; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, off);

          const int d = YMAJOR ? (ix + rd * iy) : (iy + rd * ix);
          if (lane == 0)
            out_s[d * LCORR_PIXELS_PER_BLOCK + q] = partial;
        }
      }
    }
    __syncthreads();

    scalar_t* out_lvl = corr + ((int64_t)b * levels + lvl) * RD2 * HW;
    for (int i = threadIdx.x; i < RD2 * LCORR_PIXELS_PER_BLOCK;
         i += LCORR_THREADS_PER_BLOCK) {
      const int d = i / LCORR_PIXELS_PER_BLOCK;
      const int q = i - d * LCORR_PIXELS_PER_BLOCK;
      if (q < npix)
        out_lvl[(int64_t)d * HW + p0 + q] =
            static_cast<scalar_t>(out_s[i]);
    }
    __syncthreads();

    if (lvl + 1 < levels) {
      f2_lvl += (int64_t)C * lvl_h * lvl_w;
      lvl_h >>= 1;
      lvl_w >>= 1;
    }
  }
}


template <typename scalar_t, bool YM>
static bool localcorr_pyramid_dispatch(
    const void* fmap1,
    const void* fmap2_packed,
    const void* coords,
    void* corr,
    int H1,
    int W1,
    int C,
    int radius,
    int levels,
    int dil,
    int vec,
    int citer,
    bool use_g8,
    int g8_vec,
    int g8_citer,
    size_t out_smem,
    size_t smem_bytes,
    dim3 blocks,
    dim3 threads,
    cudaStream_t stream) {
  auto launch_any = [&]() {
    if (smem_bytes > 48 * 1024) {
      cudaFuncSetAttribute(
          localcorr_pyramid_kernel_any<scalar_t, YM>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    localcorr_pyramid_kernel_any<scalar_t, YM>
        <<<blocks, threads, smem_bytes, stream>>>(
            static_cast<const scalar_t*>(fmap1),
            static_cast<const scalar_t*>(fmap2_packed),
            static_cast<const scalar_t*>(coords),
            static_cast<scalar_t*>(corr), H1, W1, C, radius, levels, dil);
  };

  auto set_smem = [&](auto kernel, size_t smem) {
    if (smem > 48 * 1024) {
      cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
  };

  // 8-lanes-per-pixel fast path: C == 8*VEC*CITER with VEC <= 16
  // elements (32 B) and radius in [1, 4]. Roughly 2x faster than the
  // 32-lane layout because the corner loads are wide and the butterfly
  // needs only 3 shuffle steps.
  auto launch_g8_r = [&](auto gvconst, auto gcconst, auto rconst) {
    constexpr int G8VEC = decltype(gvconst)::value;
    constexpr int G8CITER = decltype(gcconst)::value;
    constexpr int G8RADIUS = decltype(rconst)::value;
    constexpr int G8_PPB = LCORR_WARPS_PER_BLOCK * (LCORR_WARP_SIZE / 8);
    const dim3 g8_blocks(
        (H1 * W1 + G8_PPB - 1) / G8_PPB, blocks.y);
    auto k = localcorr_pyramid_kernel_g8<scalar_t, G8VEC, G8CITER, G8RADIUS, YM>;
    k<<<g8_blocks, threads, 0, stream>>>(
        static_cast<const scalar_t*>(fmap1),
        static_cast<const scalar_t*>(fmap2_packed),
        static_cast<const scalar_t*>(coords),
        static_cast<scalar_t*>(corr), H1, W1, C, levels);
  };

  // g8 VEC/CITER dispatch: VEC*CITER == C/8 channels per lane.
  auto launch_g8 = [&](auto rconst) {
    auto g8_ci = [&](auto gvconst) {
      switch (g8_citer) {
        case 1:
          launch_g8_r(gvconst, std::integral_constant<int, 1>{}, rconst);
          break;
        case 2:
          launch_g8_r(gvconst, std::integral_constant<int, 2>{}, rconst);
          break;
        case 3:
          launch_g8_r(gvconst, std::integral_constant<int, 3>{}, rconst);
          break;
        default:
          launch_g8_r(gvconst, std::integral_constant<int, 4>{}, rconst);
          break;
      }
    };
    switch (g8_vec) {
      case 16:
        g8_ci(std::integral_constant<int, 16>{});
        break;
      case 8:
        g8_ci(std::integral_constant<int, 8>{});
        break;
      case 4:
        g8_ci(std::integral_constant<int, 4>{});
        break;
      case 2:
        g8_ci(std::integral_constant<int, 2>{});
        break;
      default:
        g8_ci(std::integral_constant<int, 1>{});
        break;
    }
  };

  // Radius-templated fast path for common radii, generic-radius kernel
  // otherwise. VEC and CITER are dispatched by the caller. Dilation > 1
  // always takes the generic-radius kernel (the fast paths assume unit
  // displacement steps).
  auto launch_vc = [&](auto vconst, auto cconst) -> bool {
    constexpr int VEC = decltype(vconst)::value;
    constexpr int CITER = decltype(cconst)::value;
    if (dil != 1 || radius > LCORR_MAX_RADIUS) {
      auto k = localcorr_pyramid_kernel_genr<scalar_t, VEC, CITER, YM>;
      set_smem(k, out_smem);
      k<<<blocks, threads, out_smem, stream>>>(
          static_cast<const scalar_t*>(fmap1),
          static_cast<const scalar_t*>(fmap2_packed),
          static_cast<const scalar_t*>(coords),
          static_cast<scalar_t*>(corr), H1, W1, C, radius, levels, dil);
      return cudaGetLastError() == cudaSuccess;
    } else if (radius == 3) {
      if (use_g8) {
        launch_g8(std::integral_constant<int, 3>{});
        return cudaGetLastError() == cudaSuccess;
      }
      auto k = localcorr_pyramid_kernel<scalar_t, VEC, CITER, 3, YM>;
      k<<<blocks, threads, 0, stream>>>(
          static_cast<const scalar_t*>(fmap1),
          static_cast<const scalar_t*>(fmap2_packed),
          static_cast<const scalar_t*>(coords),
          static_cast<scalar_t*>(corr), H1, W1, C, levels);
      return cudaGetLastError() == cudaSuccess;
    } else if (radius == 4) {
      if (use_g8) {
        launch_g8(std::integral_constant<int, 4>{});
        return cudaGetLastError() == cudaSuccess;
      }
      auto k = localcorr_pyramid_kernel<scalar_t, VEC, CITER, 4, YM>;
      k<<<blocks, threads, 0, stream>>>(
          static_cast<const scalar_t*>(fmap1),
          static_cast<const scalar_t*>(fmap2_packed),
          static_cast<const scalar_t*>(coords),
          static_cast<scalar_t*>(corr), H1, W1, C, levels);
      return cudaGetLastError() == cudaSuccess;
    } else if (radius == 2) {
      if (use_g8) {
        launch_g8(std::integral_constant<int, 2>{});
        return cudaGetLastError() == cudaSuccess;
      }
      auto k = localcorr_pyramid_kernel<scalar_t, VEC, CITER, 2, YM>;
      k<<<blocks, threads, 0, stream>>>(
          static_cast<const scalar_t*>(fmap1),
          static_cast<const scalar_t*>(fmap2_packed),
          static_cast<const scalar_t*>(coords),
          static_cast<scalar_t*>(corr), H1, W1, C, levels);
      return cudaGetLastError() == cudaSuccess;
    } else if (radius == 1) {
      if (use_g8) {
        launch_g8(std::integral_constant<int, 1>{});
        return cudaGetLastError() == cudaSuccess;
      }
      auto k = localcorr_pyramid_kernel<scalar_t, VEC, CITER, 1, YM>;
      k<<<blocks, threads, 0, stream>>>(
          static_cast<const scalar_t*>(fmap1),
          static_cast<const scalar_t*>(fmap2_packed),
          static_cast<const scalar_t*>(coords),
          static_cast<scalar_t*>(corr), H1, W1, C, levels);
      return cudaGetLastError() == cudaSuccess;
    } else {
      auto k = localcorr_pyramid_kernel_genr<scalar_t, VEC, CITER, YM>;
      set_smem(k, out_smem);
      k<<<blocks, threads, out_smem, stream>>>(
          static_cast<const scalar_t*>(fmap1),
          static_cast<const scalar_t*>(fmap2_packed),
          static_cast<const scalar_t*>(coords),
          static_cast<scalar_t*>(corr), H1, W1, C, radius, levels, dil);
      return cudaGetLastError() == cudaSuccess;
    }
  };

  auto launch_v = [&](auto vconst) -> bool {
    switch (citer) {
      case 1:
        return launch_vc(vconst, std::integral_constant<int, 1>{});
      case 2:
        return launch_vc(vconst, std::integral_constant<int, 2>{});
      case 3:
        return launch_vc(vconst, std::integral_constant<int, 3>{});
      default:
        return launch_vc(vconst, std::integral_constant<int, 4>{});
    }
  };

  switch (vec) {
    case 16:
      launch_v(std::integral_constant<int, 16>{});
      break;
    case 8:
      launch_v(std::integral_constant<int, 8>{});
      break;
    case 4:
      launch_v(std::integral_constant<int, 4>{});
      break;
    case 2:
      launch_v(std::integral_constant<int, 2>{});
      break;
    case 1:
      launch_v(std::integral_constant<int, 1>{});
      break;
    default:
      launch_any();
      break;
  }

  return cudaGetLastError() == cudaSuccess;
}


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
    cudaStream_t stream) {
  const int HW = H1 * W1;
  if (HW <= 0 || B <= 0) {
    return true;
  }
  if (levels < 1 || levels > LCORR_MAX_LEVELS) {
    return false;
  }
  if (dilation < 1) {
    return false;
  }
  // Every level must have at least one pixel.
  for (int lvl = 0; lvl < levels; ++lvl) {
    if ((H1 >> lvl) < 1 || (W1 >> lvl) < 1) {
      return false;
    }
  }

  const int rd = 2 * radius + 1;
  const size_t out_smem =
      (size_t)rd * rd * LCORR_PIXELS_PER_BLOCK * sizeof(float);
  const int elem_size = isHalf ? 2 : 4;

  // 8-lanes-per-pixel fast path constraints: C divisible by 8, at most 64
  // channels per lane (register budget) and a compile-time radius.
  const int g8_max_vec = 32 / elem_size;
  int g8_vec = 0, g8_citer = 0;
  if (C % 8 == 0) {
    const int n = C / 8;  // channels per lane
    if (n >= 1 && n <= 64) {
      int v = 1;
      while (v < g8_max_vec && v * 2 <= n && n % (v * 2) == 0) v *= 2;
      const int ci = n / v;
      if (ci <= 4) {
        g8_vec = v;
        g8_citer = ci;
      }
    }
  }
  // The fp16 g8 path works on half2 pairs, so it needs VEC >= 2.
  if (elem_size == 2 && g8_vec == 1) {
    g8_vec = 0;
    g8_citer = 0;
  }
  const bool use_g8 =
      g8_vec > 0 && radius >= 1 && radius <= LCORR_MAX_RADIUS && dilation == 1;

  // Fast path: C == 32*VEC*CITER (same constraints as the per-level
  // kernel; see alt_cuda_corr_kernel.cu).
  const int max_vec = 32 / elem_size;
  int vec = 0, citer = 0;
  if (C % LCORR_WARP_SIZE == 0) {
    const int n = C / LCORR_WARP_SIZE;  // channels per lane = VEC*CITER
    if (n >= 1 && n <= 16) {
      int v = 1;
      while (v < max_vec && v * 2 <= n && n % (v * 2) == 0) v *= 2;
      const int ci = n / v;
      if (ci <= 4) {
        vec = v;
        citer = ci;
      }
    }
  }
  // The fp16 fast path works on half2 pairs, so it needs VEC >= 2.
  if (elem_size == 2 && vec == 1) {
    vec = 0;
    citer = 0;
  }

  const size_t smem_bytes =
      vec > 0 ? out_smem
              : out_smem + (size_t)LCORR_PIXELS_PER_BLOCK * C * sizeof(float);
  if (smem_bytes > 99 * 1024) {
    return false;
  }

  const dim3 blocks((HW + LCORR_PIXELS_PER_BLOCK - 1) / LCORR_PIXELS_PER_BLOCK,
                    B);
  const dim3 threads(LCORR_THREADS_PER_BLOCK);

  if (!isHalf) {
    if (ymajor) {
      return localcorr_pyramid_dispatch<float, true>(
          fmap1, fmap2_packed, coords, corr, H1, W1, C, radius, levels,
          dilation, vec, citer, use_g8, g8_vec, g8_citer, out_smem,
          smem_bytes, blocks, threads, stream);
    }
    return localcorr_pyramid_dispatch<float, false>(
        fmap1, fmap2_packed, coords, corr, H1, W1, C, radius, levels,
        dilation, vec, citer, use_g8, g8_vec, g8_citer, out_smem,
        smem_bytes, blocks, threads, stream);
  }
  if (ymajor) {
    return localcorr_pyramid_dispatch<__half, true>(
        fmap1, fmap2_packed, coords, corr, H1, W1, C, radius, levels,
        dilation, vec, citer, use_g8, g8_vec, g8_citer, out_smem,
        smem_bytes, blocks, threads, stream);
  }
  return localcorr_pyramid_dispatch<__half, false>(
      fmap1, fmap2_packed, coords, corr, H1, W1, C, radius, levels,
      dilation, vec, citer, use_g8, g8_vec, g8_citer, out_smem,
      smem_bytes, blocks, threads, stream);
}
