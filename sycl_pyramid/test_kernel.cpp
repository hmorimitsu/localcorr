// Correctness test for the SYCL localcorr pyramid kernel.
//
// Compares launchLocalCorrCudaPyramidForward against a plain double
// precision CPU reference for many shape/dtype/dispatch configurations
// (fast paths, g8 path, generic-radius path and the any-C fallback).
//
// Build (see build.sh):
//   icpx -fsycl -std=c++20 -O2 test_kernel.cpp -o test_kernel \
//       -L build -llocalcorr_sycl_pyramid -Wl,-rpath,$PWD/build
// Run:
//   ./build/test_kernel            # default SYCL device
//   ONEAPI_DEVICE_SELECTOR=cpu ./build/test_kernel
//   ONEAPI_DEVICE_SELECTOR=gpu ./build/test_kernel

#include "localcorr_cuda_pyramid_kernel.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

template <typename T>
static void reference(const std::vector<T> &fmap1,
                      const std::vector<T> &fmap2_packed,
                      const std::vector<T> &coords, std::vector<double> &corr,
                      int B, int H1, int W1, int C, int radius, int levels,
                      int dil, bool ymajor) {
  const int rd = 2 * radius + 1;
  const int HW = H1 * W1;
  std::vector<int64_t> lvl_off(levels + 1, 0);
  int lh = H1, lw = W1;
  for (int i = 0; i < levels; ++i) {
    lvl_off[i + 1] = lvl_off[i] + (int64_t)C * lh * lw;
    lh >>= 1;
    lw >>= 1;
  }
  const int64_t batch_elems = lvl_off[levels];
  corr.assign((size_t)B * levels * rd * rd * HW, 0.0);
  for (int b = 0; b < B; ++b) {
    for (int p = 0; p < HW; ++p) {
      const int h1 = p / W1;
      const int w1 = p - h1 * W1;
      const double cx = (double)coords[((size_t)b * 2 * HW) + p];
      const double cy = (double)coords[((size_t)b * 2 * HW) + HW + p];
      lh = H1;
      lw = W1;
      for (int lvl = 0; lvl < levels; ++lvl) {
        const int H2 = lh, W2 = lw;
        const double clx = cx / (double)(1 << lvl);
        const double cly = cy / (double)(1 << lvl);
        const double x0f = std::floor(clx);
        const double y0f = std::floor(cly);
        const double dx = clx - x0f;
        const double dy = cly - y0f;
        const int xb = (int)x0f - radius * dil;
        const int yb = (int)y0f - radius * dil;
        const T *f2 = fmap2_packed.data() + (int64_t)b * batch_elems +
                      lvl_off[lvl];
        const T *f1 = fmap1.data() + (((size_t)b * H1 + h1) * W1 + w1) * C;
        for (int iy = 0; iy < rd; ++iy) {
          for (int ix = 0; ix < rd; ++ix) {
            const int x0 = xb + dil * ix, x1 = x0 + 1;
            const int y0 = yb + dil * iy, y1 = y0 + 1;
            const bool vx0 = x0 >= 0 && x0 < W2, vx1 = x1 >= 0 && x1 < W2;
            const bool vy0 = y0 >= 0 && y0 < H2, vy1 = y1 >= 0 && y1 < H2;
            double acc = 0;
            for (int c = 0; c < C; ++c) {
              double a = 0, bq = 0, cq = 0, dq = 0;
              if (vy0 && vx0) a = (double)f2[((int64_t)y0 * W2 + x0) * C + c];
              if (vy0 && vx1) bq = (double)f2[((int64_t)y0 * W2 + x1) * C + c];
              if (vy1 && vx0) cq = (double)f2[((int64_t)y1 * W2 + x0) * C + c];
              if (vy1 && vx1) dq = (double)f2[((int64_t)y1 * W2 + x1) * C + c];
              const double f2v = (1.0 - dx) * (1.0 - dy) * a +
                                 dx * (1.0 - dy) * bq + (1.0 - dx) * dy * cq +
                                 dx * dy * dq;
              acc += (double)f1[c] * f2v;
            }
            const int d = ymajor ? (ix + rd * iy) : (iy + rd * ix);
            corr[(((size_t)b * levels + lvl) * rd * rd + d) * HW + p] = acc;
          }
        }
        lh >>= 1;
        lw >>= 1;
      }
    }
  }
}

struct TestCase {
  int B, H1, W1, C, radius, levels, dil;
  bool ymajor;
  const char *note;
};

template <typename T>
static bool run_case(const TestCase &tc, std::mt19937 &rng, sycl::queue &q,
                     bool is_half) {
  const int HW = tc.H1 * tc.W1;
  const int rd = 2 * tc.radius + 1;
  int64_t f2_elems = 0;
  {
    int lh = tc.H1, lw = tc.W1;
    for (int i = 0; i < tc.levels; ++i) {
      f2_elems += (int64_t)tc.C * lh * lw;
      lh >>= 1;
      lw >>= 1;
    }
  }
  std::vector<T> fmap1((size_t)tc.B * HW * tc.C), f2(f2_elems * tc.B),
      coords((size_t)tc.B * 2 * HW);
  std::vector<T> corr_out((size_t)tc.B * tc.levels * rd * rd * HW);
  std::normal_distribution<float> dist(0.f, 0.25f);
  for (auto &v : fmap1) v = T(dist(rng));
  for (auto &v : f2) v = T(dist(rng));
  // Random coords slightly outside the image to exercise the border
  // predication of the bilinear sampling.
  std::uniform_real_distribution<float> cx_dist(-2.f, float(tc.W1) + 2.f);
  std::uniform_real_distribution<float> cy_dist(-2.f, float(tc.H1) + 2.f);
  for (int b = 0; b < tc.B; ++b)
    for (int p = 0; p < HW; ++p) {
      coords[((size_t)b * 2 * HW) + p] = T(cx_dist(rng));
      coords[((size_t)b * 2 * HW) + HW + p] = T(cy_dist(rng));
    }

  // Device-accessible buffers (USM shared).
  T *d_f1 = sycl::malloc_shared<T>(fmap1.size(), q);
  T *d_f2 = sycl::malloc_shared<T>(f2.size(), q);
  T *d_co = sycl::malloc_shared<T>(coords.size(), q);
  T *d_out = sycl::malloc_shared<T>(corr_out.size(), q);
  q.memcpy(d_f1, fmap1.data(), fmap1.size() * sizeof(T));
  q.memcpy(d_f2, f2.data(), f2.size() * sizeof(T));
  q.memcpy(d_co, coords.data(), coords.size() * sizeof(T));
  q.wait();

  const bool ok = launchLocalCorrCudaPyramidForward(
      d_f1, d_f2, d_co, d_out, tc.B, tc.H1, tc.W1, tc.C, tc.radius, tc.levels,
      tc.dil, is_half, tc.ymajor, &q);
  if (!ok) {
    std::printf("  LAUNCH FAILED  %s\n", tc.note);
    return false;
  }
  q.wait();
  q.memcpy(corr_out.data(), d_out, corr_out.size() * sizeof(T)).wait();
  sycl::free(d_f1, q);
  sycl::free(d_f2, q);
  sycl::free(d_co, q);
  sycl::free(d_out, q);

  std::vector<double> ref;
  reference(fmap1, f2, coords, ref, tc.B, tc.H1, tc.W1, tc.C, tc.radius,
            tc.levels, tc.dil, tc.ymajor);

  double max_err = 0, ref_max = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double e = std::fabs((double)corr_out[i] - ref[i]);
    if (e > max_err) max_err = e;
    if (std::fabs(ref[i]) > ref_max) ref_max = std::fabs(ref[i]);
  }
  const double tol = is_half ? 5e-2 : 2e-3;
  const bool pass = max_err <= tol * std::max(ref_max, 1e-6) + (is_half ? 2e-3 : 1e-5);
  std::printf("  %-7s %-6s B=%d %dx%d C=%-3d r=%d lv=%d dil=%d ym=%d | %s | "
              "max_err=%.2e ref_max=%.2e\n",
              pass ? "PASS" : "FAIL", is_half ? "half" : "float", tc.B, tc.H1,
              tc.W1, tc.C, tc.radius, tc.levels, tc.dil, (int)tc.ymajor,
              tc.note, max_err, ref_max);
  return pass;
}

int main() {
  auto q = sycl::queue(sycl::default_selector_v,
                       sycl::property::queue::in_order());
  std::printf("SYCL device: %s\n\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());

  std::mt19937 rng(42);
  const std::vector<TestCase> cases = {
      // PWCNet-style single level (ymajor=1) over the channel counts used
      // by the pyramid levels of PWCDCNet.
      {2, 20, 24, 32, 4, 1, 1, true, "pwc c32 (32-lane fast/g8)"},
      {2, 20, 24, 64, 4, 1, 1, true, "pwc c64 (fast/g8)"},
      {2, 20, 24, 96, 4, 1, 1, true, "pwc c96 (g8)"},
      {2, 20, 24, 128, 4, 1, 1, true, "pwc c128 (g8)"},
      {2, 20, 24, 196, 4, 1, 1, true, "pwc c196 (any-C fallback)"},
      {2, 20, 24, 196, 3, 1, 1, true, "pwc c196 r3 (any)"},
      {1, 16, 16, 128, 4, 1, 1, true, "tiny pwc"},
      // RAFT-style multi level (ymajor=0).
      {2, 32, 40, 128, 4, 4, 1, false, "raft c128 4lv (g8)"},
      {1, 32, 40, 256, 4, 2, 1, false, "raft c256 (g8 vec16)"},
      {2, 24, 30, 32, 1, 3, 1, false, "raft c32 r1"},
      {2, 24, 30, 32, 2, 3, 1, false, "raft c32 r2"},
      {2, 24, 30, 32, 3, 3, 1, false, "raft c32 r3 (fast vec1)"},
      {1, 32, 40, 48, 2, 2, 1, false, "c48 (any-C fallback)"},
      {1, 32, 40, 512, 3, 2, 1, false, "c512"},
      // Generic-radius/dilation path.
      {1, 20, 24, 128, 2, 1, 2, true, "dilation=2 (genr)"},
      {1, 20, 24, 128, 3, 2, 3, false, "dilation=3 multi-lvl (genr)"},
      {1, 20, 24, 96, 0, 1, 1, false, "radius=0 (genr)"},
  };

  int fails = 0;
  for (const auto &tc : cases) {
    if (!run_case<float>(tc, rng, q, false)) fails++;
    if (!run_case<sycl::half>(tc, rng, q, true)) fails++;
  }
  std::printf("\n%d failure(s)\n", fails);
  return fails == 0 ? 0 : 1;
}
