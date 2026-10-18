// Torch XPU extension wrapper for the localcorr SYCL pyramid kernel.
//
// Exposes the fused correlation kernel (ported from
// localcorr/cuda_pyramid/localcorr_cuda_pyramid_kernel.cu via dpct) to
// PyTorch as a torch.utils.cpp_extension.SyclExtension. Inputs are torch
// XPU tensors; the kernel runs on torch's current XPU stream (SYCL queue)
// so it is ordered with the surrounding ops without synchronization.
//
// Layout contract (identical to the CUDA kernel):
//   fmap1:        [B, H1, W1, C]        contiguous NHWC, f32 or f16
//   fmap2_packed: [B, C*sum_i h_i*w_i]  packed NHWC levels (levels=1 for
//                                        PWCNet: just NHWC flattened)
//   coords:       [B, 2, H1, W1]        planar, x in channel 0
//   corr:         [B, levels, RD*RD, H1, W1]

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include "localcorr_cuda_pyramid_kernel.h"

torch::Tensor localcorr_pyramid_forward(
    torch::Tensor fmap1,        // [B, H1, W1, C] NHWC
    torch::Tensor fmap2_packed, // [B, P]
    torch::Tensor coords,       // [B, 2, H1, W1]
    int64_t radius,
    int64_t levels,
    int64_t dilation,
    bool ymajor) {
  TORCH_CHECK(fmap1.is_xpu() && fmap2_packed.is_xpu() && coords.is_xpu(),
              "all inputs must be XPU tensors");
  TORCH_CHECK(fmap1.is_contiguous() && fmap2_packed.is_contiguous() &&
                  coords.is_contiguous(),
              "all inputs must be contiguous");
  const auto et = fmap1.scalar_type();
  TORCH_CHECK(et == torch::kFloat32 || et == torch::kFloat16,
              "fmap1 must be f32 or f16, got ", et);
  TORCH_CHECK(coords.scalar_type() == et,
              "coords dtype must match fmap1 dtype");

  const int B = (int)fmap1.size(0);
  const int H1 = (int)fmap1.size(1);
  const int W1 = (int)fmap1.size(2);
  const int C = (int)fmap1.size(3);

  const int rd = (int)(2 * radius + 1);
  auto out = torch::empty(
      {B, levels, rd * rd, H1, W1},
      torch::TensorOptions().dtype(et).device(fmap1.device()));

  sycl::queue& q = c10::xpu::getCurrentXPUStream();
  const bool ok = launchLocalCorrCudaPyramidForward(
      fmap1.data_ptr(), fmap2_packed.data_ptr(), coords.data_ptr(),
      out.data_ptr(), B, H1, W1, C, (int)radius, (int)levels, (int)dilation,
      et == torch::kFloat16, ymajor, &q);
  TORCH_CHECK(ok, "localcorr SYCL kernel launch failed");
  return out;
}

// Backward of localcorr_pyramid_forward. Returns the gradients of
// (fmap1, fmap2_packed, coords); the coords gradient is zeros (RAFT
// detaches the sampling coordinates). For f16 inputs the scattered
// fmap2 gradient is accumulated in an fp32 shadow buffer and cast to
// half at the end (more accurate than fp16 atomics and free of any
// native-fp16-atomics requirement).
std::vector<torch::Tensor> localcorr_pyramid_backward(
    torch::Tensor fmap1,        // [B, H1, W1, C] NHWC
    torch::Tensor fmap2_packed, // [B, P]
    torch::Tensor coords,       // [B, 2, H1, W1]
    torch::Tensor corr_grad,    // [B, levels, RD*RD, H1, W1]
    int64_t radius, int64_t levels, int64_t dilation, bool ymajor) {
  TORCH_CHECK(fmap1.is_xpu() && fmap2_packed.is_xpu() && coords.is_xpu() &&
                  corr_grad.is_xpu(),
              "all inputs must be XPU tensors");
  TORCH_CHECK(fmap1.is_contiguous() && fmap2_packed.is_contiguous() &&
                  coords.is_contiguous() && corr_grad.is_contiguous(),
              "all inputs must be contiguous");
  const auto et = fmap1.scalar_type();
  TORCH_CHECK(et == torch::kFloat32 || et == torch::kFloat16,
              "fmap1 must be f32 or f16, got ", et);
  TORCH_CHECK(coords.scalar_type() == et && corr_grad.scalar_type() == et,
              "coords/corr_grad dtype must match fmap1 dtype");
  TORCH_CHECK(fmap1.dim() == 4, "fmap1 must be [B, H1, W1, C]");
  TORCH_CHECK(fmap2_packed.dim() == 2, "fmap2_packed must be [B, P]");

  const int B = (int)fmap1.size(0);
  const int H1 = (int)fmap1.size(1);
  const int W1 = (int)fmap1.size(2);
  const int C = (int)fmap1.size(3);
  const int rd = (int)(2 * radius + 1);
  TORCH_CHECK(coords.dim() == 4 && coords.size(0) == B &&
                  coords.size(1) == 2 && coords.size(2) == H1 &&
                  coords.size(3) == W1,
              "coords must be [B, 2, H1, W1] matching fmap1");
  TORCH_CHECK(corr_grad.dim() == 5 && corr_grad.size(0) == B &&
                  corr_grad.size(1) == levels &&
                  corr_grad.size(2) == rd * rd && corr_grad.size(3) == H1 &&
                  corr_grad.size(4) == W1,
              "corr_grad must be [B, levels, RD*RD, H1, W1]");
  TORCH_CHECK(fmap2_packed.size(0) == B,
              "fmap2_packed batch must match fmap1");

  sycl::queue& q = c10::xpu::getCurrentXPUStream();
  auto fmap1_grad = torch::empty_like(fmap1);
  torch::Tensor fmap2_grad;
  if (et == torch::kFloat16) {
    // fp32 shadow accumulator, cast to half afterwards.
    auto f2g32 = torch::zeros(
        fmap2_packed.sizes(),
        fmap2_packed.options().dtype(torch::kFloat32));
    const bool ok = launchLocalCorrCudaPyramidBackward(
        fmap1.data_ptr(), fmap2_packed.data_ptr(), coords.data_ptr(),
        corr_grad.data_ptr(), fmap1_grad.data_ptr(),
        f2g32.data_ptr<float>(), B, H1, W1, C, (int)radius, (int)levels,
        (int)dilation, /*isHalf=*/true, ymajor, &q);
    TORCH_CHECK(ok, "localcorr SYCL backward kernel launch failed");
    fmap2_grad = f2g32.to(et);
  } else {
    fmap2_grad = torch::zeros_like(fmap2_packed);
    const bool ok = launchLocalCorrCudaPyramidBackward(
        fmap1.data_ptr(), fmap2_packed.data_ptr(), coords.data_ptr(),
        corr_grad.data_ptr(), fmap1_grad.data_ptr(),
        fmap2_grad.data_ptr<float>(), B, H1, W1, C, (int)radius, (int)levels,
        (int)dilation, /*isHalf=*/false, ymajor, &q);
    TORCH_CHECK(ok, "localcorr SYCL backward kernel launch failed");
  }
  // RAFT detaches the sampling coordinates; align with the Triton
  // backward, which also returns zeros here.
  auto coords_grad = torch::zeros_like(coords);
  return {fmap1_grad, fmap2_grad, coords_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &localcorr_pyramid_forward,
        "fused local correlation pyramid (SYCL, XPU)");
  m.def("backward", &localcorr_pyramid_backward,
        "fused local correlation pyramid backward (SYCL, XPU)");
}
