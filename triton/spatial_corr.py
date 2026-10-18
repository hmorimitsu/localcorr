import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _correlation_forward_kernel(
    f1_ptr,
    f2_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    stride_f1_b,
    stride_f1_c,
    stride_f1_h,
    stride_f1_w,
    stride_f2_b,
    stride_f2_c,
    stride_f2_h,
    stride_f2_w,
    stride_out_b,
    stride_out_d1,
    stride_out_d2,
    stride_out_h,
    stride_out_w,
    md,
    dil,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # 3D grid: (Batch, H_blocks, W_blocks)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Compute spatial offsets for this block
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    h_mask = h_offs < H
    w_mask = w_offs < W

    # Iterate over the correlation window [-md, md]
    for dy in range(-md, md + 1):
        for dx in range(-md, md + 1):
            # Accumulator for the dot product over channels
            acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

            # Loop over channels in blocks
            for c in range(0, C, BLOCK_C):
                c_offs = c + tl.arange(0, BLOCK_C)
                c_mask = c_offs < C

                # Load f1 block
                f1_ptrs = f1_ptr + (
                    pid_b * stride_f1_b
                    + c_offs[None, None, :] * stride_f1_c
                    + h_offs[:, None, None] * stride_f1_h
                    + w_offs[None, :, None] * stride_f1_w
                )
                f1_mask = (
                    h_mask[:, None, None]
                    & w_mask[None, :, None]
                    & c_mask[None, None, :]
                )
                f1_val = tl.load(f1_ptrs, mask=f1_mask, other=0.0)

                # Load f2 block (shifted by dilated dy, dx)
                h2_offs = h_offs + dy * dil
                w2_offs = w_offs + dx * dil
                h2_mask = (h2_offs >= 0) & (h2_offs < H)
                w2_mask = (w2_offs >= 0) & (w2_offs < W)

                f2_ptrs = f2_ptr + (
                    pid_b * stride_f2_b
                    + c_offs[None, None, :] * stride_f2_c
                    + h2_offs[:, None, None] * stride_f2_h
                    + w2_offs[None, :, None] * stride_f2_w
                )
                f2_mask = (
                    h2_mask[:, None, None]
                    & w2_mask[None, :, None]
                    & c_mask[None, None, :]
                )
                f2_val = tl.load(f2_ptrs, mask=f2_mask, other=0.0)

                # Dot product and accumulate
                acc += tl.sum(f1_val * f2_val, axis=2)

            # Store result in output tensor
            dy_idx = dy + md
            dx_idx = dx + md
            out_ptrs = out_ptr + (
                pid_b * stride_out_b
                + dy_idx * stride_out_d1
                + dx_idx * stride_out_d2
                + h_offs[:, None] * stride_out_h
                + w_offs[None, :] * stride_out_w
            )
            out_mask = h_mask[:, None] & w_mask[None, :]
            tl.store(out_ptrs, acc, mask=out_mask)


class TritonSpatialCorrelationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f1, f2, md, dilation_patch=1):
        B, C, H, W = f1.shape
        patch_size = 2 * md + 1

        # Triton requires contiguous tensors to calculate memory strides correctly
        f1 = f1.contiguous()
        f2 = f2.contiguous()

        # Output shape matches PyTorch-Correlation-extension [B, patch_size, patch_size, H, W]
        out = torch.empty(
            (B, patch_size, patch_size, H, W), device=f1.device, dtype=f1.dtype
        )

        BLOCK_H = 16
        BLOCK_W = 16
        BLOCK_C = 64  # Efficient chunk size for channels (PWC-Net uses max 196)

        grid = lambda meta: (
            B,
            triton.cdiv(H, meta["BLOCK_H"]),
            triton.cdiv(W, meta["BLOCK_W"]),
        )

        _correlation_forward_kernel[grid](
            f1,
            f2,
            out,
            B,
            C,
            H,
            W,
            f1.stride(0),
            f1.stride(1),
            f1.stride(2),
            f1.stride(3),
            f2.stride(0),
            f2.stride(1),
            f2.stride(2),
            f2.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            out.stride(4),
            md,
            dilation_patch,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
            BLOCK_C=BLOCK_C,
        )

        ctx.save_for_backward(f1, f2)
        ctx.md = md
        ctx.dil = dilation_patch
        return out

    @staticmethod
    def backward(ctx, grad_out):
        f1, f2 = ctx.saved_tensors
        md = ctx.md
        dil = ctx.dil
        B, C, H, W = f1.shape

        grad_f1 = torch.zeros_like(f1)
        grad_f2 = torch.zeros_like(f2)
        grad_out = grad_out.contiguous()

        # Memory efficient native-PyTorch backward loop
        # (O(H*W) extra memory instead of materializing massive unfolded tensors)
        for dy in range(-md, md + 1):
            for dx in range(-md, md + 1):
                sy, sx = dy * dil, dx * dil  # actual pixel shift
                dy_idx, dx_idx = dy + md, dx + md
                g_o = grad_out[:, dy_idx, dx_idx, :, :]  # [B, H, W]

                # Determine valid intersection bounds
                h1_start, h1_end = max(0, -sy), min(H, H - sy)
                w1_start, w1_end = max(0, -sx), min(W, W - sx)
                h2_start, h2_end = max(0, sy), min(H, H + sy)
                w2_start, w2_end = max(0, sx), min(W, W + sx)

                if h1_start < h1_end and w1_start < w1_end:
                    g_o_slice = g_o[:, h1_start:h1_end, w1_start:w1_end].unsqueeze(
                        1
                    )  # [B, 1, H', W']

                    # Accumulate gradients
                    f2_slice = f2[:, :, h2_start:h2_end, w2_start:w2_end]
                    grad_f1[:, :, h1_start:h1_end, w1_start:w1_end] += (
                        g_o_slice * f2_slice
                    )

                    f1_slice = f1[:, :, h1_start:h1_end, w1_start:w1_end]
                    grad_f2[:, :, h2_start:h2_end, w2_start:w2_end] += (
                        g_o_slice * f1_slice
                    )

        return grad_f1, grad_f2, None, None


class TritonSpatialCorrelationSampler(nn.Module):
    def __init__(
        self,
        kernel_size=1,
        patch_size=9,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=1,
    ):
        super().__init__()
        assert (
            kernel_size == 1
        ), "This Triton implementation is optimized strictly for kernel_size=1"
        assert stride == 1, "Only stride=1 is supported"
        assert dilation == 1, "Only dilation=1 is supported"

        self.md = patch_size // 2
        self.dilation_patch = dilation_patch

    def forward(self, f1, f2):
        return TritonSpatialCorrelationFunction.apply(
            f1, f2, self.md, self.dilation_patch
        )
