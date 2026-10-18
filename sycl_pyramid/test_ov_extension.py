"""Test the localcorr SYCL OpenVINO custom-op extension with a synthetic ONNX.

Builds an ONNX model with one custom_ops::LocalCorrCudaPyramid node,
loads it with core.add_extension(<extension .so>), runs it and compares
against a NumPy reference.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# Must run before importing openvino: loads the oneAPI OpenCL runtime so
# the SYCL extension can find the OpenCL CPU platform.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import localcorr.openvino_ext as lcorr_ov

import openvino as ov

EXT_SO = Path(__file__).resolve().parent / "build" / "liblocalcorr_sycl_pyramid_ov.so"


def build_reference(fmap1, fmap2, coords, radius, ymajor):
    # Single level, identity-style coords: corr[b, 0, d, h, w]
    B, H, W, C = fmap1.shape
    rd = 2 * radius + 1
    out = np.zeros((B, 1, rd * rd, H, W), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                cx = coords[b, 0, h, w]
                cy = coords[b, 1, h, w]
                x0f, y0f = np.floor(cx), np.floor(cy)
                dx, dy = cx - x0f, cy - y0f
                xb, yb = int(x0f) - radius, int(y0f) - radius
                wa = (1 - dx) * (1 - dy)
                wb = dx * (1 - dy)
                wc = (1 - dx) * dy
                wd = dx * dy
                for iy in range(rd):
                    y0, y1 = yb + iy, yb + iy + 1
                    for ix in range(rd):
                        x0, x1 = xb + ix, xb + ix + 1
                        zeros = np.zeros(C, dtype=np.float32)
                        va = (
                            fmap2[b, y0, x0] if (0 <= y0 < H and 0 <= x0 < W) else zeros
                        )
                        vb = (
                            fmap2[b, y0, x1] if (0 <= y0 < H and 0 <= x1 < W) else zeros
                        )
                        vc = (
                            fmap2[b, y1, x0] if (0 <= y1 < H and 0 <= x0 < W) else zeros
                        )
                        vd = (
                            fmap2[b, y1, x1] if (0 <= y1 < H and 0 <= x1 < W) else zeros
                        )
                        f2v = wa * va + wb * vb + wc * vc + wd * vd
                        acc = float(np.dot(fmap1[b, h, w], f2v))
                        d = (ix + rd * iy) if ymajor else (iy + rd * ix)
                        out[b, 0, d, h, w] = acc
    return out


def main():
    rng = np.random.default_rng(0)
    B, H, W, C, radius = 2, 12, 16, 128, 4
    rd = 2 * radius + 1
    ymajor = 1
    levels = 1

    fmap1 = rng.standard_normal((B, H, W, C)).astype(np.float32) * 0.3
    fmap2 = rng.standard_normal((B, H, W, C)).astype(np.float32) * 0.3
    # Random coords around the identity to exercise the bilinear path.
    coords = (
        np.stack(
            [
                np.broadcast_to(np.arange(W, dtype=np.float32), (H, W))
                + rng.standard_normal((H, W)) * 0.5,
                np.broadcast_to(np.arange(H, dtype=np.float32).reshape(H, 1), (H, W))
                + rng.standard_normal((H, W)) * 0.5,
            ],
            axis=0,
        )[None]
        .repeat(B, axis=0)
        .astype(np.float32)
        .copy()
    )
    fmap2_packed = fmap2.reshape(B, -1)

    node = helper.make_node(
        "LocalCorrCudaPyramid",
        ["fmap1", "fmap2_packed", "coords"],
        ["corr"],
        name="localcorr_0",
        domain="custom_ops",
        radius=radius,
        levels=levels,
        ymajor=ymajor,
        dilation=1,
    )
    graph = helper.make_graph(
        [node],
        "localcorr_test",
        inputs=[
            helper.make_tensor_value_info("fmap1", TensorProto.FLOAT, [B, H, W, C]),
            helper.make_tensor_value_info(
                "fmap2_packed", TensorProto.FLOAT, [B, H * W * C]
            ),
            helper.make_tensor_value_info("coords", TensorProto.FLOAT, [B, 2, H, W]),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "corr", TensorProto.FLOAT, [B, levels, rd * rd, H, W]
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx_path = "/tmp/opencode/localcorr_ext_test.onnx"
    onnx.save(model, onnx_path)

    core = ov.Core()
    core.add_extension(str(EXT_SO))
    ov_model = core.read_model(onnx_path)
    print(
        "Ops:",
        [
            op.get_type_name()
            for op in ov_model.get_ops()
            if op.get_type_name() == "LocalCorrCudaPyramid"
        ],
    )
    compiled = core.compile_model(ov_model, "CPU")
    result = compiled(
        {
            "fmap1": fmap1,
            "fmap2_packed": fmap2_packed,
            "coords": coords,
        }
    )[0]

    ref = build_reference(fmap1, fmap2, coords, radius, ymajor)
    diff = np.abs(result - ref).max()
    scale = np.abs(ref).max()
    print(f"max_err={diff:.3e} ref_max={scale:.3e}")
    assert diff < 2e-3 * max(scale, 1e-6) + 1e-5, f"Output mismatch: {diff}"
    print("EXTENSION TEST PASSED")


if __name__ == "__main__":
    sys.exit(main())
