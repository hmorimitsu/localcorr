"""Integration tests: real torch.onnx.export of the correlation blocks.

These exercise the legacy ONNX tracer end to end (the export path the
package is designed for): the "onnx" backend must produce only standard
ops, and the default "tensorrt" backend must emit the
custom_ops::LocalCorrCudaPyramid plugin node. Skipped automatically when
onnx is not installed or the installed torch no longer supports the
legacy exporter.
"""

import inspect
import math

import pytest

onnx = pytest.importorskip("onnx", reason="onnx package required")

import torch

import localcorr

from _reference import make_coords, make_fmaps

B, C, H, W = 1, 8, 16, 24
LEVELS, RADIUS = 2, 2


class _CorrWrapper(torch.nn.Module):
    """Exportable module computing one local correlation call."""

    def __init__(self, fmap1, fmap2, num_levels, radius, corr_mode):
        super().__init__()
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.num_levels = num_levels
        self.radius = radius
        self.corr_mode = corr_mode

    def forward(self, coords):
        block = localcorr.get_localcorr_block(
            self.fmap1,
            self.fmap2,
            radius=self.radius,
            num_levels=self.num_levels,
            corr_mode=self.corr_mode,
        )
        return block(coords)


def _export_legacy(module, coords, tmp_path, name):
    """Export with the legacy tracer; skip if unsupported by this torch."""
    kwargs = {}
    signature = inspect.signature(torch.onnx.export)
    if "dynamo" in signature.parameters:
        kwargs["dynamo"] = False
    if "opset_version" in signature.parameters:
        kwargs["opset_version"] = 17
    path = str(tmp_path / name)
    try:
        torch.onnx.export(module, (coords,), path, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on torch version
        pytest.skip(f"legacy torch.onnx.export unavailable: {exc}")
    return path


def _node_ops(model):
    return [node.op_type for node in model.graph.node]


def _custom_nodes(model):
    return [n for n in model.graph.node if n.op_type == "LocalCorrCudaPyramid"]


def test_generic_backend_exports_standard_ops(tmp_path):
    localcorr.set_onnx_backend("onnx")
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=50)
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=51)
    model = _CorrWrapper(f1, f2, LEVELS, RADIUS, "localcorr_cuda")
    exported = onnx.load(_export_legacy(model, coords, tmp_path, "generic.onnx"))
    ops = _node_ops(exported)
    assert "LocalCorrCudaPyramid" not in ops
    assert "GridSample" in ops


def test_default_backend_exports_plugin_node(tmp_path):
    localcorr.set_onnx_backend("tensorrt")
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=52)
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=53)
    model = _CorrWrapper(f1, f2, LEVELS, RADIUS, "localcorr_cuda")
    exported = onnx.load(_export_legacy(model, coords, tmp_path, "plugin.onnx"))
    custom = _custom_nodes(exported)
    assert len(custom) == 1
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in custom[0].attribute}
    assert attrs["radius"] == RADIUS
    assert attrs["levels"] == LEVELS
    assert attrs["dilation"] == 1
    assert attrs["ymajor"] == 0


def test_onnx_mode_forces_standard_ops(tmp_path):
    # corr_mode=localcorr_onnx stays generic even with the tensorrt
    # backend selected globally.
    localcorr.set_onnx_backend("tensorrt")
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=54)
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=55)
    model = _CorrWrapper(f1, f2, LEVELS, RADIUS, "localcorr_onnx")
    exported = onnx.load(_export_legacy(model, coords, tmp_path, "forced.onnx"))
    assert not _custom_nodes(exported)
    assert "GridSample" in _node_ops(exported)


def test_auto_mode_exports_plugin_node(tmp_path):
    # With the default backend, corr_mode=localcorr_auto must export as
    # the plugin node (the fused Triton kernel cannot be exported).
    localcorr.set_onnx_backend("tensorrt")
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=56)
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=57)
    model = _CorrWrapper(f1, f2, LEVELS, RADIUS, "localcorr_auto")
    exported = onnx.load(_export_legacy(model, coords, tmp_path, "auto_plugin.onnx"))
    custom = _custom_nodes(exported)
    assert len(custom) == 1
    assert "GridSample" not in _node_ops(exported)


def test_auto_mode_generic_backend_exports_standard_ops(tmp_path):
    localcorr.set_onnx_backend("onnx")
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=58)
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=59)
    model = _CorrWrapper(f1, f2, LEVELS, RADIUS, "localcorr_auto")
    exported = onnx.load(_export_legacy(model, coords, tmp_path, "auto_generic.onnx"))
    assert not _custom_nodes(exported)
    assert "GridSample" in _node_ops(exported)


def test_generic_export_runtime_numeric(tmp_path):
    """The standard-ops graph executes and matches the eager reference."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime required")
    from localcorr.onnx_generic import localcorr_pyramid_onnx_generic

    localcorr.set_onnx_backend("onnx")
    f1, f2 = make_fmaps(batch=B, channels=C, height=H, width=W, seed=56)
    coords = make_coords(batch=B, height=H, width=W, max_disp=2.0, seed=57)
    model = _CorrWrapper(f1, f2, LEVELS, RADIUS, "localcorr_cuda_pyramid")
    path = _export_legacy(model, coords, tmp_path, "runtime.onnx")
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        (out,) = session.run(None, {"coords": coords.numpy()})
    except Exception as exc:  # pragma: no cover - ORT build dependent
        pytest.skip(f"onnxruntime could not run the graph: {exc}")
    ref = localcorr_pyramid_onnx_generic(f1, f2, coords, LEVELS, RADIUS)
    ref = ref.reshape(B, -1, H, W) / math.sqrt(C)
    assert out.shape == tuple(ref.shape)
    assert pytest.approx(0.0, abs=1e-3) == float(abs(out - ref.numpy()).max())
