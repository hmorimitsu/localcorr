"""Tests for localcorr.openvino_ext (OpenVINO custom-op integration)."""

import sys

import pytest

import localcorr.openvino_ext as ov_ext


class TestPaths:
    def test_extension_path_consistent(self):
        path = ov_ext.localcorr_extension_path()
        assert path is None or (path.exists() and path.name == ov_ext._EXTENSION_NAME)

    def test_kernel_path_consistent(self):
        path = ov_ext.localcorr_kernel_path()
        assert path is None or (path.exists() and path.name == ov_ext._KERNEL_NAME)


class TestAddExtension:
    def test_registers_on_core_when_built(self):
        class FakeCore:
            def __init__(self):
                self.paths = []

            def add_extension(self, path):
                self.paths.append(path)

        core = FakeCore()
        result = ov_ext.add_localcorr_extension(core)
        path = ov_ext.localcorr_extension_path()
        if path is None:
            assert result is False
            assert core.paths == []
        else:
            assert result is True
            assert core.paths == [str(path)]


class TestPreload:
    def test_preload_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("LOCALCORR_SYCL_PRELOAD", "0")
        assert ov_ext._preload_oneapi_runtime() is None


class TestOnnxHasLocalCorrNodes:
    @pytest.fixture
    def model_paths(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        def save(nodes, name):
            graph = helper.make_graph(
                nodes,
                name,
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 13)]
            )
            path = tmp_path / name
            onnx.save(model, str(path))
            return str(path)

        plain = save([helper.make_node("Relu", ["x"], ["y"], name="r")], "plain.onnx")
        custom = save(
            [
                helper.make_node("Relu", ["x"], ["t"], name="r"),
                helper.make_node(
                    "LocalCorrCudaPyramid",
                    ["x", "x", "x"],
                    ["y"],
                    domain="custom_ops",
                    name="corr",
                    radius=2,
                ),
            ],
            "custom.onnx",
        )
        return plain, custom

    def test_detection(self, model_paths):
        plain, custom = model_paths
        assert ov_ext.onnx_has_localcorr_nodes(custom) is True
        assert ov_ext.onnx_has_localcorr_nodes(plain) is False

    def test_missing_file_is_false(self):
        assert ov_ext.onnx_has_localcorr_nodes("/nonexistent/model.onnx") is False

    def test_byte_scan_fallback(self, model_paths, monkeypatch):
        # Without the onnx package the check falls back to a raw scan.
        plain, custom = model_paths
        monkeypatch.setitem(sys.modules, "onnx", None)
        assert ov_ext.onnx_has_localcorr_nodes(custom) is True
        assert ov_ext.onnx_has_localcorr_nodes(plain) is False
