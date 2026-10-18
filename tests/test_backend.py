"""Tests for localcorr.backend (ONNX export backend selection)."""

import pytest

import localcorr
import localcorr.backend as backend


@pytest.fixture(autouse=True)
def _clean_backend_state(monkeypatch):
    monkeypatch.delenv(backend._ENV_VAR, raising=False)
    backend._onnx_backend = None
    yield
    backend._onnx_backend = None


def test_default_backend_is_tensorrt():
    assert backend.get_onnx_backend() == "tensorrt"


def test_set_and_get_backend():
    localcorr.set_onnx_backend("onnx")
    assert backend.get_onnx_backend() == "onnx"
    localcorr.set_onnx_backend("tensorrt")
    assert backend.get_onnx_backend() == "tensorrt"


@pytest.mark.parametrize("bad", ["cudnn", "", "ONNX", "TensorRT", "onnx "])
def test_set_invalid_backend_raises(bad):
    with pytest.raises(ValueError, match="Unknown ONNX backend"):
        localcorr.set_onnx_backend(bad)


def test_backend_from_env_variable(monkeypatch):
    monkeypatch.setenv(backend._ENV_VAR, "onnx")
    assert backend.get_onnx_backend() == "onnx"


def test_invalid_env_variable_raises(monkeypatch):
    monkeypatch.setenv(backend._ENV_VAR, "nonsense")
    with pytest.raises(ValueError, match="Invalid value for"):
        backend.get_onnx_backend()


def test_env_value_is_read_only_once(monkeypatch):
    monkeypatch.setenv(backend._ENV_VAR, "onnx")
    assert backend.get_onnx_backend() == "onnx"
    monkeypatch.setenv(backend._ENV_VAR, "tensorrt")
    # The resolved backend is cached; later env changes are ignored.
    assert backend.get_onnx_backend() == "onnx"


def test_use_onnx_generic_ops_requires_export(monkeypatch):
    import torch

    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: False)
    localcorr.set_onnx_backend("onnx")
    assert backend.use_onnx_generic_ops() is False


def test_use_onnx_generic_ops_combinations(monkeypatch):
    import torch

    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    localcorr.set_onnx_backend("onnx")
    assert backend.use_onnx_generic_ops() is True
    localcorr.set_onnx_backend("tensorrt")
    assert backend.use_onnx_generic_ops() is False
