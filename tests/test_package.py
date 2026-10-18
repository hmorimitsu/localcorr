"""Package-surface tests: exports, version, packaging, standalone-ness."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import localcorr
import localcorr.blocks as blocks_mod
import localcorr.spatial_corr as spatial_mod


def test_version_is_a_string():
    assert isinstance(localcorr.__version__, str)
    assert localcorr.__version__


def test_all_exports_resolve():
    assert len(localcorr.__all__) > 0
    for name in localcorr.__all__:
        assert hasattr(localcorr, name), name


def test_public_api_types():
    import torch.nn as nn

    assert callable(localcorr.get_localcorr_block)
    assert callable(localcorr.set_onnx_backend)
    assert callable(localcorr.get_onnx_backend)
    assert issubclass(localcorr.LocalCorrCudaPyramidSampler, nn.Module)
    for cls in (
        blocks_mod.LocalCorrTritonBlock,
        blocks_mod.LocalCorrTritonFusedBlock,
        blocks_mod.LocalCorrCudaPyramidBlock,
        blocks_mod.LocalCorrCudaPyramidSpatialBlock,
        spatial_mod.LocalCorrCudaPyramidSampler,
    ):
        assert isinstance(cls, type)


def test_triton_symbols_available_or_none():
    # triton is optional (Linux-only wheels); the re-exported kernel
    # symbols are real classes when it is present and None otherwise.
    for name in (
        "LocalCorrTriton",
        "LocalCorrTritonFused",
        "TritonSpatialCorrelationFunction",
        "TritonSpatialCorrelationSampler",
    ):
        value = getattr(localcorr, name)
        assert value is None or isinstance(value, type), name


def test_imports_without_ptlflow():
    """The package must be importable standalone (no ptlflow dependency)."""
    code = (
        "import sys; import localcorr; "
        "leaked = [m for m in sys.modules if m.startswith('ptlflow')]; "
        "assert not leaked, leaked"
    )
    root = str(Path(localcorr.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_pyproject_and_manifest_consistent():
    """Packaging metadata matches the package (source-tree runs only)."""
    try:
        import tomllib
    except ImportError:
        pytest.skip("tomllib unavailable (Python < 3.11)")
    pkg_dir = Path(localcorr.__file__).resolve().parent
    pyproject = pkg_dir / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not running from the source tree")
    data = tomllib.loads(pyproject.read_text())
    assert data["project"]["version"] == localcorr.__version__
    setuptools_cfg = data["tool"]["setuptools"]
    assert "localcorr" in setuptools_cfg["packages"]
    assert "localcorr.triton" in setuptools_cfg["packages"]
    assert setuptools_cfg["package-dir"]["localcorr"] == "."
    deps = " ".join(data["project"]["dependencies"])
    assert "torch" in deps
    # triton must stay optional (platform marker / extra), never a hard dep.
    assert not any(d.strip() == "triton" for d in data["project"]["dependencies"])
    assert "triton" in data["project"]["optional-dependencies"]
    assert (pkg_dir / "MANIFEST.in").exists()
    # The native sources must be shipped (package-data of localcorr).
    package_data = data["tool"]["setuptools"]["package-data"]["localcorr"]
    assert "cuda_pyramid/*.cu" in package_data
    assert "sycl_pyramid/*.cpp" in package_data
    assert "xpu/*.cpp" in package_data
