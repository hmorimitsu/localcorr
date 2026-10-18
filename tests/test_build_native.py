"""Tests for localcorr.build_native (no real compilation happens)."""

import pytest

from localcorr import build_native


@pytest.fixture
def fake_pkg(tmp_path, monkeypatch):
    """Fake installed package: scripts exist, 'builds' write artifacts.

    Returns (root, calls) where calls records the subprocess.run
    invocations performed by build().
    """
    root = tmp_path / "localcorr"
    for rel in build_native._SCRIPTS.values():
        script = root / rel
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/usr/bin/env bash\ntrue\n")

    calls = []

    def fake_run(cmd, cwd=None, env=None, check=True):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        out = root / cwd.relative_to(root) / "build"
        out.mkdir(parents=True, exist_ok=True)
        for name in (
            "liblocalcorr_cuda_pyramid_plugin.so",
            "liblocalcorr_sycl_pyramid.so",
            "liblocalcorr_sycl_pyramid_ov.so",
        ):
            (out / name).write_bytes(b"")
        (out / "localcorr_sycl_xpu.cpython-311.so").write_bytes(b"")

    monkeypatch.setattr(build_native.subprocess, "run", fake_run)
    monkeypatch.setattr(build_native.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(build_native, "_PKG_DIR", root)
    return root, calls


def test_unknown_component_rejected():
    with pytest.raises(ValueError, match="unknown component"):
        build_native.build("nonsense")


def test_missing_toolchain_error_mentions_tool(monkeypatch):
    monkeypatch.setattr(build_native.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="icpx"):
        build_native.build("sycl")


def test_missing_toolchain_error_mentions_both_cuda_tools(monkeypatch):
    monkeypatch.setattr(build_native.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="cmake.*nvcc"):
        build_native.build("cuda")


def test_build_runs_script_and_returns_artifacts(fake_pkg):
    root, calls = fake_pkg
    libs = build_native.build("cuda", jobs=3)

    assert libs == [root / "cuda_pyramid" / "build" / "liblocalcorr_cuda_pyramid_plugin.so"]
    assert len(calls) == 1
    assert calls[0]["cmd"] == ["bash", str(root / "cuda_pyramid" / "build.sh")]
    assert calls[0]["cwd"] == root / "cuda_pyramid"
    assert calls[0]["env"]["BUILD_JOBS"] == "3"


def test_build_sycl_returns_both_artifacts(fake_pkg):
    root, _ = fake_pkg
    libs = build_native.build_sycl_pyramid()
    assert set(libs) == {
        root / "sycl_pyramid" / "build" / "liblocalcorr_sycl_pyramid.so",
        root / "sycl_pyramid" / "build" / "liblocalcorr_sycl_pyramid_ov.so",
    }


def test_build_xpu_globs_abi_tagged_artifact(fake_pkg):
    root, _ = fake_pkg
    libs = build_native.build_xpu()
    assert libs == [root / "xpu" / "build" / "localcorr_sycl_xpu.cpython-311.so"]


def test_failed_build_raises_runtime_error(fake_pkg, monkeypatch):
    def failing_run(cmd, **kwargs):
        raise build_native.subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(build_native.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="exit code 2"):
        build_native.build("cuda")


def test_main_success_exit_code(fake_pkg, capsys):
    assert build_native.main(["cuda"]) == 0
    assert "liblocalcorr_cuda_pyramid_plugin.so" in capsys.readouterr().out


def test_main_all_skips_missing_toolchains(fake_pkg, monkeypatch, capsys):
    # cmake/nvcc/bash available, icpx not: sycl/xpu skipped, cuda built.
    monkeypatch.setattr(
        build_native.shutil,
        "which",
        lambda name: "/usr/bin/tool" if name in ("cmake", "nvcc", "bash") else None,
    )
    assert build_native.main(["all"]) == 0
    out = capsys.readouterr().out
    assert "skipped=['sycl', 'xpu']" in out
    assert "built=['cuda']" in out


def test_main_all_fails_when_nothing_buildable(fake_pkg, monkeypatch):
    monkeypatch.setattr(build_native.shutil, "which", lambda name: None)
    assert build_native.main(["all"]) == 1


def test_main_rejects_unknown_component():
    with pytest.raises(SystemExit) as exc:
        build_native.main(["nonsense"])
    assert exc.value.code == 2


def test_component_scripts_exist():
    # The wrapped scripts ship as package data of every install.
    for rel in build_native._SCRIPTS.values():
        assert (build_native._PKG_DIR / rel).exists(), rel
