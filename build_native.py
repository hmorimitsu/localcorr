"""Build the AOT-compiled native components of localcorr from anywhere.

The CUDA/TensorRT plugin (cuda_pyramid), the SYCL/OpenVINO extension
(sycl_pyramid) and the PyTorch XPU extension (xpu) ship as sources
inside the package and are compiled on demand into <component>/build/
next to them -- exactly the locations the runtime code looks them up
in. This module wraps the per-component build scripts so that users of
an installed package do not have to locate the package directory
first:

    # from any directory, pip-installed or from a checkout:
    python -m localcorr.build_native cuda      # TensorRT plugin (nvcc + cmake)
    python -m localcorr.build_native sycl      # OpenVINO custom op (icpx)
    python -m localcorr.build_native xpu       # PyTorch XPU extension (icpx)
    python -m localcorr.build_native all       # try every component

    # when pip-installed, the equivalent console script also exists:
    localcorr-build cuda

    # or programmatically:
    from localcorr import build_native
    libs = build_native.build("cuda")          # -> [Path of the built .so]

Driving a native build from Python is the same design used by
torch.utils.cpp_extension; nothing is ever compiled at import time,
only on the explicit request made through this module.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent

COMPONENTS = ("cuda", "sycl", "xpu")

# component -> build script inside the package (shipped as package data)
_SCRIPTS = {
    "cuda": "cuda_pyramid/build.sh",
    "sycl": "sycl_pyramid/build.sh",
    "xpu": "xpu/build.sh",
}

# component -> tools that must be on PATH before the script is started
_REQUIRED_TOOLS = {
    "cuda": ("cmake", "nvcc"),
    "sycl": ("icpx",),
    "xpu": ("icpx",),
}

# component -> hint appended when the required tools are missing
_TOOLCHAIN_HINTS = {
    "cuda": (
        "requires the CUDA toolkit and CMake >= 3.18; the TensorRT "
        "headers/lib must be findable (TRT_ROOT / TENSORRT_ROOT, see "
        "cuda_pyramid/CMakeLists.txt)"
    ),
    "sycl": (
        "requires Intel oneAPI (icpx); source setvars.sh or set "
        "ONEAPI_ROOT if it is not in a default location"
    ),
    "xpu": (
        "requires Intel oneAPI (icpx) and a PyTorch XPU build in this "
        "interpreter (source setvars.sh / set ONEAPI_ROOT as needed)"
    ),
}

# component -> artifacts (relative to the package dir) that must exist
# after a successful build; the xpu extension name carries the Python
# ABI tag, so it is globbed instead of listed.
_ARTIFACTS = {
    "cuda": ("cuda_pyramid/build/liblocalcorr_cuda_pyramid_plugin.so",),
    "sycl": (
        "sycl_pyramid/build/liblocalcorr_sycl_pyramid.so",
        "sycl_pyramid/build/liblocalcorr_sycl_pyramid_ov.so",
    ),
    "xpu": (),
}


def _missing_tools(component):
    """Tools required on PATH for the component, bash included."""
    tools = [tool for tool in _REQUIRED_TOOLS[component] if shutil.which(tool) is None]
    if shutil.which("bash") is None:
        tools.append("bash")
    return tools


def _artifact_paths(component):
    if component == "xpu":
        return sorted((_PKG_DIR / "xpu" / "build").glob("localcorr_sycl_xpu*.so"))
    paths = []
    for rel in _ARTIFACTS[component]:
        path = _PKG_DIR / rel
        if path.exists():
            paths.append(path)
    return paths


def build(component, jobs=None):
    """Build one native component and return the built artifact paths.

    Args:
        component: "cuda" (TensorRT plugin), "sycl" (OpenVINO custom
            op) or "xpu" (PyTorch XPU extension).
        jobs: parallel compile jobs; forwarded to the build scripts as
            BUILD_JOBS (they default to nproc).

    Returns:
        List of Path objects of the produced libraries.

    Raises:
        ValueError: unknown component.
        RuntimeError: a required tool is missing, the build script is
            not shipped, the build failed, or expected artifacts were
            not produced.
    """
    component = component.lower()
    if component not in _SCRIPTS:
        raise ValueError(
            f"unknown component {component!r}; expected one of {COMPONENTS}"
        )

    missing = _missing_tools(component)
    if missing:
        raise RuntimeError(
            f"cannot build {component!r}: {', '.join(missing)} not found "
            f"in PATH ({_TOOLCHAIN_HINTS[component]})"
        )

    script = _PKG_DIR / _SCRIPTS[component]
    if not script.exists():
        raise RuntimeError(
            f"build script {script} not found; the native sources ship "
            "as package data -- reinstall the package if they are missing"
        )

    env = dict(os.environ)
    if jobs is not None:
        env.setdefault("BUILD_JOBS", str(jobs))

    print(f"[localcorr] building {component} ({script})")
    try:
        subprocess.run(["bash", str(script)], cwd=script.parent, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{component} build failed with exit code {exc.returncode}"
        ) from exc

    artifacts = _artifact_paths(component)
    if not artifacts:
        expected = _ARTIFACTS[component] or ("xpu/build/localcorr_sycl_xpu*.so",)
        raise RuntimeError(
            f"{component} build finished but no expected artifacts were "
            f"found (expected {list(expected)})"
        )
    for artifact in artifacts:
        print(f"[localcorr] built {artifact}")
    return artifacts


def build_cuda_pyramid(jobs=None):
    """Build the CUDA/TensorRT plugin; returns its library path list."""
    return build("cuda", jobs=jobs)


def build_sycl_pyramid(jobs=None):
    """Build the SYCL kernel + OpenVINO custom-op extension."""
    return build("sycl", jobs=jobs)


def build_xpu(jobs=None):
    """Build the PyTorch XPU (SYCL) extension."""
    return build("xpu", jobs=jobs)


def main(argv=None):
    """CLI entry point: build one (or all) native components."""
    parser = argparse.ArgumentParser(
        prog="localcorr-build",
        description="Build the AOT-compiled native components of localcorr.",
    )
    parser.add_argument(
        "component",
        choices=(*COMPONENTS, "all"),
        help="which component to build ('all' attempts every one)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="parallel compile jobs (default: nproc)",
    )
    args = parser.parse_args(argv)

    if args.component != "all":
        try:
            build(args.component, jobs=args.jobs)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    # 'all': attempt every component; a missing toolchain only skips it.
    built, skipped, failed = [], [], []
    for component in COMPONENTS:
        try:
            build(component, jobs=args.jobs)
            built.append(component)
        except RuntimeError as exc:
            if _missing_tools(component):
                print(f"[localcorr] skipped {component}: {exc}")
                skipped.append(component)
            else:
                print(f"[localcorr] FAILED {component}: {exc}", file=sys.stderr)
                failed.append(component)
    print(
        f"[localcorr] summary: built={built or '-'} skipped={skipped or '-'} "
        f"failed={failed or '-'}"
    )
    return 1 if (failed or not built) else 0


if __name__ == "__main__":
    sys.exit(main())
