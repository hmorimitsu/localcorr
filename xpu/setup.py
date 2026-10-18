import os

from setuptools import setup

# Build the localcorr SYCL kernel as a PyTorch XPU (SyclExtension)
# extension, so models on torch.xpu devices compute the fused local
# correlation with the SYCL kernel (the CUDA/TensorRT-plugin equivalent
# of localcorr/cuda_pyramid on Intel GPUs).
#
# Requirements:
#   - PyTorch XPU build (torch.xpu.is_available())
#   - oneAPI icpx on PATH (or CC/CXX=icpx): the SYCL compiler
#   - NO AOT flags: the runtime JIT works on Intel GPUs and ocloc is
#     usually not available for AOT.
#
# Run localcorr/xpu/build.sh instead of calling this directly.

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_DIR = os.environ.get(
    "LOCALCORR_SYCL_KERNEL_DIR",
    os.path.join(os.path.dirname(HERE), "sycl_pyramid"),
)

setup(
    name="localcorr_sycl_xpu",
    ext_modules=[
        # NOTE: import SyclExtension lazily so that building does not hard
        # fail on CPU-only torch installs (build.sh checks availability).
        __import__(
            "torch.utils.cpp_extension", fromlist=["SyclExtension"]
        ).SyclExtension(
            name="localcorr_sycl_xpu",
            sources=[
                "lcorr_sycl_xpu.cpp",
                os.path.join(KERNEL_DIR, "localcorr_cuda_pyramid_kernel.dp.cpp"),
            ],
            include_dirs=[KERNEL_DIR],
            extra_compile_args={"cxx": ["-O2", "-std=c++20", "-fsycl"]},
            extra_link_args=["-fsycl"],
        )
    ],
    cmdclass={
        "build_ext": __import__(
            "torch.utils.cpp_extension", fromlist=["BuildExtension"]
        ).BuildExtension
    },
)
