# localcorr

![GitHub CI python status](https://github.com/hmorimitsu/localcorr/actions/workflows/python.yml/badge.svg)
![GitHub CI pytorch status](https://github.com/hmorimitsu/localcorr/actions/workflows/pytorch.yml/badge.svg)
![GitHub CI build status](https://github.com/hmorimitsu/localcorr/actions/workflows/build.yml/badge.svg)

PyTorch local correlation kernels for optical flow networks such as
[RAFT](https://github.com/princeton-vl/RAFT/) and
[PWC-Net](https://github.com/NVlabs/PWC-Net).

The implementations are drop-in compatible with RAFT's
[alt_cuda_corr](https://github.com/princeton-vl/RAFT/) and
[spatial_correlation_sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension),
and run on NVIDIA and Intel GPUs as well as any ONNX-supported
runtime (see [Components](#components)).

You can check [PTLFlow](https://github.com/hmorimitsu/ptlflow) to see examples of optical flow models using `localcorr`.

## Quick start

```bash
pip install localcorr
```

That's enough to **train and run inference eagerly** on Linux — the default Triton
kernels are JIT-compiled on first use with no build step required (see
[Installation](#installation) for other platforms and optional AOT plugins).

**RAFT-style correlation** (feature maps + sampled coordinates):

```python
import torch
from localcorr import get_localcorr_block

# fmap1/fmap2: [B, C, H, W], coords: [B, 2, H, W] (x, y)
corr_block = get_localcorr_block(
    fmap1, fmap2, num_levels=4, radius=4,
    corr_mode="localcorr_triton",
)
corr = corr_block(coords)  # [B, levels*(2r+1)^2, H, W]
```

**PWCNet-style spatial correlation:**

```python
from localcorr import get_localcorr_sampler

# same result as SpatialCorrelationSampler(kernel_size=1, patch_size=9, padding=0)
sampler = get_localcorr_sampler(patch_size=9, corr_mode="localcorr_triton")
corr = sampler(input1, input2)  # [B, 9, 9, H, W]
```

## Choosing a corr mode

Everything in `localcorr` is selected through a single `corr_mode` argument
(accepted by both `get_localcorr_block` and `get_localcorr_sampler`):

| `corr_mode` | Runs on | Best for |
|---|---|---|
| `localcorr_triton` | Any Triton-supported GPU | **Default. Eager execution, no build step.** |
| `localcorr_onnx` | Any Triton-supported GPU | Same eager behavior; exports plugin-free ONNX |
| `localcorr_cuda` | NVIDIA GPU (AOT CUDA kernel) | Fastest CUDA path + TensorRT engines |
| `localcorr_sycl` | Intel GPU (SYCL on `torch.xpu`) | Fastest Intel-GPU path + OpenVINO custom op |
| `localcorr_auto` | NVIDIA or Intel GPU (auto) | One code path across both hardware platforms |

Notes:

- **Strict vs. automatic dispatch:** Each specific mode (`triton`, `cuda`, `sycl`) uses its own eager engine and fails with a clear error if the extension is missing or tensors are on the wrong device, rather than silently falling back. `localcorr_auto` is the exception: it inspects the tensor device and falls back to the Triton kernel when the native AOT extension is not built.
- **Export behavior:** The modes only differ in eager execution and ONNX export; see [ONNX export](#onnx-export) for how each one exports.

## Installation

```bash
pip install localcorr
```

`triton` (the engine behind `localcorr_triton` and `localcorr_onnx`) is installed
automatically on Linux. On platforms where Triton does not provide wheels,
`localcorr` still imports and the pure-PyTorch / ONNX-export paths work, while
calling a Triton kernel raises an `ImportError` on first use.

TensorRT and OpenVINO are deliberately **not** required dependencies:
- To build or run TensorRT engines with the `cuda_pyramid` plugin, install TensorRT yourself (the plugin build only needs `NvInfer.h` and `libnvinfer` discoverable).
- Likewise, OpenVINO is only needed to build the `sycl` extension and run models using the custom op.

### Building the native components (optional)

The AOT components (`cuda_pyramid`, `sycl_pyramid`, `xpu`) provide the fastest
eager engines and the TensorRT/OpenVINO plugins listed above. They are optional
and can be compiled on demand with a single command from any directory (both
the pip wheel and sdist ship the native sources):

```bash
python -m localcorr.build_native cuda   # NVIDIA: AOT kernel + TensorRT plugin
python -m localcorr.build_native sycl   # Intel: OpenVINO custom-op extension
python -m localcorr.build_native xpu    # Intel: torch.xpu SYCL extension
python -m localcorr.build_native all
```

When pip-installed, you can also use the console script (`localcorr-build cuda`)
or build programmatically (`from localcorr import build_native; build_native.build("cuda")`).
From a repository checkout, the direct scripts (`bash localcorr/<component>/build.sh`)
also work.

Requirements per target:

- **cuda** — CUDA toolkit (`nvcc`), CMake >= 3.18, and TensorRT (`NvInfer.h` +
  `libnvinfer`). Set `TRT_ROOT` or `TENSORRT_ROOT` if TensorRT is not in a
  default system path.
  Output: `localcorr/cuda_pyramid/build/liblocalcorr_cuda_pyramid_plugin.so`
- **sycl** — Intel oneAPI (`icpx` with SYCL) and the OpenVINO runtime (the pip
  package headers are used by default; override with `OV_INSTALL_DIR`).
  Output: `localcorr/sycl_pyramid/build/liblocalcorr_sycl_pyramid.so`
  (the SYCL kernel) and
  `localcorr/sycl_pyramid/build/liblocalcorr_sycl_pyramid_ov.so` (the
  OpenVINO extension, plus `test_kernel` for correctness verification).
- **xpu** — a PyTorch XPU build
  (`pip install torch --index-url [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu)`),
  oneAPI (`icpx`), and (for the Triton fallback only) `triton-xpu` plus the Level Zero headers.
  Output: `localcorr/xpu/build/localcorr_sycl_xpu*.so`

## Usage

### Eager execution & training

All modes support `fp16` and `fp32` inputs as well as PyTorch autograd for
training. Across all backends, backward passes compute gradients with respect
to the feature maps (`fmap1`/`fmap2` or `input1`/`input2`), while gradients for
the sampling coordinates (`coords`) are returned as zeros.

If you want the same model code to run unchanged across NVIDIA and Intel GPUs,
use `corr_mode="localcorr_auto"` (which also accepts an optional `xpu_backend`
argument in both factories):

```python
# RAFT-style: backend resolved once from the device of fmap1/fmap2
corr_block = get_localcorr_block(
    fmap1, fmap2, num_levels=4, radius=4, corr_mode="localcorr_auto"
)
corr = corr_block(coords)  # [B, levels*(2r+1)^2, H, W]

# PWCNet-style: backend dispatched per call from the device of input1/input2
# (uses the LocalCorrCudaPyramidSampler class under the hood)
sampler = get_localcorr_sampler(patch_size=9, corr_mode="localcorr_auto")
corr = sampler(input1, input2)  # [B, 9, 9, H, W]
```

How `localcorr_auto` dispatches:
- **NVIDIA GPUs:** Uses the AOT CUDA kernel (`localcorr_cuda`) when the extension is built, otherwise falls back to the Triton kernel.
- **Intel GPUs (`torch.xpu`):** Uses the fused SYCL kernel when the extension is built, otherwise falls back to the Triton kernel (`xpu_backend="triton"` forces Triton; defaults to `"sycl"`).
- **ONNX export:** Emits a single `LocalCorrCudaPyramid` plugin node (which also sidesteps the fact that the fused Triton kernel cannot be traced for export), or standard ONNX ops when the `"onnx"` backend is active.

### ONNX export

By default, ONNX export emits `LocalCorrCudaPyramid` custom nodes understood by
the TensorRT ONNX parser (via the CUDA plugin) and OpenVINO (via the
`sycl_pyramid` SYCL custom-op extension).

To export models for runtimes without custom-plugin support, switch to the
standard-ops backend before exporting:

```python
import localcorr

localcorr.set_onnx_backend("onnx")  # or "tensorrt" (default)
# ... then export with the legacy tracer:
# torch.onnx.export(model, args, "model.onnx", dynamo=False, opset_version=17)
```

> **Important:** Always pass `dynamo=False` when calling `torch.onnx.export`. Recent PyTorch versions default to the Dynamo exporter, which silently drops the correlation and emits an all-zeros placeholder instead of the plugin node.

Key behaviors during export:
- **Environment variable:** You can also select the standard-ops backend via `LOCALCORR_ONNX_BACKEND=onnx`. This setting only affects ONNX export; eager execution is unchanged.
- **Mode compatibility:** Models configured with `corr_mode="localcorr_triton"` cannot be exported with the default `"tensorrt"` backend (the fused Triton kernel raises an error on export). Either set the backend to `"onnx"` or initialize the model with `corr_mode="localcorr_onnx"`.
- **`localcorr_onnx` mode:** Always exports using standard ONNX ops without touching global state, while running the fast fused Triton kernel in eager mode. This produces plugin-free ONNX and TensorRT engines at the cost of a slower correlation subgraph than the fused plugin.
- **Standard-ops design:** Rather than calling `grid_sample` directly over multi-channel feature maps, the standard-ops path contracts channels via `matmul` and gathers displacement windows from single-channel correlation planes, which is significantly faster (see design notes in `localcorr/onnx_generic.py`).

### TensorRT inference (NVIDIA)

Models exported with `LocalCorrCudaPyramid` nodes require the TensorRT runtime
to load the compiled plugin library before deserializing the engine (after
[building the `cuda` target](#building-the-native-components-optional)):

```python
import ctypes
import tensorrt as trt

trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, "")
ctypes.CDLL(
    "localcorr/cuda_pyramid/build/liblocalcorr_cuda_pyramid_plugin.so",
    mode=ctypes.RTLD_GLOBAL,
)
```

For command-line tools such as `trtexec`, you can pass the library via `LD_PRELOAD` or `--staticPlugins` / `--dynamicPlugins`.

### OpenVINO inference (Intel)

After [building the `sycl` extension](#building-the-native-components-optional),
register the custom op that maps the `custom_ops::LocalCorrCudaPyramid`
ONNX node to the SYCL kernel:

```python
import localcorr.openvino_ext as lcorr_ov  # Import BEFORE openvino:
# preloads the oneAPI OpenCL runtime so the SYCL extension finds the
# OpenCL CPU platform.
import openvino as ov

core = ov.Core()
lcorr_ov.add_localcorr_extension(core)  # registers the custom op
model = core.read_model("model.onnx")    # LocalCorrCudaPyramid resolves
```

Notes:

- **SYCL device selection:** The extension computes on the SYCL GPU device by default (`auto`: GPU preferred, CPU fallback when no GPU is available). Set `LOCALCORR_SYCL_DEVICE=cpu|gpu|default|auto` to override.
- **OpenVINO device plugins:** The OpenVINO `GPU` and `NPU` plugins cannot execute the custom op directly (Reference nodes are a `CPU`-plugin capability), so load the OpenVINO model on `CPU` or `AUTO` (which delegates the custom op and dispatches the SYCL kernel onto the GPU). For standalone NPU deployment, export with the standard-ops backend (`localcorr_onnx`).
- **oneAPI CPU workaround:** The OpenCL CPU runtime in oneAPI 2026.1 crashes or hangs on SYCL sub-group shuffles; the CPU execution paths avoid them (see `localcorr/sycl_pyramid/localcorr_cuda_pyramid_kernel.dp.cpp`).
- **Symbol visibility:** The oneAPI runtime preload in `localcorr.openvino_ext` intentionally uses `RTLD_LOCAL`; loading those libraries with `RTLD_GLOBAL` breaks the OpenVINO GPU plugin's JIT compiler (see module docstring for details).

## Components

| Component | Path | Build | Used for |
|---|---|---|---|
| `localcorr_triton` | `localcorr/triton/` | none (Triton JIT) | Eager forward + backward for `localcorr_triton` and `localcorr_onnx` |
| `localcorr_cuda_pyramid` (`localcorr_cuda`) | `localcorr/cuda_pyramid/` | CMake + `nvcc` | Eager CUDA forward + backward (`ctypes`) + TensorRT engines (fastest NVIDIA path) |
| `localcorr_sycl_pyramid` | `localcorr/sycl_pyramid/` | `build.sh` + `icpx` | OpenVINO inference on Intel (SYCL custom op) |
| `localcorr_sycl_xpu` | `localcorr/xpu/` | `build.sh` + `icpx` | Eager `torch.xpu` forward + backward for `localcorr_sycl` on Intel GPUs |
| `onnx-generic` export | `localcorr/onnx_generic.py` | none | Plugin-free ONNX export using standard ops |

### localcorr_triton

Triton kernels, JIT-compiled at first use (`pip install triton`, no build step):

- `triton/pyramid_corr.py`: RAFT-style multi-level local correlation around sampled coordinates (`LocalCorrTriton` per-level, and `LocalCorrTritonFused` computing all pyramid levels in one launch).
- `triton/spatial_corr.py`: PWCNet-style single-level spatial correlation (`TritonSpatialCorrelationSampler`), equivalent to `SpatialCorrelationSampler(kernel_size=1, patch_size, padding=0)`.

### localcorr_cuda_pyramid

An AOT-compiled CUDA kernel and TensorRT `IPluginV3` plugin that computes all
correlation pyramid levels in **one kernel launch** (using templated fast paths
with warp-shuffle reductions; see `localcorr/cuda_pyramid/localcorr_cuda_pyramid_kernel.h`).
During ONNX export, each correlation call is emitted as a single
`LocalCorrCudaPyramid` node resolved by the TensorRT ONNX parser.

The same plugin serves both correlation styles:
- **RAFT-style:** `levels = N` (pyramid), x-major displacement order.
- **PWCNet-style:** `levels = 1`, y-major displacement order (`channel = dy_idx * patch_size + dx_idx`).

In eager mode (`corr_mode="localcorr_cuda"`), the kernel is invoked through
`ctypes` on PyTorch's current CUDA stream, paired with a fused backward kernel.

### localcorr_sycl_xpu (PyTorch on Intel GPUs)

Exposes the fused SYCL kernel to PyTorch as a `torch.xpu` (`SyclExtension`)
extension. Models running on `torch.xpu` devices compute the entire correlation
pyramid in a single fused kernel launch on the Intel GPU, ordered on PyTorch's
XPU stream for both forward and backward passes.

### localcorr_sycl_pyramid (OpenVINO on Intel)

A SYCL port of the fused kernel paired with an OpenVINO custom-operation
extension that maps `custom_ops::LocalCorrCudaPyramid` ONNX nodes to the SYCL
kernel. See [OpenVINO inference](#openvino-inference-intel) for usage.

### Plugin interface

ONNX node specification (domain `custom_ops`):

```text
LocalCorrCudaPyramid(fmap1, fmap2_packed, coords) -> corr
  attributes: radius (int), levels (int), dilation (int, default 1),
              ymajor (int, default 0)
  fmap1:        [B, H1, W1, C]                     (NHWC, fp16 or fp32)
  fmap2_packed: [B, C * sum_i h_i*w_i]             (NHWC levels packed,
                                                    h_i = H1>>i, w_i = W1>>i,
                                                    batch-major)
  coords:       [B, 2, H1, W1]                     (planar, x in channel 0,
                                                    level-0 scale)
  corr:         [B, levels, (2*radius+1)^2, H1, W1]
```

The displacement channel order is `d = iy + rd*ix` by default (RAFT) or
`d = ix + rd*iy` when `ymajor=1` (PWCNet / `SpatialCorrelationSampler`
order). `dilation` controls the step between consecutive displacement positions
(matching `dilation_patch` in spatial correlation samplers).

## Tests

Unit tests run on CPU by default; tests requiring a CUDA device, an Intel GPU,
or optional packages (`onnx`, `onnxruntime`, `openvino`) are skipped
automatically when unavailable. From the repository root:

```bash
pytest tests
```

(Prefer `pytest` over `python -m pytest` to avoid any local directory shadowing
with the `triton` package; from the parent directory, `pytest localcorr/tests`
also works.)

## Acknowledgments

Most of the code in this repository was written by GLM-5.3, a large
language model trained by Z.ai, with human guidance, testing, and review.
Honestly, I wouldn't have known how to write custom CUDA, SYCL, Triton,
TensorRT, and OpenVINO C++ extensions all by myself.

The implementations are based on:

- [alt_cuda_corr](https://github.com/princeton-vl/RAFT/tree/master/alt_cuda_corr)
  from the [RAFT](https://github.com/princeton-vl/RAFT/) project — the
  original RAFT-style local correlation kernels, reimplemented here as
  a single fused pyramid kernel.
- [spatial_correlation_sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension)
  by Clément Pinard — the PWCNet-style spatial correlation package whose
  semantics (`kernel_size=1, patch_size, padding=0`) this package is
  compatible with.