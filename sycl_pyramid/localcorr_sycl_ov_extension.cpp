// OpenVINO custom operation extension for the local correlation pyramid.
//
// Maps the "custom_ops::LocalCorrCudaPyramid" ONNX node (the same node
// that the TensorRT plugin of localcorr/cuda_pyramid resolves) to a
// custom OpenVINO operation whose evaluate() runs the SYCL kernel
// ported in localcorr_cuda_pyramid_kernel.dp.cpp.
//
// The library is loaded into an OpenVINO Core with
// core.add_extension("<path>/liblocalcorr_sycl_pyramid_ov.so"); the
// ONNX frontend then converts models containing the node, and the
// serialized IR can be read back with the same extension (the
// ov::OpExtension registration also resolves the op from IR).
//
// The SYCL device used by evaluate() is selected with
// LOCALCORR_SYCL_DEVICE=cpu|gpu|default|auto. The default "auto"
// prefers the GPU (where the correlation is compute-bound) and falls
// back to the CPU when no GPU is available; "cpu" and "gpu" force a
// selector, "default" uses the SYCL default selector.
//
// evaluate() stages inputs/outputs through pooled shared-USM buffers
// (see SharedUsmPool) and synchronizes the queue exactly once per call:
// on integrated GPUs, per-call malloc_shared/free and extra
// synchronizations dominate the cost of small correlation calls.

#include "localcorr_cuda_pyramid_kernel.h"

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/onnx/extension/op.hpp>
#include <openvino/op/op.hpp>
#include <openvino/runtime/tensor.hpp>

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Keep this library resident after dlclose. OpenVINO unloads custom-op
// extensions when the Core is destroyed (e.g. when the profiling or
// benchmark process tears down), but the process-lifetime SYCL queue and
// USM pool created by evaluate() keep referencing this library's
// code/data; unloading it mid-flight segfaults at process teardown
// (observed after HETERO inference with the extension).
#if defined(__linux__)
const bool kSelfPinned = [] {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&kSelfPinned), &info) &&
        info.dli_fname != nullptr) {
        // RTLD_NOLOAD|RTLD_NODELETE on an already-loaded object marks it
        // non-unloadable without loading a second copy.
        void* handle =
            dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
        return handle != nullptr;
    }
    return false;
}();
#endif  // defined(__linux__)

// Persistent pool of host-USM staging buffers, reused across evaluate()
// calls. OpenVINO waits for each Reference-op call to finish before the
// dependent ops run, so a buffer checked out for the duration of one call
// is never in flight when reacquired; the mutex + free-list also keeps
// concurrent infer requests safe (each gets distinct buffers).
//
// Host USM (page-locked, cacheable by the CPU, still dereferenceable by
// the kernels) is preferred over shared USM: the CPU-side staging copies
// read/write the buffers, and shared-USM allocations are uncacheable
// from the host on this stack (a 4.6 MB read-back costs ~2.5 ms from
// shared vs well under 1 ms from host USM). Devices without
// usm_host_allocations fall back to shared USM.
// Drain the SYCL queue during process exit (before the SYCL runtime
// tears down) so no in-flight work outlives it; registered once from
// get_queue().
sycl::queue* g_exit_drain_queue = nullptr;

void drain_queue_at_exit() {
    if (g_exit_drain_queue != nullptr) {
        g_exit_drain_queue->wait();
    }
}

class SharedUsmPool {
public:
    explicit SharedUsmPool(sycl::queue& q) : m_queue(q) {
        m_use_host_usm = false;
        try {
            m_use_host_usm =
                q.get_device().get_info<sycl::info::device::usm_host_allocations>();
        } catch (const sycl::exception&) {
            m_use_host_usm = false;
        }
    }

    SharedUsmPool(const SharedUsmPool&) = delete;
    SharedUsmPool& operator=(const SharedUsmPool&) = delete;

    ~SharedUsmPool() {
        for (const auto& entry : m_allocated) {
            sycl::free(entry.first, m_queue);
        }
    }

    // Returns a staging buffer of exactly `bytes` bytes, or nullptr on
    // allocation failure.
    void* acquire(size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_free.find(bytes);
            if (it != m_free.end() && !it->second.empty()) {
                void* ptr = it->second.back();
                it->second.pop_back();
                return ptr;
            }
        }
        void* ptr =
            (m_use_host_usm ? sycl::malloc_host(bytes, m_queue)
                            : sycl::malloc_shared(bytes, m_queue));
        if (ptr != nullptr) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_allocated.emplace_back(ptr, bytes);
        } else {
            std::fprintf(stderr,
                         "LocalCorrCudaPyramid: USM allocation of %zu bytes "
                         "failed\n",
                         bytes);
        }
        return ptr;
    }

    void release(void* ptr, size_t bytes) {
        if (ptr == nullptr || bytes == 0) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_free[bytes].push_back(ptr);
    }

private:
    sycl::queue& m_queue;
    bool m_use_host_usm = false;
    std::mutex m_mutex;
    std::unordered_map<size_t, std::vector<void*>> m_free;
    std::vector<std::pair<void*, size_t>> m_allocated;
};

class LocalCorrCudaPyramidOp : public ov::op::Op {
public:
    OPENVINO_OP("LocalCorrCudaPyramid", "custom_ops");

    LocalCorrCudaPyramidOp() = default;

    LocalCorrCudaPyramidOp(const ov::OutputVector& arguments,
                           int64_t radius,
                           int64_t levels,
                           int64_t ymajor,
                           int64_t dilation)
        : ov::op::Op(arguments),
          m_radius(radius),
          m_levels(levels),
          m_ymajor(ymajor),
          m_dilation(dilation) {
        constructor_validate_and_infer_types();
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("radius", m_radius);
        visitor.on_attribute("levels", m_levels);
        visitor.on_attribute("ymajor", m_ymajor);
        visitor.on_attribute("dilation", m_dilation);
        return true;
    }

    void validate_and_infer_types() override {
        // fmap1:        [B, H1, W1, C]         (NHWC)
        // fmap2_packed: [B, C*sum_i h_i*w_i]   (packed NHWC levels)
        // coords:       [B, 2, H1, W1]         (planar, x first)
        // corr:         [B, levels, RD*RD, H1, W1]
        const auto& f1 = get_input_partial_shape(0);
        const auto& coords = get_input_partial_shape(2);

        ov::Dimension b = f1[0].is_static() ? f1[0] : coords[0];
        ov::Dimension h1 = f1[1].is_static() ? f1[1] : coords[2];
        ov::Dimension w1 = f1[2].is_static() ? f1[2] : coords[3];

        const int64_t rd = 2 * m_radius + 1;
        set_output_type(0, get_input_element_type(0),
                        ov::PartialShape{b, ov::Dimension(m_levels),
                                         ov::Dimension(rd * rd), h1, w1});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_inputs) const override {
        return std::make_shared<LocalCorrCudaPyramidOp>(
            new_inputs, m_radius, m_levels, m_ymajor, m_dilation);
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override {
        const ov::Tensor& fmap1 = inputs[0];
        const ov::Tensor& fmap2_packed = inputs[1];
        const ov::Tensor& coords = inputs[2];
        ov::Tensor& corr = outputs[0];

        const auto et = fmap1.get_element_type();
        if (et != ov::element::f32 && et != ov::element::f16) {
            std::fprintf(stderr,
                         "LocalCorrCudaPyramid: unsupported input type %s\n",
                         et.get_type_name().c_str());
            return false;
        }
        const bool is_half = (et == ov::element::f16);

        const auto& s1 = fmap1.get_shape();
        const int B = static_cast<int>(s1[0]);
        const int H1 = static_cast<int>(s1[1]);
        const int W1 = static_cast<int>(s1[2]);
        const int C = static_cast<int>(s1[3]);

        const size_t elem = is_half ? 2 : 4;
        const size_t bytes_f1 =
            s1.size() ? fmap1.get_byte_size() : 0;
        const size_t bytes_f2 =
            fmap2_packed.get_shape().size() ? fmap2_packed.get_byte_size() : 0;
        const size_t bytes_co =
            coords.get_shape().size()
                ? coords.get_shape()[0] * coords.get_shape()[1] *
                      coords.get_shape()[2] * coords.get_shape()[3] * elem
                : 0;
        const size_t bytes_out =
            corr.get_shape().size() ? corr.get_byte_size() : 0;
        if (bytes_f1 == 0 || bytes_f2 == 0 || bytes_co == 0 ||
            bytes_out == 0) {
            return true;
        }

        sycl::queue& q = get_queue();
        SharedUsmPool& pool = get_pool();

        // Optional per-call stage timing (LOCALCORR_SYCL_TIMING=1):
        // copies/convert, kernel launch, wait; microseconds on stderr.
        const bool timing = std::getenv("LOCALCORR_SYCL_TIMING") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        auto t1 = t0;
        auto t2 = t0;
        auto t3 = t0;

        // Pooled staging buffers. The queue is in-order and every call
        // waits for its own work before returning, so a buffer is never
        // in flight when it is reacquired; concurrent infer requests get
        // distinct buffers from the pool.
        void* d_f1 = pool.acquire(bytes_f1);
        void* d_f2 = pool.acquire(bytes_f2);
        void* d_co = pool.acquire(bytes_co);
        void* d_out = pool.acquire(bytes_out);

        bool ok = (d_f1 != nullptr && d_f2 != nullptr && d_co != nullptr &&
                   d_out != nullptr);
        t1 = std::chrono::steady_clock::now();

        if (ok) {
            // Stage the inputs with host memcpys into the shared-USM
            // buffers instead of queue submissions: enqueueing a
            // q.memcpy costs ~230 us of driver overhead each on this
            // stack, while a host memcpy of shared USM is a plain RAM
            // copy (the queue is idle at entry - every call waits for
            // its own work before returning - so there is no race with
            // in-flight device work). Coords may arrive in a different
            // element type than the feature maps (e.g. f32 coords with
            // f16 tensors) and are converted into their staging buffer
            // the same way.
            std::memcpy(d_f1, fmap1.data(), bytes_f1);
            std::memcpy(d_f2, fmap2_packed.data(), bytes_f2);
            if (coords.get_element_type() == et) {
                std::memcpy(d_co, coords.data(), bytes_co);
            } else if (!convert_coords_host(d_co, et, coords.data(),
                                            coords.get_element_type(),
                                            bytes_co / elem)) {
                std::fprintf(stderr,
                             "LocalCorrCudaPyramid: unsupported coords type "
                             "combination %s -> %s\n",
                             coords.get_element_type().get_type_name().c_str(),
                             et.get_type_name().c_str());
                ok = false;
            }
        }

        t2 = std::chrono::steady_clock::now();

        auto t2b = t2;
        if (ok) {
            ok = launchLocalCorrCudaPyramidForward(
                d_f1, d_f2, d_co, d_out, B, H1, W1, C,
                static_cast<int>(m_radius), static_cast<int>(m_levels),
                static_cast<int>(m_dilation), is_half, m_ymajor != 0, &q);
            if (!ok) {
                std::fprintf(stderr,
                             "LocalCorrCudaPyramid: SYCL kernel launch failed\n");
            }
            // The single synchronization point of the call; the kernel is
            // the only queued operation. Its result is copied back from
            // the shared staging buffer with a host memcpy after the wait
            // (same reasoning as the input staging above).
            t2b = std::chrono::steady_clock::now();
            q.wait();
            if (ok) {
                std::memcpy(corr.data(), d_out, bytes_out);
            }
        }

        t3 = std::chrono::steady_clock::now();
        if (timing) {
            std::fprintf(stderr,
                         "LocalCorrCudaPyramid: %dx%dx%dx%d acquire=%ldus "
                         "copy=%ldus launch=%ldus wait=%ldus\n",
                         B, H1, W1, C,
                         std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
                         std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
                         std::chrono::duration_cast<std::chrono::microseconds>(t2b - t2).count(),
                         std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2b).count());
        }

        pool.release(d_f1, bytes_f1);
        pool.release(d_f2, bytes_f2);
        pool.release(d_co, bytes_co);
        pool.release(d_out, bytes_out);
        return ok;
    }

private:
    static sycl::queue& get_queue() {
        static std::once_flag once;
        static sycl::queue* queue = nullptr;
        std::call_once(once, [] {
            const char* env = std::getenv("LOCALCORR_SYCL_DEVICE");
            const std::string dev = env ? env : "auto";
            sycl::device device = sycl::device(sycl::default_selector_v);
            try {
                if (dev == "gpu") {
                    device = sycl::device(sycl::gpu_selector_v);
                } else if (dev == "cpu") {
                    device = sycl::device(sycl::cpu_selector_v);
                } else if (dev == "default") {
                    device = sycl::device(sycl::default_selector_v);
                } else {
                    // "auto" (default): prefer the GPU, where the
                    // correlation kernel is compute-bound and the model
                    // usually already runs on the OpenVINO GPU plugin.
                    // The CPU OpenCL runtime competes for the host cores
                    // with the OpenVINO CPU-plugin threads executing this
                    // very op (measured 3x slower end-to-end on a Core
                    // Ultra 9 285H + Arc iGPU).
                    try {
                        device = sycl::device(sycl::gpu_selector_v);
                    } catch (const sycl::exception&) {
                        device = sycl::device(sycl::cpu_selector_v);
                    }
                }
            } catch (const sycl::exception& exc) {
                std::fprintf(stderr,
                             "LocalCorrCudaPyramid: SYCL device selection "
                             "failed (%s); using the default device\n",
                             exc.what());
                device = sycl::device(sycl::default_selector_v);
            }
            queue = new sycl::queue(device,
                                    sycl::property::queue::in_order());
            g_exit_drain_queue = queue;
            std::atexit(drain_queue_at_exit);
            std::fprintf(stderr,
                         "LocalCorrCudaPyramid: SYCL queue on %s\n",
                         device.get_info<sycl::info::device::name>().c_str());
        });
        return *queue;
    }

    static SharedUsmPool& get_pool() {
        static std::once_flag once;
        static SharedUsmPool* pool = nullptr;
        std::call_once(once, [] { pool = new SharedUsmPool(get_queue()); });
        return *pool;
    }

    // Converts coords between f32 and f16 on the host (tiny tensor).
    static bool convert_coords_host(void* dst,
                                    const ov::element::Type& dst_type,
                                    const void* src,
                                    const ov::element::Type& src_type,
                                    size_t n) {
        if (src_type == ov::element::f32 && dst_type == ov::element::f16) {
            auto* d = static_cast<sycl::half*>(dst);
            auto* s = static_cast<const float*>(src);
            for (size_t i = 0; i < n; ++i) d[i] = sycl::half(s[i]);
        } else if (src_type == ov::element::f16 &&
                   dst_type == ov::element::f32) {
            auto* d = static_cast<float*>(dst);
            auto* s = static_cast<const sycl::half*>(src);
            for (size_t i = 0; i < n; ++i) d[i] = static_cast<float>(s[i]);
        } else {
            std::fprintf(stderr,
                         "LocalCorrCudaPyramid: unsupported coords type "
                         "combination %s -> %s\n",
                         src_type.get_type_name().c_str(), dst_type.get_type_name().c_str());
            return false;
        }
        return true;
    }
    int64_t m_radius = 4;
    int64_t m_levels = 1;
    int64_t m_ymajor = 0;
    int64_t m_dilation = 1;
};

}  // namespace

const std::vector<std::shared_ptr<ov::Extension>> kExtensions{
    // ONNX frontend: custom_ops::LocalCorrCudaPyramid -> this op.
    std::make_shared<ov::frontend::onnx::OpExtension<LocalCorrCudaPyramidOp>>(
        "LocalCorrCudaPyramid", "custom_ops"),
    // IR frontend: resolve the op when reading serialized XML.
    std::make_shared<ov::OpExtension<LocalCorrCudaPyramidOp>>(),
};

OPENVINO_CREATE_EXTENSIONS(kExtensions)
