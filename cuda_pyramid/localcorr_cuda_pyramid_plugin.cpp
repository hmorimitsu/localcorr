// TensorRT IPluginV3 plugin exposing the fused multi-level variant of the
// alt_cuda_corr local correlation kernel.
//
// Corresponds to ONNX nodes "custom_ops::LocalCorrCudaPyramid" produced by
// FusedAlternateCorrBlock in ptlflow/models/raft/corr.py. Inputs:
//   0 = fmap1        [B, H1, W1, C]
//   1 = fmap2_packed [B, C * sum_i (H1>>i) * (W1>>i)]  (NHWC levels, level 0
//                     first, each level's H/W halved)
//   2 = coords       [B, 2, H1, W1] (NCHW, (x, y), level-0 scale)
// Output:
//   corr [B, levels, RD*RD, H1, W1]
//
// The radius/levels/ymajor attributes are forwarded by the ONNX parser to
// createPlugin() as PluginFields (see alt_cuda_corr_plugin.cpp for the
// attribute-to-field mapping notes).

#include "localcorr_cuda_pyramid_kernel.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

constexpr char kPluginName[] = "LocalCorrCudaPyramid";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";

class LocalCorrCudaPyramidPlugin : public IPluginV3,
                               public IPluginV3OneCore,
                               public IPluginV3OneBuild,
                               public IPluginV3OneRuntime
{
public:
    LocalCorrCudaPyramidPlugin(int radius, int levels, int ymajor, int dilation)
        : radius_(radius), levels_(levels), ymajor_(ymajor), dilation_(dilation) {}

    // ------------------------------------------------------------------
    // IPluginV3
    // ------------------------------------------------------------------
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
            case PluginCapabilityType::kCORE:
                return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityType::kBUILD:
                return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityType::kRUNTIME:
            default:
                return static_cast<IPluginV3OneRuntime*>(this);
        }
    }

    IPluginV3* clone() noexcept override {
        return new LocalCorrCudaPyramidPlugin(radius_, levels_, ymajor_, dilation_);
    }

    // ------------------------------------------------------------------
    // IPluginV3OneCore
    // ------------------------------------------------------------------
    AsciiChar const* getPluginName() const noexcept override {
        return kPluginName;
    }

    AsciiChar const* getPluginVersion() const noexcept override {
        return kPluginVersion;
    }

    AsciiChar const* getPluginNamespace() const noexcept override {
        return kPluginNamespace;
    }

    // ------------------------------------------------------------------
    // IPluginV3OneBuild
    // ------------------------------------------------------------------
    int32_t getNbOutputs() const noexcept override {
        return 1;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes,
        int32_t nbOutputs,
        DataType const* inputTypes,
        int32_t nbInputs) const noexcept override {
        if (nbInputs != 3 || nbOutputs != 1) {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(
        DimsExprs const* inputs,
        int32_t nbInputs,
        DimsExprs const* shapeInputs,
        int32_t nbShapeInputs,
        DimsExprs* outputs,
        int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override {
        // inputs: 0 = fmap1 [B, H1, W1, C]
        //         1 = fmap2_packed [B, C * sum_i h_i * w_i]
        //         2 = coords [B, 2, H1, W1]
        // output: corr [B, levels, RD*RD, H1, W1]
        if (nbInputs != 3 || nbOutputs != 1) {
            return 1;
        }
        if (inputs[0].nbDims != 4 || inputs[2].nbDims != 4) {
            return 1;
        }
        outputs[0].nbDims = 5;
        outputs[0].d[0] = inputs[2].d[0];
        outputs[0].d[1] = exprBuilder.constant(levels_);
        outputs[0].d[2] = exprBuilder.constant((2 * radius_ + 1) * (2 * radius_ + 1));
        outputs[0].d[3] = inputs[2].d[2];
        outputs[0].d[4] = inputs[2].d[3];
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos,
        DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override {
        if (nbInputs != 3 || nbOutputs != 1) {
            return false;
        }
        if (pos < 0 || pos > nbInputs) {
            return false;
        }
        PluginTensorDesc const& desc = inOut[pos].desc;
        if (desc.format != TensorFormat::kLINEAR) {
            return false;
        }
        if (pos == 0) {
            return desc.type == DataType::kFLOAT || desc.type == DataType::kHALF;
        }
        // All remaining inputs and the output must match input 0.
        return desc.type == inOut[0].desc.type;
    }

    int32_t configurePlugin(
        DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override {
        return 0;
    }

    // ------------------------------------------------------------------
    // IPluginV3OneRuntime
    // ------------------------------------------------------------------
    int32_t onShapeChange(
        PluginTensorDesc const* in,
        int32_t nbInputs,
        PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override {
        return 0;
    }

    int32_t enqueue(
        PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override {
        Dims const& fmap1Dims = inputDesc[0].dims;
        Dims const& coordsDims = inputDesc[2].dims;
        if (fmap1Dims.nbDims != 4 || coordsDims.nbDims != 4) {
            return 1;
        }

        const int batch = static_cast<int>(fmap1Dims.d[0]);
        const int h1 = static_cast<int>(fmap1Dims.d[1]);
        const int w1 = static_cast<int>(fmap1Dims.d[2]);
        const int channels = static_cast<int>(fmap1Dims.d[3]);

        const bool isHalf = inputDesc[0].type == DataType::kHALF;
        const bool ok = launchLocalCorrCudaPyramidForward(
            inputs[0],
            inputs[1],
            inputs[2],
            outputs[0],
            batch,
            h1,
            w1,
            channels,
            radius_,
            levels_,
            dilation_,
            isHalf,
            ymajor_ != 0,
            stream);
        return ok ? 0 : 1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        mFieldsToSerialize.clear();
        mFieldsToSerialize.emplace_back(
            "radius", &radius_, PluginFieldType::kINT32, 1);
        mFieldsToSerialize.emplace_back(
            "levels", &levels_, PluginFieldType::kINT32, 1);
        mFieldsToSerialize.emplace_back(
            "ymajor", &ymajor_, PluginFieldType::kINT32, 1);
        mFieldsToSerialize.emplace_back(
            "dilation", &dilation_, PluginFieldType::kINT32, 1);
        mFieldCollectionToSerialize.nbFields =
            static_cast<int32_t>(mFieldsToSerialize.size());
        mFieldCollectionToSerialize.fields = mFieldsToSerialize.data();
        return &mFieldCollectionToSerialize;
    }

private:
    int radius_;
    int levels_;
    int ymajor_;
    int dilation_;
    std::vector<PluginField> mFieldsToSerialize;
    PluginFieldCollection mFieldCollectionToSerialize{};
};

int readIntPluginField(
    PluginFieldCollection const* fc,
    std::initializer_list<char const*> names,
    int defaultValue) {
    if (fc == nullptr) {
        return defaultValue;
    }
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        PluginField const& field = fc->fields[i];
        if (field.data == nullptr) {
            continue;
        }
        bool matches = false;
        for (char const* name : names) {
            if (std::strcmp(field.name, name) == 0) {
                matches = true;
                break;
            }
        }
        if (!matches) {
            continue;
        }
        // The ONNX parser may forward int64 attributes as kINT64 (or
        // kINT32, depending on the parser version), so honor the declared
        // type.
        switch (field.type) {
            case PluginFieldType::kINT64:
                return static_cast<int>(*static_cast<int64_t const*>(field.data));
            case PluginFieldType::kINT32:
                return *static_cast<int32_t const*>(field.data);
            case PluginFieldType::kINT16:
                return *static_cast<int16_t const*>(field.data);
            case PluginFieldType::kINT8:
                return *static_cast<int8_t const*>(field.data);
            default:
                // Fallback: read little-endian int32 (covers parsers that
                // pass the raw ONNX int64 storage as kUNKNOWN).
                return *static_cast<int32_t const*>(field.data);
        }
    }
    return defaultValue;
}

class LocalCorrCudaPyramidPluginCreator : public IPluginCreatorV3One {
public:
    LocalCorrCudaPyramidPluginCreator() {
        mPluginAttributes.emplace_back(
            PluginField{"radius", nullptr, PluginFieldType::kINT32, 1});
        mPluginAttributes.emplace_back(
            PluginField{"levels", nullptr, PluginFieldType::kINT32, 1});
        mPluginAttributes.emplace_back(
            PluginField{"ymajor", nullptr, PluginFieldType::kINT32, 1});
        mPluginAttributes.emplace_back(
            PluginField{"dilation", nullptr, PluginFieldType::kINT32, 1});
        mFieldCollection.nbFields =
            static_cast<int32_t>(mPluginAttributes.size());
        mFieldCollection.fields = mPluginAttributes.data();
    }

    AsciiChar const* getPluginName() const noexcept override {
        return kPluginName;
    }

    AsciiChar const* getPluginVersion() const noexcept override {
        return kPluginVersion;
    }

    AsciiChar const* getPluginNamespace() const noexcept override {
        return kPluginNamespace;
    }

    PluginFieldCollection const* getFieldNames() noexcept override {
        return &mFieldCollection;
    }

    IPluginV3* createPlugin(
        AsciiChar const* name,
        PluginFieldCollection const* fc,
        TensorRTPhase phase) noexcept override {
        return new LocalCorrCudaPyramidPlugin(
            readIntPluginField(fc, {"radius", "r", "r_i"}, 4),
            readIntPluginField(fc, {"levels", "num_levels"}, 4),
            readIntPluginField(fc, {"ymajor"}, 0),
            readIntPluginField(fc, {"dilation"}, 1));
    }

private:
    std::vector<PluginField> mPluginAttributes;
    PluginFieldCollection mFieldCollection{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(LocalCorrCudaPyramidPluginCreator);
