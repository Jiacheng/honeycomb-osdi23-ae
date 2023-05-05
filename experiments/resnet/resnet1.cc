#include "experiments/platform.h"
#include "experiments/resnet/resnet_inference.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"

#include <absl/status/status.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gpumpc::experiment {
static const char *kModules[] = {
    "convolution.bin",
    "miopen_asm_conv.bin",
    "elementwise.bin",
    "kernel_replacement.bin",
};

class ResNet1Inference : public ResNetInferenceImpl {
  public:
    explicit ResNet1Inference(bool use_orig_kernel = false)
        : use_orig_kernel_(use_orig_kernel) {}
    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;
    virtual absl::Status Run(absl::Span<const char> image) override;
    virtual absl::Status Fetch(absl::Span<char> result) override;

  private:
    enum {
        kConvGEMMPrepareOrder0,
        kConvGEMMPrepareOrder1,
    };
    enum { kElementWiseSize = 65536 };

    absl::Status LoadBinary();

    absl::Status ConvolutionForwardRun1();
    absl::Status ConvolutionForwardRun2();
    absl::Status ConvolutionForwardImplicitGEMMPrepare(hipDeviceptr_t buf,
                                                       int order);
    absl::Status
    ConvolutionForwardImplicitGEMM(std::initializer_list<hipDeviceptr_t> buf);
    absl::Status
    ConvolutionForwardReplacement1(std::initializer_list<hipDeviceptr_t> buf);
    absl::Status
    ConvolutionForwardReplacement2(std::initializer_list<hipDeviceptr_t> buf);
    absl::Status MIOpenBatchNormFwdInferSpatialEstRun1();
    absl::Status MIOpenBatchNormFwdInferSpatialEstRun2();
    absl::Status MIOpenBatchNormFwdInferSpatialEstRun3();
    absl::Status MIOpenBatchNormFwdInferSpatialEst(
        std::initializer_list<hipDeviceptr_t> buf);
    absl::Status MIOpenSp3Conv();
    absl::Status WinogradConv2dReplacement();
    absl::Status AddAlpha();
    absl::Status Relu();

    static const FunctionDescriptor kFunctions[];
    bool use_orig_kernel_;
    hipDeviceptr_t mem_[2];
};

const ResNetInference::FunctionDescriptor ResNet1Inference::kFunctions[] = {
    {"convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw_"
     "prepare",
     0},
    {"convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw", 0},
    {"batch_norm_opt_resnet1", 2},
    {"miopenSp3AsmConv_v21_1_3_gfx10_fp32_stride1", 1},
    {"addalpha", 2},
    {"relu", 2},
    {"convolution_forward_replacement1", 2},
    {"winograd_conv2d_nchw_128_128_2_3_1", 3},
    {"convolution_forward_replacement2", 2},
};

static absl::Status fetch(hipDeviceptr_t mem, int offset, int size,
                          absl::Span<char> result) {
    MonadRunner<hipError_t> runner(hipSuccess);
    runner
        .Run([&]() {
            return hipMemcpyDtoH(result.data(),
                                 reinterpret_cast<char *>(mem) + offset, size);
        })
        .Run([]() { return hipDeviceSynchronize(); });

    if (runner.code() != hipSuccess) {
        return absl::InternalError("Cannot fetch the result");
    }
    return absl::OkStatus();
}

absl::Status ResNet1Inference::Fetch(absl::Span<char> result) {
    // res = res.to('cpu')
    if (result.size() < kResultSize) {
        return absl::OutOfRangeError("Out of bounds");
    }
    return fetch(mem_[1], 1048576, kResultSize, result);
}

absl::Status ResNet1Inference::Initialize() {
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return LoadBinary(); })
        .Run([&]() {
            return HipErrToStatus(hipMalloc((void **)&mem_[0], 2097152));
        })
        .Run([&]() {
            return HipErrToStatus(hipMalloc((void **)&mem_[1], 2097152));
        })
        .Run([&]() {
            // model = model.to(device)
            std::vector<char> weight;
            auto weight_path =
                std::string(kDataPrefix) + "/weight/weight_resnet1.dat";
            auto stat = ExperimentPlatform::Instance().LoadResource(weight_path,
                                                                    &weight);
            if (!stat.ok()) {
                return stat;
            }
            return HipErrToStatus(
                hipMemcpyHtoD(reinterpret_cast<char *>(mem_[0]) + 294912,
                              weight.data(), 925184));
        })
        .Run([&]() { return HipErrToStatus(hipDeviceSynchronize()); });
    return runner.code();
}

absl::Status ResNet1Inference::Close() {
    for (size_t i = 0; i < sizeof(mem_) / sizeof(hipDeviceptr_t); i++) {
        auto _ = hipFree(mem_[i]);
        (void)_;
        mem_[i] = 0;
    }
    return DestroyFunctions();
}

absl::Status ResNet1Inference::LoadBinary() {
    std::vector<std::string> module_path;
    auto num_modules = sizeof(kModules) / sizeof(const char *);
    std::for_each(kModules, kModules + num_modules,
                  [&module_path](const char *n) {
                      auto prefix = std::string(kDataPrefix) + "/binary/";
                      module_path.push_back(prefix + n);
                  });

    module_.resize(num_modules);
    return LoadFunctions(absl::MakeSpan(module_path),
                         absl::MakeConstSpan(kFunctions));
}

absl::Status ResNet1Inference::Run(absl::Span<const char> image) {
    if (image.size() != kImageSize) {
        return absl::InvalidArgumentError("Invalid image");
    }

    // x = x.to(device)
    auto err =
        hipMemcpyHtoD(mem_[0], const_cast<char *>(image.data()), kImageSize);
    if (err != hipSuccess) {
        return absl::InvalidArgumentError(
            "Cannot copy the image to the device");
    }

    // res = model.forward(x)
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return ConvolutionForwardRun1(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun1(); })
        .Run([&]() {
            if (use_orig_kernel_) {
                return MIOpenSp3Conv();
            } else {
                return WinogradConv2dReplacement();
            }
        })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun2(); })
        .Run([&]() { return ConvolutionForwardRun2(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun3(); })
        .Run([&]() { return AddAlpha(); })
        .Run([&]() { return Relu(); });

    return runner.code();
}

absl::Status ResNet1Inference::ConvolutionForwardRun1() {
    auto stat = ConvolutionForwardImplicitGEMMPrepare(
        reinterpret_cast<char *>(mem_[0]) + 1482240, kConvGEMMPrepareOrder0);
    if (!stat.ok()) {
        return stat;
    }
    if (use_orig_kernel_) {
        return ConvolutionForwardImplicitGEMM({
            reinterpret_cast<char *>(mem_[0]) + 294912,
            reinterpret_cast<char *>(mem_[0]) + 0,
            reinterpret_cast<char *>(mem_[0]) + 1220096,
            reinterpret_cast<char *>(mem_[0]) + 1482240,
        });
    } else {
        return ConvolutionForwardReplacement1({
            reinterpret_cast<char *>(mem_[0]) + 294912,
            reinterpret_cast<char *>(mem_[0]) + 0,
            reinterpret_cast<char *>(mem_[0]) + 1220096,
            reinterpret_cast<char *>(mem_[0]) + 1482240,
        });
    }
}

absl::Status ResNet1Inference::ConvolutionForwardRun2() {
    auto stat = ConvolutionForwardImplicitGEMMPrepare(
        reinterpret_cast<char *>(mem_[0]) + 2006528, kConvGEMMPrepareOrder1);
    if (!stat.ok()) {
        return stat;
    }
    if (use_orig_kernel_) {
        return ConvolutionForwardImplicitGEMM({
            reinterpret_cast<char *>(mem_[0]) + 1184768,
            reinterpret_cast<char *>(mem_[0]) + 0,
            reinterpret_cast<char *>(mem_[1]) + 262144,
            reinterpret_cast<char *>(mem_[0]) + 2006528,
        });
    } else {
        return ConvolutionForwardReplacement2({
            reinterpret_cast<char *>(mem_[0]) + 1184768,
            reinterpret_cast<char *>(mem_[0]) + 0,
            reinterpret_cast<char *>(mem_[1]) + 262144,
            reinterpret_cast<char *>(mem_[0]) + 2006528,
        });
    }
}

absl::Status
ResNet1Inference::ConvolutionForwardImplicitGEMMPrepare(hipDeviceptr_t buf,
                                                        int order) {
    static const unsigned kOrder0[] = {3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 0};
    static const unsigned kOrder1[] = {1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 0};
    struct {
        unsigned s0[5];
        unsigned order[11];
        hipDeviceptr_t buf;
    } args = {
        .s0 = {128, 64, 3, 3, 128},
        .buf = buf,
    };
    auto t = order == kConvGEMMPrepareOrder0 ? kOrder0 : kOrder1;
    std::copy(t, t + sizeof(kOrder0) / sizeof(unsigned), args.order);

    return LaunchKernel(func_[0], 1, 1, args);
}

absl::Status ResNet1Inference::ConvolutionForwardImplicitGEMM(
    std::initializer_list<hipDeviceptr_t> buf) {
    struct {
        hipDeviceptr_t buf[4];
    } args;
    std::copy(buf.begin(), buf.end(), args.buf);
    return LaunchKernel(func_[1], 4, 256, args);
}

absl::Status ResNet1Inference::ConvolutionForwardReplacement1(
    std::initializer_list<hipDeviceptr_t> buf) {
    struct {
        hipDeviceptr_t buf[4];
    } args;
    std::copy(buf.begin(), buf.end(), args.buf);
    return LaunchKernel(func_[6], 4, 256, args);
}

absl::Status ResNet1Inference::ConvolutionForwardReplacement2(
    std::initializer_list<hipDeviceptr_t> buf) {
    struct {
        hipDeviceptr_t buf[4];
    } args;
    std::copy(buf.begin(), buf.end(), args.buf);
    return LaunchKernel(func_[8], 4, 256, args);
}

absl::Status ResNet1Inference::MIOpenBatchNormFwdInferSpatialEstRun1() {
    return MIOpenBatchNormFwdInferSpatialEst({
        reinterpret_cast<char *>(mem_[0]) + 1220096,
        reinterpret_cast<char *>(mem_[0]) + 1482240,
        reinterpret_cast<char *>(mem_[0]) + 590848,
        reinterpret_cast<char *>(mem_[0]) + 591360,
        reinterpret_cast<char *>(mem_[0]) + 589824,
        reinterpret_cast<char *>(mem_[0]) + 590336,
    });
}

absl::Status ResNet1Inference::MIOpenBatchNormFwdInferSpatialEstRun2() {
    return MIOpenBatchNormFwdInferSpatialEst({
        reinterpret_cast<char *>(mem_[0]) + 1744384,
        reinterpret_cast<char *>(mem_[1]),
        reinterpret_cast<char *>(mem_[0]) + 1183232,
        reinterpret_cast<char *>(mem_[0]) + 1183744,
        reinterpret_cast<char *>(mem_[0]) + 1182208,
        reinterpret_cast<char *>(mem_[0]) + 1182720,
    });
}

absl::Status ResNet1Inference::MIOpenBatchNormFwdInferSpatialEstRun3() {
    return MIOpenBatchNormFwdInferSpatialEst({
        reinterpret_cast<char *>(mem_[1]) + 262144,
        reinterpret_cast<char *>(mem_[1]) + 524288,
        reinterpret_cast<char *>(mem_[0]) + 1218560,
        reinterpret_cast<char *>(mem_[0]) + 1219072,
        reinterpret_cast<char *>(mem_[0]) + 1217536,
        reinterpret_cast<char *>(mem_[0]) + 1218048,
    });
}

absl::Status ResNet1Inference::MIOpenBatchNormFwdInferSpatialEst(
    std::initializer_list<hipDeviceptr_t> buf) {
    struct {
        union {
            struct {
                hipDeviceptr_t in, out, est_mean, est_variance, scale, bias;
            };
            hipDeviceptr_t buf[6];
        } buf;
        double threshold;
    } args = {
        .threshold = 1e-5,
    };
    std::copy(buf.begin(), buf.end(), args.buf.buf);
    return LaunchKernel(func_[2], 128, 128, args);
}

absl::Status ResNet1Inference::MIOpenSp3Conv() {
    struct {
        unsigned s0[8];
        hipDeviceptr_t buf[3];
        unsigned long pad0;
        unsigned s1[8];
        unsigned long pad1[5];
        unsigned s2[16];
        unsigned long pad2[6];
    } args = {
        .s0 = {128, 128, 2, 2, 128, 80, 1536, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[0]) + 1482240,
                reinterpret_cast<char *>(mem_[0]) + 592384,
                reinterpret_cast<char *>(mem_[0]) + 1744384,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 2, 2, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                2048,
                16,
                8,
                4,
                4608,
                36,
                12,
                4,
                2048,
                16,
                8,
                4,
                1,
                2048,
                589824,
                2048,
            },
        .pad2 =
            {
                0,
            },
    };
    return LaunchKernel(func_[3], 80, 256, args);
}

absl::Status ResNet1Inference::WinogradConv2dReplacement() {
    struct {
        hipDeviceptr_t buf[3];
    } args;
    args.buf[0] = reinterpret_cast<char *>(mem_[0]) + 592384;
    args.buf[1] = reinterpret_cast<char *>(mem_[0]) + 1482240;
    args.buf[2] = reinterpret_cast<char *>(mem_[0]) + 1744384;
    return LaunchKernel(func_[7], 80, 256, args);
}

absl::Status ResNet1Inference::AddAlpha() {
    struct {
        hipDeviceptr_t dst;
        hipDeviceptr_t a;
        hipDeviceptr_t b;
        float alpha;
    } args = {
        .dst = reinterpret_cast<char *>(mem_[1]) + 786432,
        .a = reinterpret_cast<char *>(mem_[1]) + 524288,
        .b = reinterpret_cast<char *>(mem_[1]) + 0,
        .alpha = 1.0,
    };
    return LaunchKernel(func_[4], kElementWiseSize / 256, 256, args);
}

absl::Status ResNet1Inference::Relu() {
    struct {
        hipDeviceptr_t dst;
        hipDeviceptr_t src;
    } args = {
        .dst = reinterpret_cast<char *>(mem_[1]) + 1048576,
        .src = reinterpret_cast<char *>(mem_[1]) + 786432,
    };
    return LaunchKernel(func_[5], 64, 256, args);
}

std::unique_ptr<ResNetInference> NewResNet1() {
    return std::unique_ptr<ResNetInference>(new ResNet1Inference(false));
}
std::unique_ptr<ResNetInference> NewResNet1Baseline() {
    return std::unique_ptr<ResNetInference>(new ResNet1Inference(true));
}

} // namespace gpumpc::experiment
