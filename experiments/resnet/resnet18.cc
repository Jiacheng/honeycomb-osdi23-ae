#include "experiments/platform.h"
#include "experiments/resnet/resnet_inference.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"

#include <absl/status/status.h>
#include <absl/types/span.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gpumpc::experiment {

static const char *kModules[] = {
    "resnet_gen_kernels.bin", // 0
    "Cijk_Ailk_Bljk.bin",     // 1
    "Cijk_S.bin",             // 2
    "miopen_batch_norm.bin",  // 3
    "meanops.bin",            // 4
    "miopen_asm_conv.bin",    // 5
    "elementwise.bin",        // 6
};

class ResNet18Inference : public ResNetInferenceImpl {
  public:
    enum {
        kImageSize = 602112,
        kResultSize = 4000,
        kWeightNum = 6,
    };

    struct MemRegion {
        unsigned idx;
        size_t offset;
        MemRegion(unsigned idx, size_t start) : idx(idx), offset(start) {}
    };

    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;
    virtual absl::Status Run(absl::Span<const char> image) override;
    virtual absl::Status RunDirect() override;
    virtual absl::Status Fetch(absl::Span<char> result) override;

    virtual void *GetImageSourceBufferAddr() override { return mem_[0]; }
    virtual void *GetResultBufferAddr() override {
        return reinterpret_cast<void *>(reinterpret_cast<char *>(mem_[0]) +
                                        2084352);
    }

  private:
    absl::Status LoadBinary();

    absl::Status BasicBlock();
    absl::Status BasicBlock2();
    absl::Status BasicDownBlock();
    absl::Status BasicDownBlock2();

    absl::Status Im2d2Col();
    absl::Status Cijk_entry();
    absl::Status MIOpenBatchNormFwdInferSpatialEstRun();
    absl::Status Relu();
    absl::Status MaxPool2D();
    absl::Status miopenSp3AsmConv_v21_1_3_gfx10_fp32_stride1();
    absl::Status AddAlpha();
    absl::Status TransposeEntry();
    absl::Status Cijk_S();
    absl::Status AdaptiveAvgPool2d();

    static const ResNetInference::FunctionDescriptor kFunctions[];
    hipDeviceptr_t mem_[18];

    // XXX
    // The below are simply hacks to accomodate the original patch.
    //
    // The memory arguments and regions are meant to tied to the operators
    // not when they are invoked.
    //
    // It needs to be addressed quickly
    void ResetExecutionCounter();
    int exec_relu_idx_;
    int exec_conv_idx_;
    int exec_transpose_idx_;
    int exec_cijk_s_idx_;
    int exec_cijk_entry_idx_;
    int exec_batch_norm_idx_;
    int exec_add_alpha_idx_;
    int exec_im2d2col_idx_;
};

const ResNetInference::FunctionDescriptor ResNet18Inference::kFunctions[] = {
    {"Im2d2Col_wg1_x4_imb56_1449_stride1", 0}, // 0: 0
    {"Cijk_Ailk_Bljk_SB_MT128x64x8_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS3_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW4_GSU1_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW4_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU0_"
     "SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT8_8_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW4_WSGRA0_WSGRB0_WS32_WG16_8_1_WGM8",
     1}, // 1: 1
    {"MIOpenBatchNormFwdInferSpatialEst",
     3}, // 2: 2,6,9,13,16,21,24,28,32,35,40,43,47,51,54,59,64,68,74,79
    {"relu_generic", 6}, // 3: 3,7,11,14,18,22,30,33,37,41,49,52,56,60,70,75,81
    {"max_pool_forward_nchw", 6},
    {"miopenSp3AsmConv_v21_1_3_gfx10_fp32_stride1",
     5},                     // 5: 5,8,12,15,23,31,34,42,50,53
    {"addalpha_generic", 6}, // 6: 10,17,29,36,48,55,69,80
    {"Cijk_Ailk_Bljk_SB_MT128x128x16_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS3_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS0_FL0_GRVW4_GSU1_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW4_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU32_"
     "SUM3_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT8_16_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW4_WSGRA0_WSGRB0_WS32_WG16_8_1_WGM8",
     1},                                 // 7: 20
    {"transpose_NCHW2CNHW_V2_2D_WG", 0}, // 8: 25,44,65
    {"Cijk_Ailk_Bljk_SB_MT32x16x8_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS0_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW1_GSU1_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW1_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU0_"
     "SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT2_2_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW1_WSGRA0_WSGRB0_WS32_WG16_8_1_WGM8",
     1},                                        // 9: 26,39
    {"transpose_CNHW2NCHW_V1_1D_WG_float4", 0}, // 10: 27
    {"Cijk_Ailk_Bljk_SB_MT32x32x8_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS0_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW1_GSU1_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW1_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU0_"
     "SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT2_2_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW1_WSGRA0_WSGRB0_WS32_WG16_16_1_WGM1",
     1},                                        // 11: 45
    {"transpose_CNHW2NCHW_V1_2D_WG_float4", 0}, // 12: 46
    {"Cijk_Ailk_Bljk_SB_MT32x16x8_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS0_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW1_GSU1_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW1_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU32_"
     "SUM3_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT2_2_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW1_WSGRA0_WSGRB0_WS32_WG16_8_1_WGM1",
     1},           // 13: 58
    {"Cijk_S", 2}, // 14: 62
    {"Cijk_Ailk_Bljk_SB_MT64x32x8_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS0_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW1_GSU4_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW1_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU0_"
     "SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_4_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW1_WSGRA0_WSGRB0_WS32_WG16_8_1_WGM1",
     1}, // 15: 63,73,78
    {"Cijk_Ailk_Bljk_SB_MT32x16x16_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS0_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW1_GSU1_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW1_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU0_"
     "SUM0_SUS0_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT2_2_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW1_WSGRA0_WSGRB0_WS32_WG16_8_1_WGM4",
     1},                                       // 16: 66
    {"transpose_CNHW2NCHW_V1_1D_WG_float", 0}, // 17: 67
    {"_ZN2at6native13reduce_kernelILi512ELi1ENS0_8ReduceOpIfNS0_"
     "7MeanOpsIffEEjfLi4EEEEEvT1_",
     4}, // 18: 82
    {"Cijk_Alik_Bljk_SB_MT64x64x8_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_"
     "AMAS0_ASAE01_ASCE01_ASEM1_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW1_GSU1_GSUAMB_"
     "GLS0_ISA1030_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW1_FMA_MIAV0_MDA2_NTC0_"
     "NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS0_SU32_"
     "SUM3_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_4_TLDS0_USFGROn1_VAW1_"
     "VSn1_VW1_WSGRA0_WSGRB0_WS32_WG16_16_1_WGM8",
     1},                                      // 19: 83
    {"Im2d2Col_wg1_x1_imb4_1105_stride1", 0}, // 22:57
    {"Im2d2Col_wg1_x1_imb2_1105_stride1", 0}, // 22:57
    {"Im2d2Col_wg1_x1_imb1_1105_stride1", 0}, // 22:57
    {"Im2d2Col_wg4_x1_imb1_544_stride0", 0},  // 22:57
    {"avg_pool_forward_nchw", 6}, //24:
};

void ResNet18Inference::ResetExecutionCounter() {
    exec_relu_idx_ = 0;
    exec_conv_idx_ = 0;
    exec_transpose_idx_ = 0;
    exec_cijk_s_idx_ = 0;
    exec_cijk_entry_idx_ = 0;
    exec_batch_norm_idx_ = 0;
    exec_add_alpha_idx_ = 0;
    exec_im2d2col_idx_ = 0;
}

absl::Status ResNet18Inference::Fetch(absl::Span<char> result) {
    // res = res.to('cpu')
    if (result.size() < kResultSize) {
        return absl::OutOfRangeError("Out of bounds");
    }
    MonadRunner<hipError_t> runner(hipSuccess);
    runner
        .Run([&]() {
            return hipMemcpyDtoH(result.data(),
                                 reinterpret_cast<char *>(mem_[0]) + 2084352,
                                 kResultSize);
        })
        .Run([]() { return hipDeviceSynchronize(); });

    if (runner.code() != hipSuccess) {
        return absl::InternalError("Cannot fetch the result");
    }
    return absl::OkStatus();
}

absl::Status ResNet18Inference::Initialize() {
    auto stat = absl::OkStatus();
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run([&]() {
            // load elf binaries
            return LoadBinary();
        })
        .Run([&]() {
            auto stat = absl::OkStatus();
            const size_t devptr_size[18] = {
                2097152,  2097152, 20971520, 20971520, 2097152, 20971520,
                33554432, 2097152, 2097152,  2097152,  2097152, 2097152,
                2097152,  2097152, 2097152,  2097152,  2097152, 33554432};
            for (int i = 0; i < 18; i++) {
                stat = HipErrToStatus(
                    hipMalloc((void **)&mem_[i], devptr_size[i]));
                if (!stat.ok()) {
                    return stat;
                }
            }
            return stat;
        })
        .Run([&]() {
            enum { kNumWeights = 6 };
            // load weights
            // model = model.to(device)
            static const struct {
                MemRegion loc;
                size_t size;
            } kWeights[kNumWeights] = {
                {MemRegion(0, 0), 2097152},  {MemRegion(1, 0), 2097152},
                {MemRegion(2, 0), 20971520}, {MemRegion(3, 0), 20971520},
                {MemRegion(4, 0), 2097152},  {MemRegion(5, 0), 20971520},
            };
            MonadRunner<absl::Status> r(absl::OkStatus());
            std::vector<char> w;
            for (size_t i = 0; i < kNumWeights; i++) {
                r.Run([&]() {
                     w.clear();
                     auto weight_path = "data/resnet/weight/resnet18/weight_" +
                                        std::to_string(i) + ".dat";
                     return ExperimentPlatform::Instance().LoadResource(
                         weight_path, &w);
                 }).Run([&]() {
                    auto ptr =
                        reinterpret_cast<char *>(mem_[kWeights[i].loc.idx]) +
                        kWeights[i].loc.offset;
                    return HipErrToStatus(
                        hipMemcpyHtoD(ptr, w.data(), kWeights[i].size));
                });
            }
            return r.code();
        })
        .Run([&]() { return HipErrToStatus(hipDeviceSynchronize()); });

    return runner.code();
}

absl::Status ResNet18Inference::Close() {
    for (size_t i = 0; i < sizeof(mem_) / sizeof(hipDeviceptr_t); i++) {
        auto _ = hipFree(mem_[i]);
        (void)_;
        mem_[i] = 0;
    }
    return DestroyFunctions();
}

absl::Status ResNet18Inference::BasicBlock() {
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return miopenSp3AsmConv_v21_1_3_gfx10_fp32_stride1(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return Relu(); })
        .Run([&]() { return miopenSp3AsmConv_v21_1_3_gfx10_fp32_stride1(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return AddAlpha(); })
        .Run([&]() { return Relu(); });
    return runner.code();
}

absl::Status ResNet18Inference::BasicBlock2() {
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return Im2d2Col(); })
        .Run([&]() { return Cijk_S(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return Relu(); })
        .Run([&]() { return Im2d2Col(); })
        .Run([&]() { return Cijk_S(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return AddAlpha(); })
        .Run([&]() { return Relu(); });
    return runner.code();
}

absl::Status ResNet18Inference::BasicDownBlock() {
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return Im2d2Col(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return Relu(); })
        .Run([&]() { return miopenSp3AsmConv_v21_1_3_gfx10_fp32_stride1(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return TransposeEntry(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return TransposeEntry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return AddAlpha(); })
        .Run([&]() { return Relu(); });
    return runner.code();
}

absl::Status ResNet18Inference::BasicDownBlock2() {
    MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([&]() { return Im2d2Col(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return Relu(); })
        .Run([&]() { return Im2d2Col(); })
        .Run([&]() { return Cijk_S(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return TransposeEntry(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return TransposeEntry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return AddAlpha(); })
        .Run([&]() { return Relu(); });
    return runner.code();
}
absl::Status ResNet18Inference::Run(absl::Span<const char> image) {
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
    return RunDirect();
}

absl::Status ResNet18Inference::RunDirect() {
    ResetExecutionCounter();

    // res = model.forward(x)
    MonadRunner<absl::Status> runner(absl::OkStatus());

    // x : 1*3*224*224
    // nn.Conv2d(3, self.inplanes,
    // kernel_size=7,stride=2,padding=3, bias=False)
    runner.Run([&]() { return Im2d2Col(); })
        .Run([&]() { return Cijk_entry(); })
        .Run([&]() { return MIOpenBatchNormFwdInferSpatialEstRun(); })
        .Run([&]() { return Relu(); })
        .Run([&]() { return MaxPool2D(); });

    // layer1
    // input: 1*64*56*56, bb1 + bb2, output: 1*64*56*56
    runner.Run([&]() { return BasicBlock(); }).Run([&]() {
        return BasicBlock();
    });

    // layer2
    // input: 1*64*56*56, basicdownblock3, output: 1*128*28*28
    // basicblock4, input: 1*128*28*28, output: 1*128*28*28
    runner.Run([&]() { return BasicDownBlock(); }).Run([&]() {
        return BasicBlock();
    });

    // layer3
    // input: 1*128*28*28, basicdownblock5, output: 1*256*14*14
    // basicblock6, input: 1*256*14*14, output: 1*256*14*14
    runner.Run([&]() { return BasicDownBlock(); }).Run([&]() {
        return BasicBlock();
    });

    // layer4
    // input 1*256*14*14 basicdownblock7 output: 1*512*7*7
    // basicblock8 input: 1*512*7*7 output: 1*512*7*7
    runner.Run([&]() { return BasicDownBlock2(); }).Run([&]() {
        return BasicBlock2();
    });

    // avgpool2d input: 1*512*7*7 output: 1*512*1*1
    runner.Run([&]() { return AdaptiveAvgPool2d(); });

    // x = torch.flatten(x, 1)
    // input: 1*512*1*1
    runner
        .Run([&]() {
            return HipErrToStatus(hipMemcpyDtoD(
                reinterpret_cast<char *>(mem_[0]) + 2084352,
                reinterpret_cast<char *>(mem_[1]) + 1969664, kResultSize));
            // output: 1*512
        })
        .Run([&]() {
            // x = self.fc(x)
            // input: 1*512
            return Cijk_entry();
            // output: 1*1000
        });
    return runner.code();
}

absl::Status ResNet18Inference::LoadBinary() {
    std::vector<std::string> module_path;
    auto num_modules = sizeof(kModules) / sizeof(const char *);
    std::for_each(kModules, kModules + num_modules,
                  [&module_path](const char *n) {
                      auto loc = std::string("data/resnet/binary/") + n;
                      module_path.push_back(loc);
                  });

    module_.resize(num_modules);
    return LoadFunctions(absl::MakeSpan(module_path),
                         absl::MakeConstSpan(kFunctions));
}

absl::Status ResNet18Inference::Im2d2Col() {
    static const struct {
        int grid;
        int func_idx;
        unsigned long data_size_off;
        MemRegion im;
        int size;
        int wei;
        int out;
        int pad;
        int stride;
        int dilation;
        MemRegion col;
    } kInvocations[7] = {
        {168, 0, 0x24c00, MemRegion(0, 0), 0xe0, 7, 0x70, 3, 2, 1,
         MemRegion(5, 9437184)},
        {256, 20, 0x31000, MemRegion(10, 802816), 0x38, 3, 0x1c, 1, 2, 1,
         MemRegion(5, 11042816)},
        {256, 21, 0x18800, MemRegion(11, 802816), 0x1c, 3, 0xe, 1, 2, 1,
         MemRegion(13, 0)},
        {256, 22, 0xc400, MemRegion(15, 602112), 0xe, 3, 7, 1, 2, 1,
         MemRegion(15, 802816)},
        {128, 23, 0x6200, MemRegion(1, 1973760), 7, 3, 7, 1, 1, 1,
         MemRegion(15, 802816)},
        {128, 23, 0x6200, MemRegion(15, 802816), 7, 3, 7, 1, 1, 1,
         MemRegion(15, 1103872)},
        {128, 23, 0x6200, MemRegion(15, 1103872), 7, 3, 7, 1, 1, 1,
         MemRegion(16, 0)},
    };

    auto idx = exec_im2d2col_idx_++;
    auto &k = kInvocations[idx];
    struct {
        unsigned long data_size_off;
        hipDeviceptr_t im;
        int im_offset;
        int h;
        int w;
        int wei_h;
        int wei_w;
        int out_h;
        int out_w;
        int pad_h;
        int pad_w;
        int stride_h;
        int stride_w;
        int dilation_h;
        int dilation_w;
        hipDeviceptr_t col;
    } arg = {
        .data_size_off = k.data_size_off,
        .im = reinterpret_cast<char *>(mem_[k.im.idx]) + k.im.offset,
        .im_offset = 0,
        .h = k.size,
        .w = k.size,
        .wei_h = k.wei,
        .wei_w = k.wei,
        .out_h = k.out,
        .out_w = k.out,
        .pad_h = k.pad,
        .pad_w = k.pad,
        .stride_h = k.stride,
        .stride_w = k.stride,
        .dilation_h = k.dilation,
        .dilation_w = k.dilation,
        .col = reinterpret_cast<char *>(mem_[k.col.idx]) + k.col.offset,
    };
    return LaunchKernel(func_[k.func_idx], k.grid, 256, arg);
}

absl::Status ResNet18Inference::Cijk_entry() {
    struct {
        unsigned long sizeC;
        unsigned long sizeA;
        unsigned long sizeB;
        hipDeviceptr_t D;
        hipDeviceptr_t C;
        hipDeviceptr_t A;
        hipDeviceptr_t B;
        float alpha;
        float beta;
        int strideD0;
        int strideD1;
        int strideC0;
        int strideC1;
        int strideA0;
        int strideA1;
        int strideB0;
        int strideB1;
        int SizesFree0;
        int SizesFree1;
        int SizesFree2;
        int SizesSum0;
        int OrigStaggerUIter;
        int NumWorkGroups0;
        int NumWorkGroups1;
        int NumFullBlocks;
        int WgmRemainder1;
        unsigned MagicNumberWgmRemainder1;
        int OffsetD;
        int OffsetC;
        int OffsetA;
        int OffsetB;
        int padding;
    } args[11] = {
        {
            .sizeC = 0xc4000,
            .sizeA = 0x1c2300,
            .sizeB = 0x24c0,
            .D = reinterpret_cast<char *>(mem_[2]) + 12976128,
            .C = reinterpret_cast<char *>(mem_[2]) + 12976128,
            .A = reinterpret_cast<char *>(mem_[5]) + 9437184,
            .B = reinterpret_cast<char *>(mem_[0]) + 602112,
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x3100,
            .strideD1 = 1,
            .strideC0 = 0x3100,
            .strideC1 = 1,
            .strideA0 = 0x3100,
            .strideA1 = 1,
            .strideB0 = 0x93,
            .strideB1 = 1,
            .SizesFree0 = 0x3100,
            .SizesFree1 = 0x40,
            .SizesFree2 = 1,
            .SizesSum0 = 0x93,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 0x62,
            .NumWorkGroups1 = 1,
            .NumFullBlocks = 0,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x18800,
            .sizeA = 0x6e400,
            .sizeB = 0x12000,
            .D = reinterpret_cast<char *>(mem_[10]) + 1605632,
            .C = reinterpret_cast<char *>(mem_[10]) + 1605632,
            .A = reinterpret_cast<char *>(mem_[5]) + 11042816,
            .B = reinterpret_cast<char *>(mem_[0]) + 1242624,
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x310,
            .strideD1 = 1,
            .strideC0 = 0x310,
            .strideC1 = 1,
            .strideA0 = 0x310,
            .strideA1 = 1,
            .strideB0 = 0x240,
            .strideB1 = 1,
            .SizesFree0 = 0x310,
            .SizesFree1 = 0x80,
            .SizesFree2 = 1,
            .SizesSum0 = 0x240,
            .OrigStaggerUIter = 0xf,
            .NumWorkGroups0 = 7,
            .NumWorkGroups1 = 1,
            .NumFullBlocks = 0,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x18800,
            .sizeA = 0xc400,
            .sizeB = 0x2000,
            .D = reinterpret_cast<char *>(mem_[4]) + 1527808,
            .C = reinterpret_cast<char *>(mem_[4]) + 1527808,
            .A = reinterpret_cast<char *>(mem_[4]) + 1327104,
            .B = reinterpret_cast<char *>(mem_[0]) + 1542656,
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x310,
            .strideD1 = 1,
            .strideC0 = 0x310,
            .strideC1 = 1,
            .strideA0 = 0x310,
            .strideA1 = 1,
            .strideB0 = 0x40,
            .strideB1 = 1,
            .SizesFree0 = 0x310,
            .SizesFree1 = 0x80,
            .SizesFree2 = 1,
            .SizesSum0 = 0x40,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 0x19,
            .NumWorkGroups1 = 8,
            .NumFullBlocks = 1,
            .WgmRemainder1 = 8,
            .MagicNumberWgmRemainder1 = 0x10000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0xc400,
            .sizeA = 0x37200,
            .sizeB = 0x48000,
            .D = reinterpret_cast<char *>(mem_[4]) + 1728512,
            .C = reinterpret_cast<char *>(mem_[4]) + 1728512,
            .A = reinterpret_cast<char *>(mem_[13]),
            .B = reinterpret_cast<char *>(mem_[2]),
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0xc4,
            .strideD1 = 1,
            .strideC0 = 0xc4,
            .strideC1 = 1,
            .strideA0 = 0xc4,
            .strideA1 = 1,
            .strideB0 = 0x480,
            .strideB1 = 1,
            .SizesFree0 = 0xc4,
            .SizesFree1 = 0x100,
            .SizesFree2 = 1,
            .SizesSum0 = 0x480,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 7,
            .NumWorkGroups1 = 0x10,
            .NumFullBlocks = 2,
            .WgmRemainder1 = 8,
            .MagicNumberWgmRemainder1 = 0x10000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0xc400,
            .sizeA = 0x6200,
            .sizeB = 0x8000,
            .D = reinterpret_cast<char *>(mem_[15]) + 100352,
            .C = reinterpret_cast<char *>(mem_[15]) + 100352,
            .A = reinterpret_cast<char *>(mem_[15]),
            .B = reinterpret_cast<char *>(mem_[1]) + 1781248,
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0xc4,
            .strideD1 = 1,
            .strideC0 = 0xc4,
            .strideC1 = 1,
            .strideA0 = 0xc4,
            .strideA1 = 1,
            .strideB0 = 0x80,
            .strideB1 = 1,
            .SizesFree0 = 0xc4,
            .SizesFree1 = 0x100,
            .SizesFree2 = 1,
            .SizesSum0 = 0x80,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 7,
            .NumWorkGroups1 = 8,
            .NumFullBlocks = 8,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x6200,
            .sizeA = 0x1b900,
            .sizeB = 0x120000,
            .D = reinterpret_cast<char *>(mem_[0]) + 1981952,
            .C = reinterpret_cast<char *>(mem_[0]) + 1981952,
            .A = reinterpret_cast<char *>(mem_[15]) + 802816,
            .B = reinterpret_cast<char *>(mem_[2]) + 8257536,
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x31,
            .strideD1 = 1,
            .strideC0 = 0x31,
            .strideC1 = 1,
            .strideA0 = 0x31,
            .strideA1 = 1,
            .strideB0 = 0x900,
            .strideB1 = 1,
            .SizesFree0 = 0x31,
            .SizesFree1 = 0x200,
            .SizesFree2 = 1,
            .SizesSum0 = 0x900,
            .OrigStaggerUIter = 0x1f,
            .NumWorkGroups0 = 2,
            .NumWorkGroups1 = 0x20,
            .NumFullBlocks = 0x20,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x6200,
            .sizeA = 0x37200,
            .sizeB = 0x240000,
            .D = reinterpret_cast<char *>(mem_[4]) + 1929216,
            .C = reinterpret_cast<char *>(mem_[4]) + 1929216,
            .A = reinterpret_cast<char *>(mem_[15]) + 802816,
            .B = reinterpret_cast<char *>(mem_[3]),
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x31,
            .strideD1 = 1,
            .strideC0 = 0x31,
            .strideC1 = 1,
            .strideA0 = 0x31,
            .strideA1 = 1,
            .strideB0 = 0x1200,
            .strideB1 = 1,
            .SizesFree0 = 0x31,
            .SizesFree1 = 0x200,
            .SizesFree2 = 1,
            .SizesSum0 = 0x1200,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 1,
            .NumWorkGroups1 = 0x10,
            .NumFullBlocks = 0x10,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x6200,
            .sizeA = 0x3100,
            .sizeB = 0x20000,
            .D = reinterpret_cast<char *>(mem_[15]) + 1053696,
            .C = reinterpret_cast<char *>(mem_[15]) + 1053696,
            .A = reinterpret_cast<char *>(mem_[15]) + 1003520,
            .B = reinterpret_cast<char *>(mem_[4]),
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x31,
            .strideD1 = 1,
            .strideC0 = 0x31,
            .strideC1 = 1,
            .strideA0 = 0x31,
            .strideA1 = 1,
            .strideB0 = 0x100,
            .strideB1 = 1,
            .SizesFree0 = 0x31,
            .SizesFree1 = 0x200,
            .SizesFree2 = 1,
            .SizesSum0 = 0x100,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 2,
            .NumWorkGroups1 = 0x20,
            .NumFullBlocks = 8,
            .WgmRemainder1 = 4,
            .MagicNumberWgmRemainder1 = 0x20000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x6200,
            .sizeA = 0x37200,
            .sizeB = 0x240000,
            .D = reinterpret_cast<char *>(mem_[15]) + 1003520,
            .C = reinterpret_cast<char *>(mem_[15]) + 1003520,
            .A = reinterpret_cast<char *>(mem_[15]) + 1103872,
            .B = reinterpret_cast<char *>(mem_[3]) + 9437184,
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x31,
            .strideD1 = 1,
            .strideC0 = 0x31,
            .strideC1 = 1,
            .strideA0 = 0x31,
            .strideA1 = 1,
            .strideB0 = 0x1200,
            .strideB1 = 1,
            .SizesFree0 = 0x31,
            .SizesFree1 = 0x200,
            .SizesFree2 = 1,
            .SizesSum0 = 0x1200,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 1,
            .NumWorkGroups1 = 0x10,
            .NumFullBlocks = 0x10,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x6200,
            .sizeA = 0x37200,
            .sizeB = 0x240000,
            .D = reinterpret_cast<char *>(mem_[15]) + 1204224,
            .C = reinterpret_cast<char *>(mem_[15]) + 1204224,
            .A = reinterpret_cast<char *>(mem_[16]),
            .B = reinterpret_cast<char *>(mem_[5]),
            .alpha = 1.0,
            .beta = 0.0,
            .strideD0 = 0x31,
            .strideD1 = 1,
            .strideC0 = 0x31,
            .strideC1 = 1,
            .strideA0 = 0x31,
            .strideA1 = 1,
            .strideB0 = 0x1200,
            .strideB1 = 1,
            .SizesFree0 = 0x31,
            .SizesFree1 = 0x200,
            .SizesFree2 = 1,
            .SizesSum0 = 0x1200,
            .OrigStaggerUIter = 0,
            .NumWorkGroups0 = 1,
            .NumWorkGroups1 = 0x10,
            .NumFullBlocks = 0x10,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
        {
            .sizeC = 0x3e8,
            .sizeA = 0x7d000,
            .sizeB = 0x200,
            .D = reinterpret_cast<char *>(mem_[0]) + 2084352,
            .C = reinterpret_cast<char *>(mem_[0]) + 2084352,
            .A = reinterpret_cast<char *>(mem_[3]) + 18874368,
            .B = reinterpret_cast<char *>(mem_[0]) + 2082304,
            .alpha = 1.0,
            .beta = 1.0,
            .strideD0 = 0x3e8,
            .strideD1 = 0,
            .strideC0 = 0x3e8,
            .strideC1 = 0,
            .strideA0 = 0x200,
            .strideA1 = 0,
            .strideB0 = 0x200,
            .strideB1 = 0,
            .SizesFree0 = 0x3e8,
            .SizesFree1 = 1,
            .SizesFree2 = 1,
            .SizesSum0 = 0x200,
            .OrigStaggerUIter = 0xf,
            .NumWorkGroups0 = 0x10,
            .NumWorkGroups1 = 1,
            .NumFullBlocks = 0,
            .WgmRemainder1 = 1,
            .MagicNumberWgmRemainder1 = 0x80000001,
            .OffsetD = 0,
            .OffsetC = 0,
            .OffsetA = 0,
            .OffsetB = 0,
            .padding = 0,
        },
    };
    const static int CijkBinaryIdx[] = {1, 7, 9, 9, 11, 13, 15, 16, 15, 15, 19};
    const static Dimension CijkGlobalWorkSizeConfig[] = {98,
                                                         7,
                                                         Dimension(25, 8),
                                                         Dimension(7, 16),
                                                         Dimension(7, 8),
                                                         Dimension(2, 32),
                                                         Dimension(1, 64),
                                                         Dimension(2, 32),
                                                         Dimension(1, 64),
                                                         Dimension(1, 64),
                                                         16};
    const static int CijkBlockDimSizeConfig[] = {128, 128, 128, 128, 256, 128,
                                                 128, 128, 128, 128, 256};
    auto idx = exec_cijk_entry_idx_++;
    return LaunchKernel(func_[CijkBinaryIdx[idx]],
                        CijkGlobalWorkSizeConfig[idx],
                        CijkBlockDimSizeConfig[idx], args[idx]);
}

absl::Status ResNet18Inference::MIOpenBatchNormFwdInferSpatialEstRun() {
    struct {
        hipDeviceptr_t buf[6];
        double threshold;
        unsigned scalars[3];
        unsigned paddings[15];
    } args[20] =
        {
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[2]) + 12976128,
                        reinterpret_cast<char *>(mem_[2]) + 16187392,
                        reinterpret_cast<char *>(mem_[0]) + 641024,
                        reinterpret_cast<char *>(mem_[0]) + 641536,
                        reinterpret_cast<char *>(mem_[0]) + 640000,
                        reinterpret_cast<char *>(mem_[0]) + 640512,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x3100,
                        0xc4000,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[7]),
                        reinterpret_cast<char *>(mem_[7]) + 802816,
                        reinterpret_cast<char *>(mem_[0]) + 791040,
                        reinterpret_cast<char *>(mem_[0]) + 791552,
                        reinterpret_cast<char *>(mem_[0]) + 790016,
                        reinterpret_cast<char *>(mem_[0]) + 790528,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc40,
                        0x31000,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[8]),
                        reinterpret_cast<char *>(mem_[8]) + 802816,
                        reinterpret_cast<char *>(mem_[0]) + 941056,
                        reinterpret_cast<char *>(mem_[0]) + 941568,
                        reinterpret_cast<char *>(mem_[0]) + 940032,
                        reinterpret_cast<char *>(mem_[0]) + 940544,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc40,
                        0x31000,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[9]),
                        reinterpret_cast<char *>(mem_[9]) + 802816,
                        reinterpret_cast<char *>(mem_[0]) + 1091072,
                        reinterpret_cast<char *>(mem_[0]) + 1091584,
                        reinterpret_cast<char *>(mem_[0]) + 1090048,
                        reinterpret_cast<char *>(mem_[0]) + 1090560,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc40,
                        0x31000,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[10]),
                        reinterpret_cast<char *>(mem_[10]) + 802816,
                        reinterpret_cast<char *>(mem_[0]) + 1241088,
                        reinterpret_cast<char *>(mem_[0]) + 1241600,
                        reinterpret_cast<char *>(mem_[0]) + 1240064,
                        reinterpret_cast<char *>(mem_[0]) + 1240576,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc40,
                        0x31000,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[10]) + 1605632,
                        reinterpret_cast<char *>(mem_[9]) + 1605632,
                        reinterpret_cast<char *>(mem_[0]) + 1538560,
                        reinterpret_cast<char *>(mem_[0]) + 1539072,
                        reinterpret_cast<char *>(mem_[0]) + 1537536,
                        reinterpret_cast<char *>(mem_[0]) + 1538048,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x310,
                        0x18800,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[8]) + 1605632,
                        reinterpret_cast<char *>(mem_[7]) + 1605632,
                        reinterpret_cast<char *>(mem_[0]) + 1541120,
                        reinterpret_cast<char *>(mem_[0]) + 1541632,
                        reinterpret_cast<char *>(mem_[0]) + 1540096,
                        reinterpret_cast<char *>(mem_[0]) + 1540608,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x310,
                        0x18800,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[0]) + 1580544,
                        reinterpret_cast<char *>(mem_[4]) + 1327104,
                        reinterpret_cast<char *>(mem_[0]) + 1576448,
                        reinterpret_cast<char *>(mem_[0]) + 1576960,
                        reinterpret_cast<char *>(mem_[0]) + 1575424,
                        reinterpret_cast<char *>(mem_[0]) + 1575936,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x310,
                        0x18800,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[4]) + 1327104,
                        reinterpret_cast<char *>(mem_[11]),
                        reinterpret_cast<char *>(mem_[0]) + 1579008,
                        reinterpret_cast<char *>(mem_[0]) + 1579520,
                        reinterpret_cast<char *>(mem_[0]) + 1577984,
                        reinterpret_cast<char *>(mem_[0]) + 1578496,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x310,
                        0x18800,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[11]) + 401408,
                        reinterpret_cast<char *>(mem_[11]) + 802816,
                        reinterpret_cast<char *>(mem_[1]) + 1770496,
                        reinterpret_cast<char *>(mem_[1]) + 1771008,
                        reinterpret_cast<char *>(mem_[1]) + 1769472,
                        reinterpret_cast<char *>(mem_[1]) + 1769984,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x310,
                        0x18800,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[4]) + 1728512,
                        reinterpret_cast<char *>(mem_[11]) + 1204224,
                        reinterpret_cast<char *>(mem_[1]) + 1774080,
                        reinterpret_cast<char *>(mem_[1]) + 1775104,
                        reinterpret_cast<char *>(mem_[1]) + 1772032,
                        reinterpret_cast<char *>(mem_[1]) + 1773056,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc4,
                        0xc400,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[11]) + 1404928,
                        reinterpret_cast<char *>(mem_[11]) + 1605632,
                        reinterpret_cast<char *>(mem_[1]) + 1778688,
                        reinterpret_cast<char *>(mem_[1]) + 1779712,
                        reinterpret_cast<char *>(mem_[1]) + 1776640,
                        reinterpret_cast<char *>(mem_[1]) + 1777664,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc4,
                        0xc400,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[11]) + 1806336,
                        reinterpret_cast<char *>(mem_[15]),
                        reinterpret_cast<char *>(mem_[1]) + 1914368,
                        reinterpret_cast<char *>(mem_[1]) + 1915392,
                        reinterpret_cast<char *>(mem_[1]) + 1912320,
                        reinterpret_cast<char *>(mem_[1]) + 1913344,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc4,
                        0xc400,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[15]),
                        reinterpret_cast<char *>(mem_[15]) + 200704,
                        reinterpret_cast<char *>(mem_[1]) + 1918976,
                        reinterpret_cast<char *>(mem_[1]) + 1920000,
                        reinterpret_cast<char *>(mem_[1]) + 1916928,
                        reinterpret_cast<char *>(mem_[1]) + 1917952,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc4,
                        0xc400,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[15]) + 401408,
                        reinterpret_cast<char *>(mem_[15]) + 602112,
                        reinterpret_cast<char *>(mem_[1]) + 1923584,
                        reinterpret_cast<char *>(mem_[1]) + 1924608,
                        reinterpret_cast<char *>(mem_[1]) + 1921536,
                        reinterpret_cast<char *>(mem_[1]) + 1922560,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0xc4,
                        0xc400,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[0]) + 1981952,
                        reinterpret_cast<char *>(mem_[1]) + 1973760,
                        reinterpret_cast<char *>(mem_[1]) + 1930240,
                        reinterpret_cast<char *>(mem_[1]) + 1932288,
                        reinterpret_cast<char *>(mem_[1]) + 1926144,
                        reinterpret_cast<char *>(mem_[1]) + 1928192,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x31,
                        0x6200,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[4]) + 1929216,
                        reinterpret_cast<char *>(mem_[15]) + 802816,
                        reinterpret_cast<char *>(mem_[1]) + 1938944,
                        reinterpret_cast<char *>(mem_[1]) + 1940992,
                        reinterpret_cast<char *>(mem_[1]) + 1934848,
                        reinterpret_cast<char *>(mem_[1]) + 1936896,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x31,
                        0x6200,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[15]) + 903168,
                        reinterpret_cast<char *>(mem_[15]) + 1003520,
                        reinterpret_cast<char *>(mem_[1]) + 1947648,
                        reinterpret_cast<char *>(mem_[1]) + 1949696,
                        reinterpret_cast<char *>(mem_[1]) + 1943552,
                        reinterpret_cast<char *>(mem_[1]) + 1945600,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x31,
                        0x6200,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[15]) + 1003520,
                        reinterpret_cast<char *>(mem_[15]) + 1103872,
                        reinterpret_cast<char *>(mem_[1]) + 1956352,
                        reinterpret_cast<char *>(mem_[1]) + 1958400,
                        reinterpret_cast<char *>(mem_[1]) + 1952256,
                        reinterpret_cast<char *>(mem_[1]) + 1954304,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x31,
                        0x6200,
                    },
                .paddings =
                    {
                        0,
                    },
            },
            {
                .buf =
                    {
                        reinterpret_cast<char *>(mem_[15]) + 1204224,
                        reinterpret_cast<char *>(mem_[15]) + 1304576,
                        reinterpret_cast<char *>(mem_[1]) + 1965056,
                        reinterpret_cast<char *>(mem_[1]) + 1967104,
                        reinterpret_cast<char *>(mem_[1]) + 1960960,
                        reinterpret_cast<char *>(mem_[1]) + 1963008,
                    },
                .threshold = 1e-5,
                .scalars =
                    {
                        1,
                        0x31,
                        0x6200,
                    },
                .paddings =
                    {
                        0,
                    },
            },
        };
    const static Dimension dimConfig[20] = {
        Dimension(64, 49), Dimension(64, 13), Dimension(64, 13),
        Dimension(64, 13), Dimension(64, 13), Dimension(128, 4),
        Dimension(128, 4), Dimension(128, 4), Dimension(128, 4),
        Dimension(128, 4), Dimension(256, 1), Dimension(256, 1),
        Dimension(256, 1), Dimension(256, 1), Dimension(256, 1),
        Dimension(512, 1), Dimension(512, 1), Dimension(512, 1),
        Dimension(512, 1), Dimension(512, 1)};

    auto idx = exec_batch_norm_idx_++;
    return LaunchKernel(func_[2], dimConfig[idx], Dimension(1, 256), args[idx]);
}

absl::Status ResNet18Inference::Relu() {
    enum {
        kBlockSize = 256,
        kParallelism = 4,
    };
    static const struct {
        int grid;
        MemRegion mem;
    } kInvocations[17] = {
        {.grid = 784, .mem = MemRegion(2, 16187392)},
        {.grid = 196, .mem = MemRegion(7, 802816)},
        {.grid = 196, .mem = MemRegion(8, 802816)},
        {.grid = 196, .mem = MemRegion(9, 802816)},
        {.grid = 196, .mem = MemRegion(10, 802816)},
        {.grid = 98, .mem = MemRegion(9, 1605632)},
        {.grid = 98, .mem = MemRegion(7, 1605632)},
        {.grid = 98, .mem = MemRegion(11, 0)},
        {.grid = 98, .mem = MemRegion(11, 802816)},
        {.grid = 49, .mem = MemRegion(11, 1204224)},
        {.grid = 49, .mem = MemRegion(11, 1605632)},
        {.grid = 49, .mem = MemRegion(15, 200704)},
        {.grid = 49, .mem = MemRegion(15, 602112)},
        {.grid = 25, .mem = MemRegion(1, 1973760)},
        {.grid = 25, .mem = MemRegion(15, 802816)},
        {.grid = 25, .mem = MemRegion(15, 1103872)},
        {.grid = 25, .mem = MemRegion(15, 1304576)},
    };

    auto idx = exec_relu_idx_++;

    auto buf = reinterpret_cast<char *>(mem_[kInvocations[idx].mem.idx]) +
               kInvocations[idx].mem.offset;
    struct {
        int N;
        hipDeviceptr_t dst;
        hipDeviceptr_t src;
    } arg = {
        .N = kInvocations[idx].grid * kParallelism * kBlockSize,
        .dst = buf,
        .src = buf,
    };
    return LaunchKernel(func_[3], kInvocations[idx].grid, kBlockSize, arg);
}

absl::Status ResNet18Inference::MaxPool2D() {
    struct {
        unsigned s0[2];
        hipDeviceptr_t ptr1;
        unsigned s1[6];
        hipDeviceptr_t ptr2;
        hipDeviceptr_t ptr3;
    } args = {
        .s0 = {0x31000, 0},
        .ptr1 = reinterpret_cast<char *>(mem_[2]) + 16187392,
        .s1 = {1, 0x40, 0x70, 0x70, 0x38, 0x38},
        .ptr2 = reinterpret_cast<char *>(mem_[4]) + 524288,
        .ptr3 = reinterpret_cast<char *>(mem_[5]) + 9437184,
    };
    return LaunchKernel(func_[4], 784, 256, args);
}

absl::Status ResNet18Inference::miopenSp3AsmConv_v21_1_3_gfx10_fp32_stride1() {
    struct {
        unsigned s0[8];
        hipDeviceptr_t buf[3];
        unsigned long pad0;
        unsigned s1[8];
        unsigned long pad1[5];
        unsigned s2[16];
        unsigned long pad2[6];
    } args[10] = {
        {
        .s0 = {1, 0x40, 0x38, 0x38, 0x40, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[4]) + 524288,
                reinterpret_cast<char *>(mem_[0]) + 642560,
                reinterpret_cast<char *>(mem_[7]) + 0,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0x38, 0x38, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0xc4000,
                0x3100,
                0xe0,
                4,
                0x900,
                0x24,
                0xc,
                4,
                0xc4000,
                0x3100,
                0xe0,
                4,
                1,
                0xc4000,
                0x24000,
                0xc4000,
            },
        .pad2 =
            {
                0,
            },},{        .s0 = {1, 0x40, 0x38, 0x38, 0x40, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[7]) + 802816,
                reinterpret_cast<char *>(mem_[0]) + 792576,
                reinterpret_cast<char *>(mem_[8]) + 0,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0x38, 0x38, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0xc4000,
                0x3100,
                0xe0,
                4,
                0x900,
                0x24,
                0xc,
                4,
                0xc4000,
                0x3100,
                0xe0,
                4,
                1,
                0xc4000,
                0x24000,
                0xc4000,
            },
        .pad2 = { 0, },},{ .s0 = {1, 0x40, 0x38, 0x38, 0x40, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[8]) + 802816,
                reinterpret_cast<char *>(mem_[0]) + 942592,
                reinterpret_cast<char *>(mem_[9]) + 0,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0x38, 0x38, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0xc4000,
                0x3100,
                0xe0,
                4,
                0x900,
                0x24,
                0xc,
                4,
                0xc4000,
                0x3100,
                0xe0,
                4,
                1,
                0xc4000,
                0x24000,
                0xc4000,
            },
        .pad2 =
            {
                0,
            },},{.s0 = {1, 0x40, 0x38, 0x38, 0x40, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[9]) + 802816,
                reinterpret_cast<char *>(mem_[0]) + 1092608,
                reinterpret_cast<char *>(mem_[10]) + 0,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0x38, 0x38, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0xc4000,
                0x3100,
                0xe0,
                4,
                0x900,
                0x24,
                0xc,
                4,
                0xc4000,
                0x3100,
                0xe0,
                4,
                1,
                0xc4000,
                0x24000,
                0xc4000,
            },
        .pad2 =
            {
                0,
            },},{   .s0 = {1, 0x80, 0x1c, 0x1c, 0x80, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[9]) + 1605632,
                reinterpret_cast<char *>(mem_[1]),
                reinterpret_cast<char *>(mem_[8]) + 1605632,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0x1c, 0x1c, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0x62000,
                0xc40,
                0x70,
                4,
                0x1200,
                0x24,
                0xc,
                4,
                0x62000,
                0xc40,
                0x70,
                4,
                1,
                0x62000,
                0x90000,
                0x62000,
            },
        .pad2 =
            {
                0,
            },},{.s0 = {1, 0x80, 0x1c, 0x1c, 0x80, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[7]) + 1605632,
                reinterpret_cast<char *>(mem_[1]) + 589824,
                reinterpret_cast<char *>(mem_[4]) + 1327104,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0x1c, 0x1c, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0x62000,
                0xc40,
                0x70,
                4,
                0x1200,
                0x24,
                0xc,
                4,
                0x62000,
                0xc40,
                0x70,
                4,
                1,
                0x62000,
                0x90000,
                0x62000,
            },
        .pad2 =
            {
                0,
            },},{.s0 = {1, 0x80, 0x1c, 0x1c, 0x80, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[11]),
                reinterpret_cast<char *>(mem_[1]) + 1179648,
                reinterpret_cast<char *>(mem_[11]) + 401408,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0x1c, 0x1c, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0x62000,
                0xc40,
                0x70,
                4,
                0x1200,
                0x24,
                0xc,
                4,
                0x62000,
                0xc40,
                0x70,
                4,
                1,
                0x62000,
                0x90000,
                0x62000,
            },
        .pad2 =
            {
                0,
            },},{.s0 = {1, 0x100, 0xe, 0xe, 0x100, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[11]) + 1204224,
                reinterpret_cast<char *>(mem_[2]) + 1179648,
                reinterpret_cast<char *>(mem_[11]) + 1404928,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0xe, 0xe, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0x31000,
                0x310,
                0x38,
                4,
                0x2400,
                0x24,
                0xc,
                4,
                0x31000,
                0x310,
                0x38,
                4,
                1,
                0x31000,
                0x240000,
                0x31000,
            },
        .pad2 =
            {
                0,
            },},{.s0 = {1, 0x100, 0xe, 0xe, 0x100, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[11]) + 1605632,
                reinterpret_cast<char *>(mem_[2]) + 3538944,
                reinterpret_cast<char *>(mem_[15]),
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0xe, 0xe, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0x31000,
                0x310,
                0x38,
                4,
                0x2400,
                0x24,
                0xc,
                4,
                0x31000,
                0x310,
                0x38,
                4,
                1,
                0x31000,
                0x240000,
                0x31000,
            },
        .pad2 =
            {
                0,
            },},{.s0 = {1, 0x100, 0xe, 0xe, 0x100, 0x50, 0x600, 0},
        .buf =
            {
                reinterpret_cast<char *>(mem_[15]) + 200704,
                reinterpret_cast<char *>(mem_[2]) + 5898240,
                reinterpret_cast<char *>(mem_[15]) + 401408,
            },
        .pad0 = 0,
        .s1 = {3, 3, 1, 1, 0xe, 0xe, 0, 0},
        .pad1 =
            {
                0,
            },
        .s2 =
            {
                0x31000,
                0x310,
                0x38,
                4,
                0x2400,
                0x24,
                0xc,
                4,
                0x31000,
                0x310,
                0x38,
                4,
                1,
                0x31000,
                0x240000,
                0x31000,
            },
        .pad2 =
            {
                0,
            },},
    };
    auto idx = exec_conv_idx_++;
    return LaunchKernel(func_[5], 80, 256, args[idx]);
}

absl::Status ResNet18Inference::AddAlpha() {
    enum {
        kBlockSize = 256,
        kParallelism = 4,
    };
    static const struct {
        int grid;
        MemRegion l;
        MemRegion r;
    } kInvocations[8] = {
        {196, MemRegion(8, 802816), MemRegion(4, 524288)},
        {196, MemRegion(10, 802816), MemRegion(8, 802816)},
        {98, MemRegion(7, 1605632), MemRegion(4, 1327104)},
        {98, MemRegion(11, 802816), MemRegion(7, 1605632)},
        {49, MemRegion(11, 1605632), MemRegion(15, 0)},
        {49, MemRegion(15, 602112), MemRegion(11, 1605632)},
        {25, MemRegion(15, 802816), MemRegion(15, 1003520)},
        {25, MemRegion(15, 1304576), MemRegion(15, 802816)},
    };
    auto idx = exec_add_alpha_idx_++;
    auto &k = kInvocations[idx];
    struct {
        int N;
        float alpha;
        hipDeviceptr_t buf[3];
    } arg = {
        .N = k.grid * kBlockSize * kParallelism,
        .alpha = 1.0f,
        .buf =
            {
                reinterpret_cast<char *>(mem_[k.l.idx]) + k.l.offset,
                reinterpret_cast<char *>(mem_[k.l.idx]) + k.l.offset,
                reinterpret_cast<char *>(mem_[k.r.idx]) + k.r.offset,
            },
    };
    return LaunchKernel(func_[6], k.grid, kBlockSize, arg);
}

absl::Status ResNet18Inference::TransposeEntry() {
    struct {
        hipDeviceptr_t buf[2];
        unsigned s0[10];
        unsigned long pad[7];
    } args[6] = {
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[10]) + 802816,
                    reinterpret_cast<char *>(mem_[4]) + 1327104,
                },
            .s0 = {0, 0, 0x38, 0x1c, 1, 0x40, 2, 2, 0xc40, 0x310},
            .pad =
                {
                    0,
                },
        },
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[4]) + 1327104,
                    reinterpret_cast<char *>(mem_[0]) + 1580544,
                },
            .s0 = {0xc400, 0, 4, 0xc4, 1, 0x80, 0x1c, 0x1c},
            .pad =
                {
                    0,
                },
        },
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[11]) + 802816,
                    reinterpret_cast<char *>(mem_[15]),
                },
            .s0 = {0, 0, 0x1c, 0xe, 1, 0x80, 2, 2, 0x310, 0xc4},
            .pad =
                {
                    0,
                },
        },
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[15]),
                    reinterpret_cast<char *>(mem_[11]) + 1806336,
                },
            .s0 = {0x6200, 0, 4, 0x31, 1, 0x100, 0xe, 0xe},
            .pad =
                {
                    0,
                },
        },
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[15]) + 602112,
                    reinterpret_cast<char *>(mem_[15]) + 1003520,
                },
            .s0 = {0, 0, 0xe, 0x7, 1, 0x100, 2, 2, 0xc4, 0x31},
            .pad =
                {
                    0,
                },
        },
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[15]) + 1003520,
                    reinterpret_cast<char *>(mem_[15]) + 903168,
                },
            .s0 = {0x3100, 0, 1, 0x31, 1, 0x200, 7, 7},
            .pad =
                {
                    0,
                },
        },
    };
    const static int transposeBinaryIndex[] = {8, 10, 8, 12, 8, 17};
    const static Dimension globalWorkSize[] = {Dimension(16, 1, 64), 512,
                                               Dimension(4, 1, 128), 256,
                                               Dimension(1, 1, 256), 512};
    auto idx = exec_transpose_idx_++;
    return LaunchKernel(func_[transposeBinaryIndex[idx]], globalWorkSize[idx],
                        49, args[idx]);
}

absl::Status ResNet18Inference::Cijk_S() {
    struct {
        hipDeviceptr_t buf[2];
        unsigned s0[10];
        unsigned long pad[7];
    } args[3] = {
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[4]) + 1929216,
                    reinterpret_cast<char *>(mem_[4]) + 1929216,
                },
            .s0 = {0x31, 0, 0x31, 0, 0x31, 0x200, 1, 0, 0, 0},
            .pad =
                {
                    0,
                },
        },
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[15]) + 1003520,
                    reinterpret_cast<char *>(mem_[15]) + 1003520,
                },
            .s0 = {0x31, 0, 0x31, 0, 0x31, 0x200, 1, 0, 0, 0},
            .pad =
                {
                    0,
                },
        },
        {
            .buf =
                {
                    reinterpret_cast<char *>(mem_[15]) + 1204224,
                    reinterpret_cast<char *>(mem_[15]) + 1204224,
                },
            .s0 = {0x31, 0, 0x31, 0, 0x31, 0x200, 1, 0, 0, 0},
            .pad =
                {
                    0,
                },
        },
    };
    auto idx = exec_cijk_s_idx_++;
    return LaunchKernel(func_[14], 98, 256, args[idx]);
}

absl::Status ResNet18Inference::AdaptiveAvgPool2d() {
    struct {
        hipDeviceptr_t buf[2];
    } args = {
        .buf =
            {
                reinterpret_cast<char *>(mem_[15]) + 1304576,
                reinterpret_cast<char *>(mem_[0]) + 2082304,
            },
    };

    return LaunchKernel(func_[24], 1, 512, args);
}

std::unique_ptr<ResNetInference> NewResNet18() {
    return std::unique_ptr<ResNetInference>(new ResNet18Inference());
}

}; // namespace gpumpc::experiment
