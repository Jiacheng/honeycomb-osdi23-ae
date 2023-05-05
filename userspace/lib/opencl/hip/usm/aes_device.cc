#include "aes_device.h"
#include "aes_buffer.h"
#include "secure_memcpy.h"

#include "opencl/hip/device_context.h"
#include "opencl/hsa/runtime_options.h"

#include "utils/align.h"
#include "utils/hip_helper.h"

#include <absl/status/status.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <hip/hip_runtime.h>

namespace ocl::hip {

using namespace gpumpc::experiment;
using namespace gpumpc;
using namespace std;

static const char *kModule = "/data/aes_kernel.bin";

static const int kBlock = 256;

const char *AESDevice::kFunctions[AESDevice::kNumFunctions] = {
    "AES256_CTR_init",
    "AES256_CTR_compute",
    "AES256_CTR_compute_unaligned",
};

enum {
    // 2K
    // the first 1K for keyschedule
    // the next 1K for iv
    kMetadataDeviceSize = 2 << 10,
    kKeyScheduleSize = 1 << 10,
};

absl::Status AESDevice::Initialize(DeviceContext *parent) {
    parent_ = parent;

    auto stat = LoadBinary();
    if (!stat.ok()) {
        return stat;
    }

    auto hipError =
        parent_->GetMemoryManager()->hipMalloc(&mem_, kMetadataDeviceSize);
    if (hipError != hipSuccess) {
        return absl::InvalidArgumentError("can not allocate mem");
    }

    ks_ = mem_;
    iv_ = mem_ + kKeyScheduleSize;
    return absl::OkStatus();
}

absl::Status AESDevice::LoadModule(hipModule_t *module,
                                   std::string module_path) {
    auto opt = GetRuntimeOptions();
    if (!opt) {
        return absl::InvalidArgumentError("can not get runtime options");
    }
    auto resource_dir = opt->GetResourceDir();

    auto filepath = resource_dir + module_path;

    std::ifstream elf_file(filepath, ios::binary);
    if (!elf_file.is_open()) {
        return absl::InvalidArgumentError("can not open file");
    }

    std::string str((std::istreambuf_iterator<char>(elf_file)),
                    std::istreambuf_iterator<char>());
    elf_file.close();

    return HipErrToStatus(
        parent_->GetComputeContext()->hipModuleLoadData(module, str.c_str()));
}

absl::Status AESDevice::LoadBinary() {
    auto stat = LoadModule(&module_, kModule);
    if (!stat.ok()) {
        return absl::InvalidArgumentError("Cannot load kernel");
    }

    for (size_t i = 0; i < kNumFunctions; i++) {
        auto err = parent_->GetComputeContext()->hipModuleGetFunction(
            &func_[i], module_, kFunctions[i]);
        if (err != hipSuccess) {
            return absl::InvalidArgumentError("Cannot load function");
        }
    }
    return absl::OkStatus();
}

absl::Status AESDevice::Key(const uint32_t *ukey) {
    // in-place expand
    auto hipError = parent_->GetMemoryManager()->hipMemcpyHtoD(
        ks_, const_cast<uint32_t *>(ukey), kAES256KeySize);
    if (hipError != hipSuccess) {
        return absl::InvalidArgumentError("Cannot memcpy");
    }

    struct {
        ocl::hsa::gpu_addr_t ks;
        ocl::hsa::gpu_addr_t ukey;
    } args = {
        .ks = ks_,
        .ukey = ks_,
    };
    auto err = parent_->GetComputeContext()->hipModuleLaunchKernel(
        func_[0], 1, 1, 1, 1, 1, 1, &args, sizeof(args));
    if (err != hipSuccess) {
        return absl::InvalidArgumentError(
            "Failed to initialize aes device key");
    }
    return absl::OkStatus();
}

absl::Status AESDevice::IV(const uint32_t *iv) {
    // reset iv_start_
    iv_start_ = 0;

    auto hipError = parent_->GetMemoryManager()->hipMemcpyHtoD(
        iv_, const_cast<uint32_t *>(iv), kAESIVSize);
    if (hipError != hipSuccess) {
        return absl::InvalidArgumentError("Cannot memcpy");
    }
    return absl::OkStatus();
}

absl::Status AESDevice::EncryptSome(ocl::hsa::gpu_addr_t out,
                                    ocl::hsa::gpu_addr_t in,
                                    const unsigned long len, bool aligned) {
    struct {
        ocl::hsa::gpu_addr_t out;
        ocl::hsa::gpu_addr_t in;
        ocl::hsa::gpu_addr_t iv;
        ocl::hsa::gpu_addr_t ks;
        unsigned long size_bytes;
        unsigned long iv_start;
    } args = {
        .out = out,
        .in = in,
        .iv = iv_,
        .ks = ks_,
        .size_bytes = len,
        .iv_start = iv_start_,
    };
    int blocks = gpumpc::AlignUp(len, kAESBlockSize) / kAESBlockSize;
    auto grid = gpumpc::AlignUp(blocks, kBlock) / kBlock;

    // update iv_start_
    iv_start_ += blocks;

    auto err = parent_->GetComputeContext()->hipModuleLaunchKernel(
        aligned ? func_[1] : func_[2], grid * kBlock, 1, 1, kBlock, 1, 1, &args,
        sizeof(args));
    if (err != hipSuccess) {
        return absl::InvalidArgumentError("Failed to launch kernel");
    }
    return absl::OkStatus();
}

absl::Status AESDevice::Encrypt(ocl::hsa::gpu_addr_t dst,
                                const AESDeviceSrc &device_src) {
    AESBuffer device_dst(reinterpret_cast<uintptr_t>(dst), device_src);

    if (device_src.IsHead()) {
        auto stat = EncryptSome(device_dst.GetHead(), device_src.GetHead(),
                                device_src.GetHeadLen(), false);
        if (!stat.ok()) {
            return stat;
        }
    }
    if (device_src.IsAligned()) {
        auto stat =
            EncryptSome(device_dst.GetAligned(), device_src.GetAligned(),
                        device_src.GetAlignedLen(), true);
        if (!stat.ok()) {
            return stat;
        }
    }
    if (device_src.IsTail()) {
        auto stat = EncryptSome(device_dst.GetTail(), device_src.GetTail(),
                                device_src.GetTailLen(), false);
        if (!stat.ok()) {
            return stat;
        }
    }
    return absl::OkStatus();
}

void AESDevice::Close() {
    checkHipErrors(parent_->GetComputeContext()->ReleaseDeviceMemoryFence());

    auto _ = parent_->GetComputeContext()->hipModuleUnload(module_);

    _ = parent_->GetMemoryManager()->hipFree(mem_);

    (void)_;
}
} // namespace ocl::hip
