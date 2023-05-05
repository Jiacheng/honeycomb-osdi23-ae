#include "kfd_device.h"
#include "kfd_memory_manager.h"
#include "opencl/hsa/platform.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/utils.h"

#include <hsa/kfd_ioctl.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>

namespace ocl::hsa {

namespace fs = std::filesystem;

enum { kHardcodedDoorbellPageSize = 8192 };

KFDDevice::KFDDevice(unsigned node_id, unsigned gpu_id)
    : drm_render_fd_(0), drm_render_minor_(0), node_id_(node_id),
      gpu_id_(gpu_id), doorbell_page_size_(kHardcodedDoorbellPageSize) {}

void KFDDevice::SetMemoryManager(std::unique_ptr<MemoryManager> &&mm) {
    mm_.swap(mm);
}

std::unique_ptr<DeviceQueue> KFDDevice::CreateSDMAQueue() {
    return std::unique_ptr<DeviceQueue>(new SDMAQueue(this));
}

std::unique_ptr<DeviceQueue> KFDDevice::CreateAQLQueue() {
    return std::unique_ptr<DeviceQueue>(new AQLQueue(this));
}

void *KFDDevice::GetOrInitializeDoorbell(uint64_t doorbell_mmap_offset) {
    if (doorbell_) {
        return doorbell_->GetBuffer();
    }
    auto mm = static_cast<KFDMemoryManager *>(mm_.get());
    doorbell_ = mm->NewDoorbell(doorbell_page_size_, doorbell_mmap_offset);
    return doorbell_->GetBuffer();
}

const KFDDevice::Properties &KFDDevice::GetHardCodedProperties() {
    // For RX6900 only
    enum {
        kVGPRSizePerCU = 0x40000,
        kSGPRSizePerCU = 0x4000,
        kLDSSizePerCU = 0x10000,
        kHWRegSizePerCU = 0x1000,
        kEOPBufferSize = 0x1000,
        kDebuggerBytesAlignment = 64,
        kDebuggerBytesPerWave = 32,
    };

    static const Properties kProperties = {
        .num_fcompute_cores = 160,
        .num_simd_per_cu = 2,
        .num_waves = 160 / 2 * 32,
        .control_stack_bytes_per_wave = 12,
        .wg_context_data_size_per_cu =
            kVGPRSizePerCU + kSGPRSizePerCU + kLDSSizePerCU + kHWRegSizePerCU,
        .eop_buffer_size = kEOPBufferSize,
        .debugger_bytes_per_wave = kDebuggerBytesPerWave,
        .debugger_bytes_align = kDebuggerBytesAlignment,
    };
    return kProperties;
}

const KFDDevice::Properties &KFDDevice::GetProperties() const {
    return KFDDevice::GetHardCodedProperties();
}

absl::Status KFDDevice::Initialize() {
    auto stat = ParseProperties();
    if (!stat.ok()) {
        return stat;
    }
    stat = OpenRenderFD();
    if (!stat.ok()) {
        return stat;
    }
    return AcquireVM();
}

absl::Status KFDDevice::OpenRenderFD() {
    enum {
        kDRMFirstRenderNode = 128,
        kDRMLastRenderNode = 255,
    };

    if (drm_render_minor_ < kDRMFirstRenderNode ||
        drm_render_minor_ > kDRMLastRenderNode) {
        return absl::InvalidArgumentError("DRM render minor out of range");
    }

    std::stringstream ss;
    ss << "/dev/dri/renderD" << drm_render_minor_;
    std::string s = ss.str();
    auto fd = open(s.c_str(), O_RDWR | O_CLOEXEC);

    if (fd < 0) {
        return absl::InvalidArgumentError("Failed to open the DRM fd");
    }
    drm_render_fd_ = fd;
    return absl::OkStatus();
}

absl::Status KFDDevice::Close() {
    if (drm_render_fd_) {
        close(drm_render_fd_);
        drm_render_fd_ = 0;
    }
    return absl::OkStatus();
}

absl::Status KFDDevice::AcquireVM() {
    int kfd_fd = Platform::Instance().GetKFDFD();
    struct kfd_ioctl_acquire_vm_args args;
    args.gpu_id = gpu_id_;
    args.drm_fd = drm_render_fd_;
    if (kmtIoctl(kfd_fd, AMDKFD_IOC_ACQUIRE_VM, &args)) {
        return absl::InvalidArgumentError("AMDKFD_IOC_ACQUIRE_VM failed");
    }
    return absl::OkStatus();
}

absl::Status KFDDevice::ParseProperties() {
    static const std::regex kPropRegex("([a-z_]+) ([0-9]+)$");
    const fs::path path =
        fs::path("/sys/devices/virtual/kfd/kfd/topology/nodes/") /
        std::to_string(node_id_) / "properties";
    std::ifstream ifs(path.string());
    std::string line;
    while (std::getline(ifs, line)) {
        std::smatch m;
        if (!std::regex_match(line, m, kPropRegex)) {
            continue;
        }
        if (m[1] == "drm_render_minor") {
            drm_render_minor_ = std::stoi(m[2]);
        }
    }
    return absl::OkStatus();
}

} // namespace ocl::hsa