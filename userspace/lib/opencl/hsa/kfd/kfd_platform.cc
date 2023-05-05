#include "kfd_platform.h"
#include "kfd_device.h"
#include "kfd_memory_manager.h"
#include "opencl/hsa/kfd_event.h"

#include <hsa/hsa.h>
#include <hsa/hsakmttypes.h>
#include <hsa/kfd_ioctl.h>

#include <memory>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

#include <fcntl.h>

namespace ocl::hsa {

KFDPlatform::KFDPlatform() : initialized_(false), kfd_fd_(0) {}

Platform &KFDPlatform::Instance() {
    static KFDPlatform inst;
    return inst;
}

absl::Status KFDPlatform::Initialize() {
    kfd_fd_ = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (kfd_fd_ < 0) {
        return absl::InvalidArgumentError("Cannot open /dev/fd");
    }

    auto stat = EnumerateDevices();
    if (!stat.ok()) {
        return stat;
    }
    stat = resource_.Initialize(devices_.front());
    if (!stat.ok()) {
        return stat;
    }
    initialized_ = true;
    return absl::OkStatus();
}

absl::Status KFDPlatform::EnumerateDevices() {
    namespace fs = std::filesystem;
    static const std::regex kKFDNodesRegex(
        "^/sys/devices/virtual/kfd/kfd/topology/nodes/([0-9]+)$");
    static const fs::path kKFDNodesBase(
        "/sys/devices/virtual/kfd/kfd/topology/nodes/");

    for (const auto &p : fs::directory_iterator(kKFDNodesBase)) {
        std::smatch m;
        auto path_string = p.path().string();
        if (!std::regex_match(path_string, m, kKFDNodesRegex)) {
            continue;
        }
        uint32_t node_id = std::stoi(m[1]);

        auto path = kKFDNodesBase / std::to_string(node_id) / "gpu_id";
        std::ifstream ifs(path.string());
        auto gpu_id_str = std::string(std::istreambuf_iterator<char>(ifs),
                                      std::istreambuf_iterator<char>());
        auto gpu_id = std::stoi(gpu_id_str);
        if (!gpu_id) {
            continue;
        }

        auto d = std::unique_ptr<Device>(NewDevice(node_id, gpu_id));
        auto stat = d->Initialize();
        if (!stat.ok()) {
            spdlog::warn("Failed to initialize device with node_id {}",
                         node_id);
        }

        devices_.push_back(d.release());
    }

    return absl::OkStatus();
}

Device *KFDPlatform::NewDevice(unsigned node_id, unsigned gpu_id) {
    auto dev = new KFDDevice(node_id, gpu_id);
    dev->SetMemoryManager(
        std::unique_ptr<MemoryManager>(new KFDMemoryManager(dev)));
    return dev;
}

absl::Status KFDPlatform::Close() {
    resource_.Destroy();
    for (auto d : devices_) {
        auto stat = d->Close();
        if (!stat.ok()) {
            spdlog::warn("Failed to close device {}", stat.ToString());
        }
    }
    if (kfd_fd_) {
        close(kfd_fd_);
        kfd_fd_ = 0;
    }
    return absl::OkStatus();
}

void PlatformResource::Destroy() {
    auto _ = vm_fault_event_->Destroy();
    vm_fault_event_.reset();
    _ = vm_fault_signal_event_->Destroy();
    vm_fault_signal_event_.reset();
    // The signal BO will be automatically freed by the kernel
    // Surpress the warning message here
    _ = event_page_->Destroy();
}

absl::Status PlatformResource::Initialize(Device *dev) {
    auto stat = InitializeEventPage(dev);
    if (!stat.ok()) {
        return stat;
    }
    vm_fault_event_ = Platform::Instance().NewEvent(HSA_EVENTTYPE_MEMORY,
                                                    event_page_->GetHandle());
    vm_fault_signal_event_ = Platform::Instance().NewSignalEvent();
    return stat;
}

absl::Status PlatformResource::InitializeEventPage(Device *dev) {
    event_page_ = dev->GetMemoryManager()->NewEventPage();
    if (!event_page_) {
        return absl::ResourceExhaustedError("Cannot allocate event page");
    }
    return absl::OkStatus();
}

gpu_addr_t KFDPlatform::GetEventPageBase() {
    return resource_.event_page_->GetGPUAddress();
}

std::unique_ptr<Event> KFDPlatform::NewEvent(int type,
                                             uint64_t event_page_handle) {
    return KFDEvent::New(type, event_page_handle);
}

} // namespace ocl::hsa