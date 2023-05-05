#include "g6_platform.h"
#include "g6_device.h"
#include "g6_memory_manager.h"
#include "opencl/hsa/kfd_event.h"

#include <hsa/hsakmttypes.h>

#include <spdlog/spdlog.h>

#include <fcntl.h>

namespace ocl::hsa {

G6Platform::G6Platform() : signal_va_(0) {}

Platform &G6Platform::Instance() {
    static G6Platform inst;
    return inst;
}

absl::Status G6Platform::Initialize() {
    kfd_fd_ = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (kfd_fd_ < 0) {
        return absl::InvalidArgumentError("Cannot open /dev/fd");
    }

    auto stat = EnumerateDevices();
    if (!stat.ok()) {
        return stat;
    }

    stat = InitializeSignalBO(devices_.front());
    if (!stat.ok()) {
        return stat;
    }

    initialized_ = true;
    return absl::OkStatus();
}

absl::Status G6Platform::Close() {
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

gpu_addr_t G6Platform::GetEventPageBase() { return signal_va_; }

absl::Status G6Platform::InitializeSignalBO(Device *dev) {
    enum {
        kSignalBOSize = KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t),
        kSignalBOPage = kSignalBOSize / Device::kPageSize,
    };
    auto gdev = static_cast<G6Device *>(dev);
    auto stat = gdev->InitializeSignalBO(&signal_bo_);
    if (!stat.ok()) {
        return stat;
    }
    signal_va_ = signal_bo_->GetGPUAddress();
    return absl::OkStatus();
}

Device *G6Platform::NewDevice(unsigned node_id, unsigned gpu_id) {
    return new G6Device(node_id, gpu_id);
}

std::unique_ptr<Event> G6Platform::NewEvent(int type,
                                            uint64_t event_page_handle) {
    return KFDEvent::New(type, event_page_handle);
}

} // namespace ocl::hsa