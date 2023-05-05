#include "guest_memory.h"
#include "guest_rpc_client.h"
#include <absl/status/status.h>

#include <spdlog/spdlog.h>

namespace ocl::hsa::enclave {

EnclaveGuestMemory::EnclaveGuestMemory(EnclaveGuestDevice *parent, void *buf,
                                       size_t size)
    : dev_(parent), buf_(buf), size_(size), scope_(ResourceScope::kNone) {}

absl::Status EnclaveGuestMemory::AllocGPUMemory(uint32_t gpu_id, uint32_t flag,
                                                uint64_t mmap_offset) {
    auto rpc = dev_->GetRPCClient();
    rpc->AllocateGPUMemory(flag, reinterpret_cast<uintptr_t>(buf_), size_,
                           mmap_offset, &handle_);
    return absl::OkStatus();
}

absl::Status EnclaveGuestMemory::MapGPUMemory() {
    auto rpc = dev_->GetRPCClient();
    rpc->MapGPUMemory(handle_);
    return absl::OkStatus();
}

absl::Status EnclaveGuestMemory::Destroy() {
    if (!buf_) {
        return absl::OkStatus();
    }

    auto stat = UnmapFromGPU();
    if (!(scope_ & kUnmanagedBO)) {
        stat = DeallocateMemoryFromGPU();
    }
    // Don't need to unmap any memory as the whole address spaces are now
    // pre-mapped
    handle_ = 0;
    buf_ = nullptr;
    return stat;
}

EnclaveGuestMemory::~EnclaveGuestMemory() {
    auto stat = Destroy();
    if (!stat.ok()) {
        spdlog::warn("Failed to destroy the memory");
    }
}

absl::Status EnclaveGuestMemory::UnmapFromGPU() {
    auto rpc = dev_->GetRPCClient();
    rpc->UnmapGPUMemory(handle_);
    return absl::OkStatus();
}

absl::Status EnclaveGuestMemory::DeallocateMemoryFromGPU() {
    auto rpc = dev_->GetRPCClient();
    rpc->DeallocateGPUMemory(handle_);
    return absl::OkStatus();
}

} // namespace ocl::hsa::enclave