#include "guest_memory.h"
#include "guest_rpc_client.h"
#include "opencl/hsa/enclave/guest_platform.h"
#include "opencl/hsa/enclave/idl.h"
#include "opencl/hsa/platform.h"
#include <absl/status/status.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>

namespace ocl::hsa::enclave {

EnclaveGuestMemory::EnclaveGuestMemory(EnclaveGuestDevice *parent, void *buf,
                                       size_t size)
    : dev_(parent), buf_(buf), size_(size), scope_(ResourceScope::kNone) {}

absl::Status EnclaveGuestMemory::AllocGPUMemory(bool map_remote_pages,
                                                uint32_t gpu_id, uint32_t flag,
                                                uint64_t mmap_offset) {
    auto rpc = dev_->GetRPCClient();
    idl::AllocateGPUMemoryResponse resp;
    rpc->AllocateGPUMemory(map_remote_pages, flag,
                           reinterpret_cast<uintptr_t>(buf_), size_,
                           mmap_offset, &resp);
    if (map_remote_pages) {
        // Map into the virtual address space
        auto &plat = static_cast<EnclaveGuestPlatform &>(Platform::Instance());
        int fd = plat.GetAgentPhysicalMemoryFD();
        for (size_t i = 0; i < resp.pfns.size(); i++) {
            auto va_addr = reinterpret_cast<uintptr_t>(buf_) + i * kPageSize;
            auto ret = mmap(reinterpret_cast<void *>(va_addr), kPageSize,
                            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
                            resp.pfns[i] * kPageSize);
            if (ret == MAP_FAILED) {
                return absl::InternalError("Failed to map GTT memory");
            }
        }
    }
    handle_ = resp.handle;
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