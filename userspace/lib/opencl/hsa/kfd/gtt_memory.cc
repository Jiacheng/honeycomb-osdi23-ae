#include "kfd_memory.h"
#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/platform.h"
#include "opencl/hsa/utils.h"

#include <hsa/kfd_ioctl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>

namespace ocl::hsa {
KFDMemory::KFDMemory(void *buf, size_t size)
    : buf_(buf), size_(size), gpu_id_(0), handle_(0),
      scope_(ResourceScope::kNone) {}

absl::Status KFDMemory::AllocGPUMemory(uint32_t gpu_id, uint32_t flag,
                                       uint64_t mmap_offset) {
    if (handle_) {
        return absl::InvalidArgumentError("Already allocated");
    }

    struct kfd_ioctl_alloc_memory_of_gpu_args args = {0};
    args.gpu_id = gpu_id;
    args.size = size_;
    args.flags = flag;
    args.va_addr = (uint64_t)buf_;
    args.mmap_offset = mmap_offset;

    int kfd_fd = Platform::Instance().GetKFDFD();
    if (kmtIoctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args)) {
        return absl::InvalidArgumentError("Cannot alloc memory on GPU");
    }
    handle_ = args.handle;
    gpu_id_ = gpu_id;
    return absl::OkStatus();
}

absl::Status KFDMemory::MapGPUMemory() {
    if (!gpu_id_) {
        return absl::InvalidArgumentError("Memory has not been allocated");
    }

    struct kfd_ioctl_map_memory_to_gpu_args args = {0};

    args.handle = handle_;
    args.device_ids_array_ptr = (uint64_t)&gpu_id_;
    args.n_devices = 1;
    args.n_success = 0;

    int kfd_fd = Platform::Instance().GetKFDFD();
    int ret = kmtIoctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &args);
    return ret ? absl::InvalidArgumentError("Cannot map memory to GPU")
               : absl::OkStatus();
}

absl::Status KFDMemory::UnmapFromGPU() {
    struct kfd_ioctl_unmap_memory_from_gpu_args args = {0};
    args.handle = handle_;
    args.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&gpu_id_);
    args.n_devices = 1;
    args.n_success = 0;
    int kfd_fd = Platform::Instance().GetKFDFD();
    int ret = kmtIoctl(kfd_fd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &args);
    return ret ? absl::InvalidArgumentError("Cannot unmap memory from GPU")
               : absl::OkStatus();
}

absl::Status KFDMemory::DeallocateMemoryFromGPU() {
    struct kfd_ioctl_free_memory_of_gpu_args args = {0};
    int kfd_fd = Platform::Instance().GetKFDFD();
    args.handle = handle_;
    int ret = kmtIoctl(kfd_fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &args);
    return ret ? absl::InvalidArgumentError("Cannot deallocate memory from GPU")
               : absl::OkStatus();
}

absl::Status KFDMemory::Destroy() {
    if (!buf_) {
        return absl::OkStatus();
    }

    auto stat = UnmapFromGPU();
    if (!(scope_ & kUnmanagedBO)) {
        stat = DeallocateMemoryFromGPU();
    }
    handle_ = 0;
    if (!(scope_ & kUnmanagedMmap)) {
        munmap(buf_, size_);
    }
    buf_ = nullptr;
    return stat;
}

KFDMemory::~KFDMemory() {
    auto stat = Destroy();
    if (!stat.ok()) {
        spdlog::warn("Failed to destroy the memory");
    }
}

} // namespace ocl::hsa
