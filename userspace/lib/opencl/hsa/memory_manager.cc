#include "memory_manager.h"
#include "assert.h"
#include "kfd/kfd_memory.h"
#include "opencl/hsa/utils.h"
#include "platform.h"
#include "runtime_options.h"

#include <absl/status/status.h>
#include <hsa/kfd_ioctl.h>
#include <sys/mman.h>

#include "hsa/amdgpu_vm.h"

namespace ocl::hsa {

MemoryManager::MemoryManager(Device *dev) : dev_(dev), strict_layout_(false) {
    auto opt = GetRuntimeOptions();
    if (opt) {
        strict_layout_ = opt->IsStrictLayout();
    }
}

void *MemoryManager::Mmap(void *addr, size_t size, size_t alignment,
                          int flags) {
    const auto map_size = alignment + std::max<size_t>(size, alignment);
    unsigned mmap_flag = 0;
    unsigned prot = 0;
    if (flags & kMapIntoHostAddressSpace) {
        prot = PROT_READ | PROT_WRITE;
        mmap_flag = MAP_ANONYMOUS | MAP_PRIVATE;
    } else {
        prot = PROT_NONE;
        mmap_flag = MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE;
    }
    if (flags & kFixed) {
        mmap_flag |= MAP_FIXED;
    }

    void *ret = mmap(addr, map_size, prot, mmap_flag, -1, 0);
    if (ret == MAP_FAILED) {
        return nullptr;
    }

    char *ret_start = reinterpret_cast<char *>(ret);
    char *real_start = reinterpret_cast<char *>(
        (reinterpret_cast<uintptr_t>(ret_start + alignment - 1) &
         ~(alignment - 1)));
    if (ret_start != real_start) {
        munmap(ret_start, real_start - ret_start);
    }
    char *real_end = real_start + size;
    char *buf_end = ret_start + map_size;
    if (real_end != buf_end) {
        munmap(real_end, buf_end - real_end);
    }

    madvise(real_start, size, MADV_DONTFORK);

    if (flags & kClearHost) {
        memset(real_start, 0, size);
    }
    return real_start;
}

} // namespace ocl::hsa
