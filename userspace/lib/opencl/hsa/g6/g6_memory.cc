#include "g6_memory.h"
#include "opencl/hsa/types.h"

namespace ocl::hsa {
G6Memory::G6Memory(gpu_addr_t gpu_addr, size_t size)
    : gpu_addr_(gpu_addr), size_(size) {}

} // namespace ocl::hsa