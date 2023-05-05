#include "amdgpu_program.h"
#include "utils/align.h"

#include <cstring>
#include <numeric>

namespace ocl::hip {

KernelArgument::KernelArgument(const std::string &name, unsigned type,
                               size_t size, size_t offset)
    : name_(name), type_(type), size_(size), offset_(offset) {}

size_t AMDGPUProgram::GetVMAEnd() const {
    auto s = std::accumulate(load_segments_.begin(), load_segments_.end(), 0,
                             [&](size_t e, const Segment &s) {
                                 return std::max(e, s.vma_start + s.vma_length);
                             });
    return gpumpc::AlignUp(s, 4096);
}

} // namespace ocl::hip
