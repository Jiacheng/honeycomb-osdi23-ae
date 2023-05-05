#include "slab_allocator.h"

namespace ocl::hsa {

void SlabAllocatorBase::Initialize(absl::Span<char> region,
                                   size_t element_size) {
    size_t n = region.size() / element_size;
    char *base = &region.front();
    free_list_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        free_list_[i] = base + i * element_size;
    }
}

char *SlabAllocatorBase::Allocate() {
    if (free_list_.empty()) {
        return nullptr;
    }
    auto r = free_list_.back();
    free_list_.pop_back();
    return r;
}

void SlabAllocatorBase::Free(char *value) { free_list_.push_back(value); }

} // namespace ocl::hsa