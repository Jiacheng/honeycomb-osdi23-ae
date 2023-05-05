#include "bump_allocator.h"
#include "utils/align.h"

namespace ocl::hsa {
BumpAllocator::BumpAllocator() : base_(0), offset_(0), length_(0) {}

void BumpAllocator::Initialize(uintptr_t base, size_t length) {
    base_ = base;
    offset_ = 0;
    length_ = length;
}

uintptr_t BumpAllocator::Allocate(size_t size) {
    if (offset_ + size > length_) {
        return 0;
    }
    auto r = offset_ + base_;
    offset_ += size;
    return r;
}

uintptr_t BumpAllocator::AllocateAlign(size_t size, size_t alignment) {
    auto offset_aligned = gpumpc::AlignUp(offset_, alignment);
    if (offset_aligned + size > length_) {
        return 0;
    }
    auto r = offset_aligned + base_;
    offset_ = offset_aligned + size;
    return r;
}

} // namespace ocl::hsa
