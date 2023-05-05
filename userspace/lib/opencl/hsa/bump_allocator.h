#pragma once

#include <cstddef>
#include <cstdint>

namespace ocl::hsa {

class BumpAllocator {
  public:
    explicit BumpAllocator();
    void Initialize(uintptr_t base, size_t length);
    uintptr_t Allocate(size_t size);
    uintptr_t AllocateAlign(size_t size, size_t align);
    size_t GetOffset() const { return offset_; }

  private:
    uintptr_t base_;
    size_t offset_;
    size_t length_;
};

} // namespace ocl::hsa