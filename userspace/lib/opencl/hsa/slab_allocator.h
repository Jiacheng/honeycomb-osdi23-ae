#pragma once

#include <absl/types/span.h>

namespace ocl::hsa {

class SlabAllocatorBase {
  protected:
    void Initialize(absl::Span<char> region, size_t element_size);
    char *Allocate();
    void Free(char *value);
    std::vector<char *> free_list_;
};

template <class T> class SlabAllocator : private SlabAllocatorBase {
  public:
    void Initialize(absl::Span<char> region) {
        SlabAllocatorBase::Initialize(region, sizeof(T));
    }
    T *Allocate() {
        return reinterpret_cast<T *>(SlabAllocatorBase::Allocate());
    }
    void Free(T *value) {
        SlabAllocatorBase::Free(reinterpret_cast<char *>(value));
    }
};

} // namespace ocl::hsa