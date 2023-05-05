#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <absl/status/status.h>

namespace ocl::hsa {

class RingAllocator {
  public:
    explicit RingAllocator();
    size_t Initialize(uintptr_t base, size_t length, size_t align);
    bool AvailableAlign(size_t size, size_t align);
    absl::Status AllocateAlign(uintptr_t *target, size_t size, size_t align);
    absl::Status Free();

  private:
    struct TryAllocateResult {
        bool available;
        size_t offset;
        size_t allocated_length;
    };

    TryAllocateResult TryAllocateAlign(size_t size, size_t align);

    uintptr_t base_;
    size_t length_;
    size_t length_mask_;
    size_t align_;

    std::vector<size_t> entries_;

    // head points to the first allocated byte
    // tail points to right after the last allocated byte
    // note that when head == tail, no byte is allocated or all bytes are
    // allocated always allocate aligned length so head/tail length are aligned
    size_t head_length_;
    size_t tail_length_;
};

} // namespace ocl::hsa
