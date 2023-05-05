#include "ring_allocator.h"
#include "utils/align.h"

namespace ocl::hsa {
RingAllocator::RingAllocator()
    : base_(0), length_(0), length_mask_(0x0), align_(0), head_length_(0),
      tail_length_(0) {}

size_t RingAllocator::Initialize(uintptr_t base, size_t length, size_t align) {
    align_ = align;
    base_ = gpumpc::AlignUp(base, align);
    size_t unaligned_head = base_ - base;
    if (length < unaligned_head) {
        length_ = 0;
    } else {
        length_ = gpumpc::AlignDown(length - unaligned_head, align);
    }
    while (((length_mask_ << 1) | 0x1) < length_) {
        length_mask_ <<= 1;
        length_mask_ |= 1;
    }
    // make length_ in the form of 2^k
    length_ = length_mask_ + 1;
    return length_;
}

bool RingAllocator::AvailableAlign(size_t size, size_t align) {
    return TryAllocateAlign(size, align).available;
}

RingAllocator::TryAllocateResult RingAllocator::TryAllocateAlign(size_t size,
                                                                 size_t align) {
    // allocated size should be aligned
    size = gpumpc::AlignUp(size, align_);
    size_t head_rounded = head_length_ & length_mask_;
    size_t tail_rounded = tail_length_ & length_mask_;
    // allocated address should be aligned
    size_t tail_aligned = gpumpc::AlignUp(base_ + tail_rounded, align) - base_;
    size_t zero_aligned = gpumpc::AlignUp(base_ + 0, align) - base_;

    bool full = tail_length_ - head_length_ == length_;
    bool empty = tail_length_ == head_length_;

    TryAllocateResult tail_alloc = {true, tail_aligned,
                                    size + (tail_aligned - tail_rounded)};
    TryAllocateResult zero_alloc = {true, zero_aligned,
                                    size + (length_ - tail_rounded) +
                                        (zero_aligned - 0)};
    TryAllocateResult whole_alloc = {true, zero_aligned, length_};

    if (head_rounded <= tail_rounded && !full) {
        if (length_ - tail_aligned >= size) {
            // allocate after tail with align
            return tail_alloc;
        } else if (head_rounded - zero_aligned >= size) {
            // allocate at the aligned begining of the buffer
            return zero_alloc;
        } else if (empty) {
            // special case: when it is empty and the size is too large to
            // fit in the above two range, we can just allocate the whole space
            if (length_ - zero_aligned >= size) {
                return whole_alloc;
            }
        }
    } else if (head_rounded > tail_aligned) {
        if (head_rounded - tail_aligned >= size) {
            // allocate after tail with align
            return tail_alloc;
        }
    }

    return {false, 0, 0};
}

absl::Status RingAllocator::Free() {
    if (!entries_.empty()) {
        auto len = *entries_.begin();
        head_length_ += len;
        entries_.erase(entries_.begin());
        return absl::OkStatus();
    } else {
        return absl::InvalidArgumentError("Free more than allocate");
    }
}

absl::Status RingAllocator::AllocateAlign(uintptr_t *target, size_t size,
                                          size_t align) {
    auto res = TryAllocateAlign(size, align);
    auto available = res.available;
    auto offset = res.offset;
    auto allocated_length = res.allocated_length;
    if (!available) {
        return absl::InvalidArgumentError(
            "Called allocate without checking if available");
    } else {
        tail_length_ += allocated_length;
        entries_.push_back(allocated_length);
        if (target) {
            *target = base_ + offset;
        }
        return absl::OkStatus();
    }
}

} // namespace ocl::hsa
