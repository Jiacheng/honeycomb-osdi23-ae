#pragma once

#include <cstddef>
#include <cstdint>

namespace ocl::hip {

class AESDeviceSrc {
  public:
    explicit AESDeviceSrc(uintptr_t ptr, const size_t len);

    bool IsHead() const { return head_len_ != 0; }
    uintptr_t GetHead() const { return head_; }
    size_t GetHeadLen() const { return head_len_; }

    // in case total len < 16
    bool IsAligned() const { return aligned_len_ != 0; }
    uintptr_t GetAligned() const { return aligned_head_; }
    size_t GetAlignedLen() const { return aligned_len_; }

    bool IsTail() const { return tail_len_ != 0; }
    uintptr_t GetTail() const { return aligned_tail_; }
    size_t GetTailLen() const { return tail_len_; }

    size_t GetTotalLen() const { return head_len_ + aligned_len_ + tail_len_; }

  private:
    uintptr_t head_;
    size_t head_len_;

    uintptr_t aligned_head_;
    size_t aligned_len_;

    uintptr_t aligned_tail_;
    size_t tail_len_;
};

class AESBuffer {
  public:
    explicit AESBuffer(uintptr_t ptr, const AESDeviceSrc &src);

    bool IsHead() const { return head_len_ != 0; }
    uintptr_t GetHead() const { return head_; }
    size_t GetHeadLen() const { return head_len_; }

    bool IsAligned() const { return aligned_len_ != 0; }
    uintptr_t GetAligned() const { return aligned_head_; }
    size_t GetAlignedLen() const { return aligned_len_; }

    bool IsTail() const { return tail_len_ != 0; }
    uintptr_t GetTail() const { return aligned_tail_; }
    size_t GetTailLen() const { return tail_len_; }

    size_t GetTotalLen() const { return head_len_ + aligned_len_ + tail_len_; }

  private:
    uintptr_t head_;
    size_t head_len_;

    uintptr_t aligned_head_;
    size_t aligned_len_;

    uintptr_t aligned_tail_;
    size_t tail_len_;
};

} // namespace ocl::hip
