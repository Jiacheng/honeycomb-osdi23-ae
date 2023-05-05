#include "aes_buffer.h"
#include "secure_memcpy.h"
#include "utils/align.h"

namespace ocl::hip {

AESDeviceSrc::AESDeviceSrc(uintptr_t ptr, const size_t len) {
    head_ = ptr;
    auto tail = head_ + len;
    aligned_head_ = gpumpc::AlignUp(head_, kAESBlockSize);
    aligned_tail_ = gpumpc::AlignDown(tail, kAESBlockSize);
    head_len_ = aligned_head_ - head_;
    tail_len_ = tail - aligned_tail_;
    aligned_len_ = aligned_tail_ - aligned_head_;
}

AESBuffer::AESBuffer(uintptr_t ptr, const AESDeviceSrc &src) {
    head_ = ptr;
    // may not be aligned
    aligned_head_ = head_ + src.GetHeadLen();
    aligned_tail_ = aligned_head_ + src.GetAlignedLen();
    head_len_ = src.GetHeadLen();
    tail_len_ = src.GetTailLen();
    aligned_len_ = src.GetAlignedLen();
}

} // namespace ocl::hip
