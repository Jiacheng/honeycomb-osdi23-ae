#pragma once

#include <absl/types/span.h>

#include <atomic>

namespace ocl::hsa {
class SDMAQueue;

//
// Write SDMA commands into the rignt buffer
//
class SDMAOpsBuilder {
  public:
    explicit SDMAOpsBuilder(const absl::Span<char> &buffer);
    static const uint32_t kFenceCommandSize;
    static const uint32_t kAtomicCommandSize;
    static const uint32_t kTimestampCommandSize;
    static const uint32_t kTrapCommandSize;
    static const uint32_t kLinearCopyCommandSize;

    SDMAOpsBuilder &Fence(uint32_t *fence, uint32_t fence_value);
    SDMAOpsBuilder &AtomicDecrement(void *addr);
    SDMAOpsBuilder &GetGlobalTimestamp(void *write_address);
    SDMAOpsBuilder &Trap(uint32_t event_id);
    SDMAOpsBuilder &Copy(void *dst, const void *src, size_t size);

  private:
    static const int SizeToCountOffset = -1;

    size_t AcquireBuffer(size_t size);
    template <class T> T *Current() {
        return reinterpret_cast<T *>(buffer_.begin() + offset_);
    }
    absl::Span<char> buffer_;
    size_t offset_;
};

//
// Higher level APIs to interact with the SDMA queue
class SDMAOpsQueue {
  public:
    using RingIndexTy = uint64_t;
    static const bool HwIndexMonotonic = true;
    static const int SizeToCountOffset = -1;
    static const bool useGCR = true;

    explicit SDMAOpsQueue(SDMAQueue *queue);

    /// @brief Acquires the address into queue buffer where a new command
    /// packet of specified size could be written. The address that is
    /// returned is guaranteed to be unique even in a multi-threaded access
    /// scenario. This function is guaranteed to return a pointer for writing
    /// data into the queue buffer.
    ///
    /// @param cmd_size Command packet size in bytes.
    ///
    /// @param curr_index (output) Index to pass to ReleaseWriteAddress.
    ///
    /// @return pointer into the queue buffer where a PM4 packet of specified
    /// size could be written. NULL if input size is greater than the size of
    /// queue buffer.
    char *AcquireWriteAddress(uint32_t cmd_size, RingIndexTy &curr_index);

    /// @brief Updates the Write Register of compute device to the end of
    /// SDMA packet written into queue buffer. The update to Write Register
    /// will be safe under multi-threaded usage scenario. Furthermore, updates
    /// to Write Register are blocking until all prior updates are completed
    /// i.e. if two threads T1 & T2 were to call release, then updates by T2
    /// will block until T1 has completed its update (assumes T1 acquired the
    /// write address first).
    ///
    /// @param curr_index Index passed back from AcquireWriteAddress.
    ///
    /// @param cmd_size Command packet size in bytes.
    void ReleaseWriteAddress(RingIndexTy curr_index, uint32_t cmd_size);

  private:
    void UpdateWriteAndDoorbellRegister(RingIndexTy curr_index,
                                        RingIndexTy new_index);

    /// @brief Writes NO-OP words into queue buffer in case writing a command
    /// causes the queue buffer to wrap.
    ///
    /// @param curr_index Index to begin padding from.
    void PadRingToEnd(RingIndexTy curr_index);
    uint32_t WrapIntoRing(RingIndexTy index);
    bool CanWriteUpto(RingIndexTy upto_index);

    SDMAQueue *queue_;

    // Monotonic ring indices, in bytes, tracking written and submitted
    // commands.
    std::atomic<RingIndexTy> cached_reserve_index_;
    std::atomic<RingIndexTy> cached_commit_index_;
};

} // namespace ocl::hsa