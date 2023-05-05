#include "sdma_ops.h"
#include "assert.h"
#include "queue.h"
#include "sdma_registers.h"

#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ocl::hsa {

using namespace rocr::AMD;
static const size_t kMaxSingleCopySize = 0x3fffe0;

static inline uint32_t ptrlow32(const void *p) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

static inline uint32_t ptrhigh32(const void *p) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) >> 32);
}

const uint32_t SDMAOpsBuilder::kFenceCommandSize = sizeof(SDMA_PKT_FENCE);
const uint32_t SDMAOpsBuilder::kAtomicCommandSize = sizeof(SDMA_PKT_ATOMIC);
const uint32_t SDMAOpsBuilder::kTimestampCommandSize =
    sizeof(SDMA_PKT_TIMESTAMP);
const uint32_t SDMAOpsBuilder::kLinearCopyCommandSize =
    sizeof(SDMA_PKT_COPY_LINEAR);
const uint32_t SDMAOpsBuilder::kTrapCommandSize = sizeof(SDMA_PKT_TRAP);

SDMAOpsBuilder::SDMAOpsBuilder(const absl::Span<char> &buffer)
    : buffer_(buffer), offset_(0) {}

SDMAOpsBuilder &SDMAOpsBuilder::Fence(uint32_t *fence, uint32_t fence_value) {
    auto packet_addr = Current<SDMA_PKT_FENCE>();
    offset_ = AcquireBuffer(kFenceCommandSize);

    memset(packet_addr, 0, sizeof(SDMA_PKT_FENCE));

    packet_addr->HEADER_UNION.op = SDMA_OP_FENCE;
    // ISA Major >= 10
    packet_addr->HEADER_UNION.mtype = 3;

    packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(fence);
    packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(fence);
    packet_addr->DATA_UNION.data = fence_value;
    return *this;
}

SDMAOpsBuilder &SDMAOpsBuilder::AtomicDecrement(void *addr) {
    auto packet_addr = Current<SDMA_PKT_ATOMIC>();
    offset_ = AcquireBuffer(kAtomicCommandSize);

    memset(packet_addr, 0, sizeof(SDMA_PKT_ATOMIC));
    packet_addr->HEADER_UNION.op = SDMA_OP_ATOMIC;
    packet_addr->HEADER_UNION.operation = SDMA_ATOMIC_ADD64;
    packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(addr);
    packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(addr);
    packet_addr->SRC_DATA_LO_UNION.src_data_31_0 = 0xffffffff;
    packet_addr->SRC_DATA_HI_UNION.src_data_63_32 = 0xffffffff;
    return *this;
}

SDMAOpsBuilder &SDMAOpsBuilder::GetGlobalTimestamp(void *write_address) {
    HSA_ASSERT((uintptr_t)write_address % 32 == 0);

    auto packet_addr = Current<SDMA_PKT_TIMESTAMP>();
    offset_ = AcquireBuffer(kTimestampCommandSize);

    memset(packet_addr, 0, sizeof(SDMA_PKT_TIMESTAMP));
    packet_addr->HEADER_UNION.op = SDMA_OP_TIMESTAMP;
    packet_addr->HEADER_UNION.sub_op = SDMA_SUBOP_TIMESTAMP_GET_GLOBAL;
    packet_addr->ADDR_LO_UNION.addr_31_0 = ptrlow32(write_address);
    packet_addr->ADDR_HI_UNION.addr_63_32 = ptrhigh32(write_address);
    return *this;
}

SDMAOpsBuilder &SDMAOpsBuilder::Trap(uint32_t event_id) {
    auto packet_addr = Current<SDMA_PKT_TRAP>();
    offset_ = AcquireBuffer(kTrapCommandSize);

    memset(packet_addr, 0, sizeof(SDMA_PKT_TRAP));
    packet_addr->HEADER_UNION.op = SDMA_OP_TRAP;
    packet_addr->INT_CONTEXT_UNION.int_ctx = event_id;
    return *this;
}

SDMAOpsBuilder &SDMAOpsBuilder::Copy(void *dst, const void *src, size_t size) {
    HSA_ASSERT(size <= kMaxSingleCopySize);

    auto packet_addr = Current<SDMA_PKT_COPY_LINEAR>();
    offset_ = AcquireBuffer(kLinearCopyCommandSize);
    memset(packet_addr, 0, sizeof(SDMA_PKT_COPY_LINEAR));

    packet_addr->HEADER_UNION.op = SDMA_OP_COPY;
    packet_addr->HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
    packet_addr->COUNT_UNION.count = size + SizeToCountOffset;

    packet_addr->SRC_ADDR_LO_UNION.src_addr_31_0 = ptrlow32(src);
    packet_addr->SRC_ADDR_HI_UNION.src_addr_63_32 = ptrhigh32(src);

    packet_addr->DST_ADDR_LO_UNION.dst_addr_31_0 = ptrlow32(dst);
    packet_addr->DST_ADDR_HI_UNION.dst_addr_63_32 = ptrhigh32(dst);

    return *this;
}

size_t SDMAOpsBuilder::AcquireBuffer(size_t size) {
    if (offset_ + size > buffer_.size()) {
        throw std::out_of_range("Insufficient size of buffer");
    }
    return offset_ + size;
}

SDMAOpsQueue::SDMAOpsQueue(SDMAQueue *queue)
    : queue_(queue), cached_reserve_index_(0), cached_commit_index_(0) {}

char *SDMAOpsQueue::AcquireWriteAddress(uint32_t cmd_size,
                                        RingIndexTy &curr_index) {
    // Ring is full when all but one byte is written.
    if (cmd_size >= queue_->GetBufferSize()) {
        return nullptr;
    }

    while (true) {
        curr_index = cached_reserve_index_.load(std::memory_order_acquire);

        // Check whether a linear region of the requested size is available.
        // If == cmd_size: region is at beginning of ring.
        // If < cmd_size: region intersects end of ring, pad with no-ops and
        // retry.
        if (WrapIntoRing(curr_index + cmd_size) < cmd_size) {
            PadRingToEnd(curr_index);
            continue;
        }

        // Check whether the engine has finished using this region.
        const RingIndexTy new_index = curr_index + cmd_size;

        if (CanWriteUpto(new_index) == false) {
            // Wait for read index to move and try again.
            std::this_thread::yield();
            continue;
        }

        // Try to reserve this part of the ring.
        if (cached_reserve_index_.compare_exchange_strong(curr_index,
                                                          new_index)) {
            return reinterpret_cast<char *>(queue_->GetBuffer()) +
                   WrapIntoRing(curr_index);
        }

        // Another thread reserved curr_index, try again.
        std::this_thread::yield();
    }

    return nullptr;
}

void SDMAOpsQueue::UpdateWriteAndDoorbellRegister(RingIndexTy curr_index,
                                                  RingIndexTy new_index) {
    while (true) {
        // Make sure that the address before ::curr_index is already released.
        // Otherwise the CP may read invalid packets.
        if (cached_commit_index_.load(std::memory_order_acquire) ==
            curr_index) {
#if 0
            if (core::Runtime::runtime_singleton_->flag().sdma_wait_idle()) {
                // TODO: remove when sdma wpointer issue is resolved.
                // Wait until the SDMA engine finish processing all packets
                // before updating the wptr and doorbell.
                while (WrapIntoRing(*reinterpret_cast<RingIndexTy *>(
                           queue_resource->Queue_read_ptr)) !=
                       WrapIntoRing(curr_index)) {
                    std::this_thread::yield();
                }
            }
#endif

            // Update write pointer and doorbel register.
            queue_->GetWriteDispatchPtr()->store(new_index,
                                                 std::memory_order_release);

            // Ensure write pointer is visible to GPU before doorbell.
            std::atomic_thread_fence(std::memory_order_release);

            queue_->UpdateDoorbell(new_index);

            cached_commit_index_.store(new_index, std::memory_order_release);
            break;
        }

        // Waiting for another thread to submit preceding commands first.
        std::this_thread::yield();
    }
}

void SDMAOpsQueue::ReleaseWriteAddress(RingIndexTy curr_index,
                                       uint32_t cmd_size) {
    if (cmd_size > queue_->GetBufferSize()) {
        assert(false && "cmd_addr is outside the queue buffer range");
        return;
    }

    UpdateWriteAndDoorbellRegister(curr_index, curr_index + cmd_size);
}

void SDMAOpsQueue::PadRingToEnd(RingIndexTy curr_index) {
    // Reserve region from here to the end of the ring.
    RingIndexTy new_index =
        curr_index + (queue_->GetBufferSize() - WrapIntoRing(curr_index));

    // Check whether the engine has finished using this region.
    if (CanWriteUpto(new_index) == false) {
        // Wait for read index to move and try again.
        return;
    }

    if (cached_reserve_index_.compare_exchange_strong(curr_index, new_index)) {
        // Write and submit NOP commands in reserved region.
        char *nop_address = reinterpret_cast<char *>(queue_->GetBuffer()) +
                            WrapIntoRing(curr_index);
        memset(nop_address, 0, new_index - curr_index);

        UpdateWriteAndDoorbellRegister(curr_index, new_index);
    }
}

uint32_t SDMAOpsQueue::WrapIntoRing(RingIndexTy index) {
    return index & (queue_->GetBufferSize() - 1);
}

bool SDMAOpsQueue::CanWriteUpto(RingIndexTy upto_index) {
    // Get/calculate the monotonic read index.
    RingIndexTy hw_read_index =
        queue_->GetReadDispatchPtr()->load(std::memory_order_acquire);
    RingIndexTy read_index = hw_read_index;

    // Check whether the read pointer has passed the given index.
    // At most we can submit (kQueueSize - 1) bytes at a time.
    return (upto_index - read_index) < queue_->GetBufferSize();
}

} // namespace ocl::hsa