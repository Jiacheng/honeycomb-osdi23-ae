#include "g6/g6_queue.h"
#include "queue.h"
#include "utils.h"

#include "utils/align.h"

#include <atomic>
#include <hsa/hsa.h>
#include <hsa/hsakmttypes.h>
#include <hsa/kfd_ioctl.h>

namespace ocl::hsa {

enum {
    kAQLPacketSize = 64,
    kQueueSizeInBytes = AQLQueue::kQueuePackets * kAQLPacketSize,
};

enum {
    kAqlHeaderInvalid = (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE),
};

AQLQueue::AQLQueue(Device *dev) : DeviceQueue(dev) {}

std::unique_ptr<AQLQueue> AQLQueue::Create(Device *device, absl::Status *stat) {
    std::unique_ptr<AQLQueue> q(
        static_cast<AQLQueue *>(device->CreateAQLQueue().release()));

    *stat = q->Initialize();
    if (!stat->ok()) {
        return nullptr;
    }

    *stat = absl::OkStatus();
    return q;
}

uint64_t AQLQueue::AcquireRingBuffer(int packet) {
    auto ptr = GetWriteDispatchPtr();
    return ptr->fetch_add(packet, std::memory_order_release);
}

uint64_t AQLQueue::AcquireReadDispatchIndex() const {
    auto ptr = GetReadDispatchPtr();
    return ptr->load(std::memory_order_acquire);
}

void AQLQueue::ReleaseRingBuffer(int packet_idx) {
    UpdateDoorbell(packet_idx, std::memory_order_release);
}

absl::Status AQLQueue::Initialize() {
    auto stat = CreateRingBuffer(kQueueSizeInBytes);
    if (!stat.ok()) {
        return stat;
    }

    for (size_t i = 0; i < kQueuePackets; ++i) {
        auto pkt_header =
            reinterpret_cast<char *>(ring_->GetBuffer()) + kAQLPacketSize * i;
        *reinterpret_cast<uint16_t *>(pkt_header) = kAqlHeaderInvalid;
    }

    stat = AllocateSharedInfo();
    if (!stat.ok()) {
        return stat;
    }
    stat = CreateQueue();
    if (!stat.ok()) {
        return stat;
    }

    return absl::OkStatus();
}

absl::Status AQLQueue::Destroy() {
    auto stat = DeviceQueue::Destroy();
    if (!stat.ok()) {
        return stat;
    }
    return ctx_save_->Destroy();
}

absl::Status
AQLQueue::AllocateCtxBuffer(struct kfd_ioctl_create_queue_args *args) {
    const auto &p = dev_->GetProperties();
    auto cu_num = p.num_fcompute_cores / p.num_simd_per_cu;
    auto wave_num = p.num_waves;

    // RX6900 has a limit of 0x7000 of hardware stack size
    auto ctl_stack_size = std::min<size_t>(
        0x7000, gpumpc::AlignUp(wave_num * p.control_stack_bytes_per_wave + 8 +
                                    sizeof(HsaUserContextSaveAreaHeader),
                                Device::kPageSize));
    auto wg_data_size = cu_num * p.wg_context_data_size_per_cu;
    auto debug_memory_size = gpumpc::AlignUp(
        wave_num * p.debugger_bytes_per_wave, p.debugger_bytes_align);
    auto ctx_save_restore_size =
        ctl_stack_size +
        gpumpc::AlignUp(wg_data_size + debug_memory_size, Device::kPageSize);

    auto ctx_buffer_size = gpumpc::AlignUp(
        ctx_save_restore_size + debug_memory_size + p.eop_buffer_size,
        Device::kHugeGPUPageSize);
    ctx_save_ = dev_->GetMemoryManager()->NewGTTMemory(ctx_buffer_size, false);
    if (!ctx_save_) {
        return absl::ResourceExhaustedError("Cannot allocate context buffer");
    }
    args->ctx_save_restore_address = (uintptr_t)ctx_save_->GetBuffer();
    args->ctx_save_restore_size = ctx_save_restore_size;
    args->ctl_stack_size = ctl_stack_size;

    auto header = (HsaUserContextSaveAreaHeader *)ctx_save_->GetBuffer();
    header->ErrorEventId = 0;
    header->ErrorReason = 0;
    header->DebugOffset = ctx_save_restore_size;
    header->DebugSize = debug_memory_size;

    args->eop_buffer_size = p.eop_buffer_size;
    args->eop_buffer_address = (uintptr_t)ctx_save_->GetBuffer() +
                               ctx_save_->GetSize() - args->eop_buffer_size;

    return absl::OkStatus();
}

absl::Status AQLQueue::CreateQueue() {
    auto q = reinterpret_cast<DeviceQueue::SharedQueueWatermark *>(
        dispatch_region_->GetBuffer());
    struct kfd_ioctl_create_queue_args args = {0};

    args.gpu_id = dev_->GetGPUID();
    args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    args.read_pointer_address = (uintptr_t)&q->rptr;
    args.write_pointer_address = (uintptr_t)&q->wptr;
    args.ring_base_address = (uintptr_t)ring_->GetBuffer();
    args.ring_size = ring_->GetSize();
    args.queue_percentage = 100;
    args.queue_priority = 7;

    auto stat = AllocateCtxBuffer(&args);
    if (!stat.ok()) {
        return stat;
    }

    stat = RegisterQueue(&args);
    return stat;
}

uint64_t AQLQueue::DispatchAQLPacket(const void *pkt, uint16_t header,
                                     uint16_t rest) {
    enum { kQueueMask = kQueuePackets - 1 };

    // Check for queue full and wait if needed.
    uint64_t index = AcquireRingBuffer(1);

    // Make sure the slot is free for usage
    while ((index - AcquireReadDispatchIndex()) >= kQueueMask) {
        sched_yield();
    }

    auto dst = reinterpret_cast<char *>(ring_->GetBuffer()) +
               (index & kQueueMask) * kAQLPacketSize;
    memcpy(dst, pkt, kAQLPacketSize);
    if (header != 0) {
        auto h = reinterpret_cast<std::atomic<uint32_t> *>(dst);
        h->store((uint32_t)header | (rest << 16), std::memory_order_release);
    }

    ReleaseRingBuffer(index);

    return index;
}

} // namespace ocl::hsa
