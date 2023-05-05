#include "g6/g6_queue.h"
#include "queue.h"
#include "utils.h"

#include <hsa/kfd_ioctl.h>

namespace ocl::hsa {

SDMAQueue::SDMAQueue(Device *dev) : DeviceQueue(dev) {}

std::unique_ptr<SDMAQueue> SDMAQueue::Create(Device *device,
                                             absl::Status *stat) {
    static_assert((size_t)kQueueSize < (size_t)Memory::kGPUHugePageSize, "");
    std::unique_ptr<SDMAQueue> q(
        static_cast<SDMAQueue *>(device->CreateSDMAQueue().release()));

    *stat = q->Initialize();
    if (!stat->ok()) {
        return nullptr;
    }

    *stat = absl::OkStatus();
    return q;
}

absl::Status SDMAQueue::Initialize() {
    auto stat = CreateRingBuffer(kQueueSize);
    if (!stat.ok()) {
        return stat;
    }
    stat = AllocateSharedInfo();
    if (!stat.ok()) {
        return stat;
    }

    if (!stat.ok()) {
        return stat;
    }
    return absl::OkStatus();
}

absl::Status SDMAQueue::Register() {
    struct kfd_ioctl_create_queue_args args = {0};

    auto q = reinterpret_cast<DeviceQueue::SharedQueueWatermark *>(
        dispatch_region_->GetBuffer());
    args.gpu_id = dev_->GetGPUID();
    args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA;
    args.read_pointer_address = (uintptr_t)&q->rptr;
    args.write_pointer_address = (uintptr_t)&q->wptr;
    args.ring_base_address = (uintptr_t)ring_->GetBuffer();
    args.ring_size = ring_->GetSize();
    args.queue_percentage = 100;
    args.queue_priority = 7;

    auto stat = RegisterQueue(&args);
    return stat;
}

} // namespace ocl::hsa
