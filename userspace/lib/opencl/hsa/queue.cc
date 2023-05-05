#include "queue.h"
#include "memory_manager.h"
#include "platform.h"
#include "utils.h"

#include "utils/align.h"

#include <absl/status/status.h>
#include <hsa/kfd_ioctl.h>

#include <memory>

#include <sys/ioctl.h>
#include <sys/mman.h>

namespace ocl::hsa {

DeviceQueue::DeviceQueue(Device *dev)
    : dev_(dev), queue_id_(kInvalidQueueID), doorbell_base_(nullptr),
      doorbell_offset_(0) {}

absl::Status
DeviceQueue::RegisterQueue(struct kfd_ioctl_create_queue_args *args) {
    int kfd_fd = Platform::Instance().GetKFDFD();
    int err = kmtIoctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, args);
    if (err) {
        return absl::InvalidArgumentError("Cannot create queue");
    }

    queue_id_ = args->queue_id;
    return InitializeDoorbell(args->doorbell_offset);
}

absl::Status DeviceQueue::UnregisterQueue() {
    if (queue_id_ == kInvalidQueueID) {
        return absl::OkStatus();
    }
    struct kfd_ioctl_destroy_queue_args args = {0};
    args.queue_id = queue_id_;

    int kfd_fd = Platform::Instance().GetKFDFD();
    int err = kmtIoctl(kfd_fd, AMDKFD_IOC_DESTROY_QUEUE, &args);
    if (err) {
        return absl::InvalidArgumentError("Cannot destroy queue");
    }
    queue_id_ = kInvalidQueueID;
    return absl::OkStatus();
}

absl::Status DeviceQueue::CreateRingBuffer(size_t size) {
    ring_ = dev_->GetMemoryManager()->NewRingBuffer(size);
    if (!ring_) {
        return absl::InvalidArgumentError("Cannot create ring buffer");
    }
    return absl::OkStatus();
}

absl::Status DeviceQueue::AllocateSharedInfo() {
    static const size_t kAlignedQueueSize =
        gpumpc::AlignUp(sizeof(SharedQueueWatermark), Device::kPageSize);

    dispatch_region_ =
        dev_->GetMemoryManager()->NewGTTMemory(kAlignedQueueSize, true);
    memset(dispatch_region_->GetBuffer(), 0, dispatch_region_->GetSize());
    if (!dispatch_region_) {
        return absl::InvalidArgumentError("Cannot allocate shared info");
    }
    return absl::OkStatus();
}

std::atomic<uint64_t> *DeviceQueue::GetReadDispatchPtr() const {
    auto q =
        reinterpret_cast<SharedQueueWatermark *>(dispatch_region_->GetBuffer());
    auto ptr = reinterpret_cast<std::atomic<uint64_t> *>(&q->rptr);
    return ptr;
}

std::atomic<uint64_t> *DeviceQueue::GetWriteDispatchPtr() const {
    auto q =
        reinterpret_cast<SharedQueueWatermark *>(dispatch_region_->GetBuffer());
    auto ptr = reinterpret_cast<std::atomic<uint64_t> *>(&q->wptr);
    return ptr;
}

void DeviceQueue::UpdateDoorbell(uint64_t new_index,
                                 std::memory_order memory_order) {
    auto doorbell = doorbell_base_ + doorbell_offset_ / sizeof(uint64_t);
    doorbell->store(new_index, memory_order);
}

absl::Status DeviceQueue::Destroy() {
    if (!ring_) {
        return absl::OkStatus();
    }
    auto stat = UnregisterQueue();
    if (!stat.ok()) {
        return stat;
    }

    stat = dispatch_region_->Destroy();
    if (!stat.ok()) {
        return stat;
    }

    stat = ring_->Destroy();
    if (!stat.ok()) {
        return stat;
    }
    ring_.reset();
    return absl::OkStatus();
}

absl::Status DeviceQueue::InitializeDoorbell(uint64_t doorbell_offset) {
    const uint64_t doorbell_page_size = dev_->GetDoorbellPageSize();
    /* On SOC15 chips, the doorbell offset within the
     * doorbell page is included in the doorbell offset
     * returned by KFD. This allows CP queue doorbells to be
     * allocated dynamically (while SDMA queue doorbells fixed)
     * rather than based on the its process queue ID.
     */
    uint64_t doorbell_mmap_offset = doorbell_offset & ~(doorbell_page_size - 1);
    doorbell_offset_ = doorbell_offset & (dev_->GetDoorbellPageSize() - 1);

    doorbell_base_ = reinterpret_cast<std::atomic<uint64_t> *>(
        dev_->GetOrInitializeDoorbell(doorbell_mmap_offset));
    if (!doorbell_base_) {
        return absl::InvalidArgumentError("No doorbell");
    }
    return absl::OkStatus();
}

} // namespace ocl::hsa
