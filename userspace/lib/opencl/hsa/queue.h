#pragma once

#include "memory_manager.h"
#include "platform.h"
#include <atomic>
#include <memory>

extern "C" {
struct kfd_ioctl_create_queue_args;
}

namespace ocl::hsa {

class Device;
class DeviceQueue {
  public:
    static constexpr int kInvalidQueueID = -1;

    virtual ~DeviceQueue() = default;
    void *GetBuffer() const { return ring_->GetBuffer(); }
    int GetQueueID() const { return queue_id_; }
    virtual absl::Status Destroy();

    std::atomic<uint64_t> *GetReadDispatchPtr() const;
    std::atomic<uint64_t> *GetWriteDispatchPtr() const;
    virtual absl::Status Register() { return absl::OkStatus(); }
    virtual void
    UpdateDoorbell(uint64_t new_index,
                   std::memory_order memory_order = std::memory_order_seq_cst);

    struct SharedQueueWatermark {
        uint64_t wptr;
        uint64_t rptr;
    };

  protected:
    DeviceQueue(Device *dev);

    //
    // Register the user-level queue to the kernel
    virtual absl::Status
    RegisterQueue(struct kfd_ioctl_create_queue_args *args);
    virtual absl::Status UnregisterQueue();
    absl::Status CreateRingBuffer(size_t size);
    // Allocate read / write dispatch indices that are shared between CPU / GPU
    absl::Status AllocateSharedInfo();
    absl::Status InitializeDoorbell(uint64_t doorbell_offset);

    Device *const dev_;
    std::unique_ptr<Memory> ring_;
    std::unique_ptr<Memory> dispatch_region_;
    int queue_id_;
    std::atomic<uint64_t> *doorbell_base_;
    unsigned doorbell_offset_;
};

class SDMAQueue : public DeviceQueue {
  public:
    enum : size_t {
        kQueueAlignment = Memory::kGPUHugePageSize,
        kQueueSize = 1 << 20,
    };
    static std::unique_ptr<SDMAQueue> Create(Device *device,
                                             absl::Status *stat);
    size_t GetBufferSize() const { return kQueueSize; }
    virtual absl::Status Register() override;

    SDMAQueue(const SDMAQueue &) = delete;
    SDMAQueue &operator=(const SDMAQueue &) = delete;

  protected:
    friend class KFDDevice;
    explicit SDMAQueue(Device *dev);
    absl::Status Initialize();
};

class AQLQueue : public DeviceQueue {
  public:
    enum : size_t {
        kQueueAlignment = Memory::kGPUHugePageSize,
        kQueuePackets = 4096,
    };

    static std::unique_ptr<AQLQueue> Create(Device *device, absl::Status *stat);
    size_t GetBufferSize() const;
    size_t GetSize() const { return kQueuePackets; }
    virtual absl::Status Destroy() override;
    //
    // Dispatch a 64-byte AQL packet to the hardware queue
    // returns the aql packet index
    uint64_t DispatchAQLPacket(const void *pkt, uint16_t header, uint16_t rest);

    AQLQueue(const AQLQueue &) = delete;
    AQLQueue &operator=(const AQLQueue &) = delete;

  protected:
    friend class KFDDevice;
    explicit AQLQueue(Device *dev);
    uint64_t AcquireRingBuffer(int packet);
    uint64_t AcquireReadDispatchIndex() const;
    void ReleaseRingBuffer(int packet_idx);

    absl::Status Initialize();
    virtual absl::Status CreateQueue(); // FIXME: use Register()
    absl::Status AllocateCtxBuffer(struct kfd_ioctl_create_queue_args *args);

    std::unique_ptr<Memory> ctx_save_;
};

} // namespace ocl::hsa
