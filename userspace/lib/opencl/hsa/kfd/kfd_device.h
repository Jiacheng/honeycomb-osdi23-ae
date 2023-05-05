#pragma once

#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/platform.h"

namespace ocl::hsa {

namespace enclave {
class EnclaveHostAgentPlatform;
}

class Platform;
class KFDPlatform;
class KFDEvent;
class DeviceQueue;

class KFDDevice : public Device {
  public:
    friend class KFDPlatform;
    friend class enclave::EnclaveHostAgentPlatform;
    virtual unsigned GetNodeID() const override { return node_id_; }
    virtual unsigned GetGPUID() const override { return gpu_id_; }
    virtual unsigned GetDoorbellPageSize() const override {
        return doorbell_page_size_;
    }
    virtual void *
    GetOrInitializeDoorbell(uint64_t doorbell_mmap_offset) override;
    virtual const Properties &GetProperties() const override;
    virtual MemoryManager *GetMemoryManager() override { return mm_.get(); }
    virtual std::unique_ptr<DeviceQueue> CreateSDMAQueue() override;
    virtual std::unique_ptr<DeviceQueue> CreateAQLQueue() override;

    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;
    static const Properties &GetHardCodedProperties();

  protected:
    explicit KFDDevice(unsigned node_id, unsigned gpu_id);
    void SetMemoryManager(std::unique_ptr<MemoryManager> &&mm);
    absl::Status ParseProperties();
    absl::Status OpenRenderFD();
    absl::Status AcquireVM();

    int drm_render_fd_;
    unsigned drm_render_minor_;

    unsigned node_id_;
    unsigned gpu_id_;

    const unsigned doorbell_page_size_;

    std::unique_ptr<MemoryManager> mm_;

    // Doorbells are process-wide per-device GTT memory.
    // It is lazily initialized when creating the first queue.
    std::unique_ptr<Memory> doorbell_;
};

} // namespace ocl::hsa