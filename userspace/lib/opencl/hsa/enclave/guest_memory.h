#pragma once

#include "opencl/hsa/enclave/guest_platform.h"
#include "opencl/hsa/memory_manager.h"

namespace ocl::hsa::enclave {

/*
 * The guest memory has identical addresss space w.r.t the host agent.
 * All ioctls are forwarded to the agents.
 */
class EnclaveGuestMemory : public Memory {
  public:
    friend class MemoryManager;
    virtual void *GetBuffer() const override { return buf_; }
    virtual size_t GetSize() const override { return size_; }
    virtual uint64_t GetHandle() const override { return handle_; }
    virtual gpu_addr_t GetGPUAddress() const override {
        return reinterpret_cast<gpu_addr_t>(buf_);
    }

    explicit EnclaveGuestMemory(EnclaveGuestDevice *parent, void *buf,
                                size_t size);
    //
    // map_remote_pfn requests the dom0 VM to return the pfn of the pages
    // it is a hack to allow the enclave application to gain access of the
    // GTT buffer of the dom0 VM by directly mapping the pages of the dom0 VM.
    //
    // It should be done in the reverse direction -- it requires making HMM work
    // with I/O memory or reserved memory.
    absl::Status AllocGPUMemory(bool map_remote_pages, uint32_t gpu_id,
                                uint32_t flag, uint64_t mmap_offset);
    absl::Status MapGPUMemory();
    void SetResourceScopeFlag(ResourceScope scope) { scope_ = scope; }
    virtual absl::Status Destroy() override;
    ~EnclaveGuestMemory();

  private:
    absl::Status UnmapFromGPU();
    absl::Status DeallocateMemoryFromGPU();
    EnclaveGuestDevice *dev_;
    void *buf_;
    size_t size_;
    uint32_t gpu_id_;
    uint64_t handle_;
    ResourceScope scope_;
};

} // namespace ocl::hsa::enclave