#pragma once

#include "opencl/hsa/memory_manager.h"

namespace ocl::hsa {

namespace enclave {
class HostRequestHandler;
}

//
// GTT memory is a contingous virtual memory on the host that mapped into the
// GPUs with identical virtual addresses. The userspace driver is responsible to
// allocate the virtual address space of the GPU.
//
// The current implementation lacks a number of interesting features:
// (1) Mapping the same memory into multiple GPUs. (2) Bookkeeping the address
// spaces of the GPU.
//
// The KFDMemory class is a wrapper to help proper bookkeeping of the
// resouruces.
class KFDMemory : public Memory {
  public:
    friend class MemoryManager;
    friend class enclave::HostRequestHandler;
    virtual void *GetBuffer() const override { return buf_; }
    virtual size_t GetSize() const override { return size_; }
    virtual uint64_t GetHandle() const override { return handle_; }
    virtual gpu_addr_t GetGPUAddress() const override {
        return reinterpret_cast<gpu_addr_t>(buf_);
    }

    explicit KFDMemory(void *buf, size_t size);
    absl::Status AllocGPUMemory(uint32_t gpu_id, uint32_t flag,
                                uint64_t mmap_offset);
    absl::Status MapGPUMemory();
    void SetResourceScopeFlag(ResourceScope scope) { scope_ = scope; }
    virtual absl::Status Destroy() override;
    ~KFDMemory();

  private:
    absl::Status UnmapFromGPU();
    absl::Status DeallocateMemoryFromGPU();
    void *buf_;
    size_t size_;
    uint32_t gpu_id_;
    uint64_t handle_;
    ResourceScope scope_;
};

} // namespace ocl::hsa