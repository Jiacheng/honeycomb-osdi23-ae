#pragma once

#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/types.h"

namespace ocl::hsa {

class G6Memory : public Memory {
  public:
    explicit G6Memory(gpu_addr_t gpu_addr, size_t size);
    virtual size_t GetSize() const { return size_; }
    //
    // Return the underlying handle for the GPU memory.
    virtual uint64_t GetHandle() const { return 0; }
    //
    // Return the CPU address if the memory is a piece of GTT memory and is
    // mapped into the user-level process.
    virtual void *GetBuffer() const { return (void *)gpu_addr_; }

    // Return the GPU address of the memory.
    //
    // The ROCM driver uses mmap() to manage the virtual address spaces for all
    // GPUs and the user-level process, thus (1) a piece of memory that is
    // shared across multiple GPUs has a unique address, and (2) a piece of GTT
    // memory has identical a CPU and GPU address.
    //
    // We introduce a separate API here to support a fixed memory layout for GPU
    // programs. The memory layout can help simplify the proofs on showing all
    // memory accesses are safe and not leaking information.
    virtual gpu_addr_t GetGPUAddress() const { return gpu_addr_; }

    virtual absl::Status Destroy() override { return absl::OkStatus(); }

  protected:
    gpu_addr_t gpu_addr_;
    size_t size_;
};
} // namespace ocl::hsa