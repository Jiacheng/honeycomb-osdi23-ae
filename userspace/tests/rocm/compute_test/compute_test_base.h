#pragma once

#include "opencl/hsa/hsa_program.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/signals.h"

namespace ocl::hsa {

class ComputeTestBase {
  protected:
    void HSAEnqueue(void *arg_buffer, uintptr_t kernel_code_handle,
                    Signal *signal, uint32_t group_segment_size);

    void TestLaunchKernel();
    void TestSharedMemory();

    MemoryManager *mm_;
    std::unique_ptr<SignalPool> signals_;
    std::unique_ptr<AQLQueue> queue_;
    Device *dev_;
};

} // namespace ocl::hsa
