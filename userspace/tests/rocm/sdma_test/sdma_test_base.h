#pragma once

#include "opencl/hsa/sdma_ops.h"
#include "opencl/hsa/signals.h"
#include "opencl/hsa/types.h"

namespace ocl::hsa {

class SDMATestBase {
  protected:
    void TestGetTimestamp(uint64_t *ts_addr, bool skip_signals = false);
    void TestMemcpy(uint64_t *gtt_cpu_addr, gpu_addr_t gtt_buf,
                    gpu_addr_t tmp_buf);
    void SimpleMemcpy(gpu_addr_t dst, gpu_addr_t src, size_t size);

    std::unique_ptr<SignalPool> signals_;
    Signal *signal_;
    std::unique_ptr<SDMAOpsQueue> ops_queue_;
};

} // namespace ocl::hsa