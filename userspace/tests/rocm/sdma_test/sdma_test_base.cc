#include "sdma_test_base.h"
#include "opencl/hsa/sdma_ops.h"
#include "opencl/hsa/signals.h"
#include "opencl/hsa/types.h"
#include <gtest/gtest.h>

namespace ocl::hsa {

void SDMATestBase::TestGetTimestamp(uint64_t *ts_addr, bool skip_signals) {
    static const size_t kTotalCommandSize =
        2 * SDMAOpsBuilder::kTimestampCommandSize +
        SDMAOpsBuilder::kAtomicCommandSize + SDMAOpsBuilder::kFenceCommandSize +
        SDMAOpsBuilder::kTrapCommandSize;

    if (!skip_signals) {
        signal_->Set(1);
    }

    *ts_addr = 0;

    uint64_t curr_index;
    char *command_addr =
        ops_queue_->AcquireWriteAddress(kTotalCommandSize, curr_index);
    ASSERT_NE(command_addr, nullptr);

    ocl::hsa::SDMAOpsBuilder builder(
        absl::Span<char>(command_addr, kTotalCommandSize));

    builder.GetGlobalTimestamp(reinterpret_cast<void *>(ts_addr));

    if (!skip_signals) {
        builder.AtomicDecrement((void *)signal_->ValueLocation())
            .Fence(reinterpret_cast<uint32_t *>(signal_->GetEventMailboxPtr()),
                   signal_->GetEventId())
            .Trap(signal_->GetEventId());
    }

    ops_queue_->ReleaseWriteAddress(curr_index, kTotalCommandSize);
    if (!skip_signals) {
        signal_->Barrier();
    } else {
        // A sufficiently long enough sleep to let the GPU catches up
        sleep(1);
    }

    ASSERT_NE(*ts_addr, 0);
}

void SDMATestBase::TestMemcpy(uint64_t *gtt_cpu_addr, gpu_addr_t gtt_buf,
                              gpu_addr_t tmp_buf) {
    static const uint64_t kMagic = 0xdeadbeefbeefdeadull;

    volatile uint64_t *base = const_cast<volatile uint64_t *>(gtt_cpu_addr);
    base[0] = kMagic;
    base[1] = 0;

    SimpleMemcpy(tmp_buf, gtt_buf, sizeof(uint64_t));
    SimpleMemcpy(gtt_buf + sizeof(uint64_t), tmp_buf, sizeof(uint64_t));

    ASSERT_EQ(base[1], kMagic);
}

void SDMATestBase::SimpleMemcpy(gpu_addr_t dst, gpu_addr_t src, size_t size) {
    static const size_t kTotalCommandSize =
        SDMAOpsBuilder::kLinearCopyCommandSize +
        SDMAOpsBuilder::kAtomicCommandSize + SDMAOpsBuilder::kFenceCommandSize +
        SDMAOpsBuilder::kTrapCommandSize;
    signal_->Set(1);

    uint64_t curr_index;
    char *command_addr =
        ops_queue_->AcquireWriteAddress(kTotalCommandSize, curr_index);
    ASSERT_NE(command_addr, nullptr);

    ocl::hsa::SDMAOpsBuilder builder(
        absl::Span<char>(command_addr, kTotalCommandSize));

    builder
        .Copy(reinterpret_cast<uint64_t *>(dst),
              reinterpret_cast<const uint64_t *>(src), sizeof(uint64_t))
        .AtomicDecrement((void *)signal_->ValueLocation())
        .Fence(reinterpret_cast<uint32_t *>(signal_->GetEventMailboxPtr()),
               signal_->GetEventId())
        .Trap(signal_->GetEventId());

    ops_queue_->ReleaseWriteAddress(curr_index, kTotalCommandSize);
    signal_->Barrier();
}

} // namespace ocl::hsa