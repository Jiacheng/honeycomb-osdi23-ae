#include "device_context.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/sdma_ops.h"
#include "opencl/hsa/signals.h"
#include "utils/monad_runner.h"

#include <absl/status/status.h>
#include <map>
#include <memory>

namespace ocl::hip {

using gpu_addr_t = ocl::hsa::gpu_addr_t;
using ocl::hsa::SDMAOpsBuilder;

class HsaMemoryManager : public HipMemoryManagerBase {
  public:
    enum { kStagingSize = 1024 * 1024 };
    virtual hipError_t hipMalloc(gpu_addr_t *ptr, size_t size) override;
    virtual hipError_t hipFree(gpu_addr_t ptr) override;
    virtual hipError_t hipMemcpyDtoH(void *dst, gpu_addr_t src,
                                     size_t size) override;
    virtual hipError_t hipMemcpyHtoD(gpu_addr_t dst, const void *src,
                                     size_t size) override;
    virtual hipError_t hipMemcpyDtoD(gpu_addr_t dst, gpu_addr_t src,
                                     size_t size) override;

    virtual absl::Status Initialize() override;
    virtual absl::Status Destroy() override;

    explicit HsaMemoryManager(DeviceContext *parent, hsa::Signal *barrier);
    hipError_t CopySome(gpu_addr_t dst, gpu_addr_t src, size_t size);

    DeviceContext *parent_;
    std::unique_ptr<hsa::Memory> staging_;
    std::unique_ptr<hsa::SDMAQueue> sdma_queue_;
    std::unique_ptr<hsa::SDMAOpsQueue> sdma_ops_queue_;

    hsa::Signal *barrier_;
    std::map<gpu_addr_t, std::unique_ptr<hsa::Memory>> allocated_;
};

HsaMemoryManager::HsaMemoryManager(DeviceContext *parent, hsa::Signal *barrier)
    : parent_(parent), barrier_(barrier) {}

absl::Status HsaMemoryManager::Initialize() {
    auto dev = parent_->GetImplementation();
    auto mm = dev->GetMemoryManager();
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run([&]() {
            staging_ = mm->NewGTTMemory(kStagingSize, false);
            if (!staging_) {
                return absl::ResourceExhaustedError(
                    "Cannot allocate staging area");
            }
            return absl::OkStatus();
        })
        .Run([&]() {
            absl::Status stat;
            sdma_queue_ = hsa::SDMAQueue::Create(dev, &stat);
            return stat;
        })
        .Run([&]() { return sdma_queue_->Register(); })
        .Run([&]() {
            sdma_ops_queue_ =
                std::make_unique<hsa::SDMAOpsQueue>(sdma_queue_.get());
            return absl::OkStatus();
        });
    return runner.code();
}

absl::Status HsaMemoryManager::Destroy() {
    auto stat = staging_->Destroy();
    if (!stat.ok()) {
        return stat;
    }
    staging_.reset();
    sdma_ops_queue_.reset();
    stat = sdma_queue_->Destroy();
    if (!stat.ok()) {
        return stat;
    }
    sdma_queue_.reset();

    // XXX: A quick hack to catch all leaked memory in the benchmarks.
    // Technically, tce client can leak the memory and expect the driver / OS to
    // clean them up
    HSA_ASSERT(allocated_.empty() && "Leaked memory");
    return absl::OkStatus();
}

hipError_t HsaMemoryManager::hipMalloc(gpu_addr_t *ptr, size_t size) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    auto mm = parent_->GetImplementation()->GetMemoryManager();

    auto vram = mm->NewDeviceMemory(size);
    if (!vram) {
        return hipErrorOutOfMemory;
    }
    *ptr = vram->GetGPUAddress();
    allocated_[*ptr] = std::move(vram);

    return hipSuccess;
}

hipError_t HsaMemoryManager::hipFree(gpu_addr_t ptr) {
    if (!ptr) {
        return hipSuccess;
    }
    auto it = allocated_.find(ptr);
    if (it == allocated_.end()) {
        return hipErrorInvalidDevicePointer;
    }
    auto stat = it->second->Destroy();
    allocated_.erase(it);
    if (!stat.ok()) {
        return hipErrorInvalidDevicePointer;
    }
    return hipSuccess;
}

hipError_t HsaMemoryManager::CopySome(gpu_addr_t dst, gpu_addr_t src,
                                      size_t size) {
    static const size_t kTotalCommandSize =
        SDMAOpsBuilder::kLinearCopyCommandSize +
        SDMAOpsBuilder::kAtomicCommandSize + SDMAOpsBuilder::kFenceCommandSize +
        SDMAOpsBuilder::kTrapCommandSize;

    barrier_->Set(1);

    uint64_t curr_index;
    char *command_addr =
        sdma_ops_queue_->AcquireWriteAddress(kTotalCommandSize, curr_index);
    if (!command_addr) {
        return hipErrorInvalidValue;
    }

    SDMAOpsBuilder builder(absl::Span<char>(command_addr, kTotalCommandSize));

    builder
        .Copy(reinterpret_cast<uint64_t *>(dst),
              reinterpret_cast<const uint64_t *>(src), size)
        .AtomicDecrement((void *)barrier_->ValueLocation())
        .Fence(reinterpret_cast<uint32_t *>(barrier_->GetEventMailboxPtr()),
               barrier_->GetEventId())
        .Trap(barrier_->GetEventId());

    sdma_ops_queue_->ReleaseWriteAddress(curr_index, kTotalCommandSize);
    barrier_->Barrier();

    return hipSuccess;
}

hipError_t HsaMemoryManager::hipMemcpyDtoH(void *dst, gpu_addr_t src,
                                           size_t size) {
    // potentially barrier if there are dispatching kernels
    hipError_t err = parent_->GetComputeContext()->ReleaseDeviceMemoryFence();
    if (err != hipSuccess) {
        return err;
    }

    size_t remaining = size;
    size_t offset = 0;
    while (remaining > 0) {
        auto s = std::min<size_t>(remaining, kStagingSize);
        if ((err = CopySome(staging_->GetGPUAddress(), src + offset, s)) !=
            hipSuccess) {
            return err;
        }
        memcpy(reinterpret_cast<char *>(dst) + offset, staging_->GetBuffer(),
               s);
        offset += s;
        remaining -= s;
    }
    return hipSuccess;
}

hipError_t HsaMemoryManager::hipMemcpyHtoD(gpu_addr_t dst, const void *src,
                                           size_t size) {
    size_t remaining = size;
    size_t offset = 0;
    while (remaining > 0) {
        auto s = std::min<size_t>(remaining, kStagingSize);
        hipError_t err;
        memcpy(staging_->GetBuffer(),
               reinterpret_cast<const char *>(src) + offset, s);
        if ((err = CopySome(dst + offset, staging_->GetGPUAddress(), s)) !=
            hipSuccess) {
            return err;
        }
        offset += s;
        remaining -= s;
    }

    // AQL packet right after this must acquire in system scope
    parent_->GetComputeContext()->AddSystemScopeAcquire();

    return hipSuccess;
}

hipError_t HsaMemoryManager::hipMemcpyDtoD(gpu_addr_t dst, gpu_addr_t src,
                                           size_t size) {
    // potentially barrier if there are dispatching kernels
    hipError_t err = parent_->GetComputeContext()->ReleaseDeviceMemoryFence();
    if (err != hipSuccess) {
        return err;
    }
    return CopySome(dst, src, size);
}

std::unique_ptr<HipMemoryManagerBase>
HipMemoryManagerBase::NewHsaMemoryManager(DeviceContext *parent,
                                          hsa::Signal *barrier) {
    return std::unique_ptr<HipMemoryManagerBase>(
        new HsaMemoryManager(parent, barrier));
}

} // namespace ocl::hip
