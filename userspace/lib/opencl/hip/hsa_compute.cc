#include "device_context.h"
#include "module.h"
#include "opencl/hip/elf/amdgpu_program.h"
#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/ring_allocator.h"
#include "opencl/hsa/signals.h"

#include <hsa/hsa.h>
#include <memory>
#include <vector>

namespace ocl::hip {

static constexpr uint16_t kInvalidAql =
    (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);

static constexpr uint16_t kBarrierPacketHeader =
    (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

//
// Dispatch headers.
//
// often use this header
static const auto kDispatchPacketAgentScopeHeader =
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

// use this header only after memcpy
static const auto kDispatchPacketSystemScopeHeader =
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

class HsaComputeContext : public HipComputeContextBase {
  public:
    enum {
        kKernelArgRegionSize = 1024 * 1024,
        kKernelArgRegionAlign = 64,
        // "Additionally, the code aligns each kernarg block on a 64-byte cache
        // line boundary." see
        // https://rocmdocs.amd.com/en/latest/Tutorial/Optimizing-Dispatches.html
    };
    virtual hipError_t hipModuleLoadData(hipModule_t *module,
                                         const char *data) override;
    virtual hipError_t hipModuleGetFunction(hipFunction_t *function,
                                            hipModule_t module,
                                            const char *kname) override;
    virtual hipError_t hipModuleLaunchKernel(
        hipFunction_t f, unsigned globalDimX, unsigned globalDimY,
        unsigned globalDimZ, unsigned blockDimX, unsigned blockDimY,
        unsigned blockDimZ, const void *args, size_t arg_size) override;
    virtual hipError_t hipModuleUnload(hipModule_t module) override;
    virtual hipError_t
    ReleaseDeviceMemoryFence() override; // conditionally barrier

    explicit HsaComputeContext(DeviceContext *parent, hsa::Signal *barrier);

  private:
    // Dispatch a kernel to the hardware queue
    //
    // TODO: Return an object so that it is possible to track it in the host
    // queue and reclaim the argument buffer once the request retires.
    void Enqueue(hsa::gpu_addr_t arg_buffer, const Kernel *kern,
                 unsigned globalDimX, unsigned globalDimY, unsigned globalDimZ,
                 unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
                 hsa::Signal *barrier);
    hipError_t Barrier(); // always barrier
    absl::Status KernArgAllocate(uintptr_t *arg,
                                 const AMDGPUProgram::KernelInfo *ki);

    virtual absl::Status Initialize() override;
    virtual absl::Status Destroy() override;

    DeviceContext *parent_;
    hsa::Signal *barrier_;
    std::unique_ptr<hsa::AQLQueue> queue_;
    std::unique_ptr<hsa::Memory> kern_args_;
    hsa::RingAllocator kern_args_alloc_;
    std::vector<uint64_t> kernel_indices_;
};

HsaComputeContext::HsaComputeContext(DeviceContext *parent,
                                     hsa::Signal *barrier)
    : parent_(parent), barrier_(barrier) {}

absl::Status HsaComputeContext::Initialize() {
    auto dev = parent_->GetImplementation();
    auto mm = dev->GetMemoryManager();
    absl::Status stat;
    queue_ = hsa::AQLQueue::Create(dev, &stat);
    if (!stat.ok()) {
        return stat;
    }
    stat = queue_->Register();
    if (!stat.ok()) {
        return stat;
    }
    kern_args_ = mm->NewGTTMemory(kKernelArgRegionSize, false);
    if (!kern_args_) {
        return absl::ResourceExhaustedError("Cannot allocate kernel arguments");
    }
    kern_args_alloc_.Initialize(
        reinterpret_cast<uintptr_t>(kern_args_->GetBuffer()),
        kern_args_->GetSize(), kKernelArgRegionAlign);
    return absl::OkStatus();
}

absl::Status HsaComputeContext::Destroy() {
    auto stat = kern_args_->Destroy();
    if (!stat.ok()) {
        return stat;
    }
    stat = queue_->Destroy();
    if (!stat.ok()) {
        return stat;
    }
    return absl::OkStatus();
}

hipError_t HsaComputeContext::hipModuleLoadData(hipModule_t *module,
                                                const char *data) {
    std::unique_ptr<Module> mod;
    auto stat = Module::New(parent_, data, &mod);
    if (!stat.ok()) {
        auto s = std::string(stat.message());
        return hipErrorInvalidImage;
    }
    *module = reinterpret_cast<hipModule_t>(mod.release());
    return hipSuccess;
}

hipError_t HsaComputeContext::hipModuleGetFunction(hipFunction_t *function,
                                                   hipModule_t module,
                                                   const char *kname) {
    auto mod = reinterpret_cast<Module *>(module);
    auto kern = mod->GetKernel(kname);
    if (!kern) {
        return hipErrorNotFound;
    }
    *function = (hipFunction_t)kern;
    return hipSuccess;
}

absl::Status
HsaComputeContext::KernArgAllocate(uintptr_t *arg,
                                   const AMDGPUProgram::KernelInfo *ki) {
    while (
        !kern_args_alloc_.AvailableAlign(ki->kernarg_size, ki->kernarg_align)) {
        // "packet ID of the next AQL packet to be consumed by the compute unit
        // hardware" see HSA SysArch spec v1.2
        auto next_index =
            queue_->GetReadDispatchPtr()->load(std::memory_order_acquire);

        if (kernel_indices_.empty()) {
            sched_yield();
            continue;
        }

        auto index = kernel_indices_.front();
        if (index >= next_index) {
            sched_yield();
            continue;
        }

        while (index < next_index) {
            auto stat = kern_args_alloc_.Free();
            if (!stat.ok()) {
                return stat;
            }
            kernel_indices_.erase(kernel_indices_.begin());
            if (kernel_indices_.size() != 0) {
                index = kernel_indices_.front();
            } else {
                index = UINT64_MAX;
            }
        }
    }
    return kern_args_alloc_.AllocateAlign(arg, ki->kernarg_size,
                                          ki->kernarg_align);
}

hipError_t HsaComputeContext::hipModuleLaunchKernel(
    hipFunction_t f, unsigned globalDimX, unsigned globalDimY,
    unsigned globalDimZ, unsigned blockDimX, unsigned blockDimY,
    unsigned blockDimZ, const void *args, size_t arg_size) {
    auto kern = reinterpret_cast<const Kernel *>(f);
    auto ki = kern->GetKernelInfo();

    // No support for spills yet
    if (ki->private_segment_fixed_size) {
        return hipErrorLaunchFailure;
    }

    uintptr_t d_arg;
    auto stat = KernArgAllocate(&d_arg, ki);
    if (!stat.ok()) {
        return hipErrorLaunchFailure;
    }
    auto s = std::min(ki->kernarg_size, arg_size);
    auto dst = reinterpret_cast<char *>(d_arg);
    memcpy(dst, args, s);
    if (ki->kernarg_size > s) {
        memset(dst + s, 0, ki->kernarg_size - s);
    }

    Enqueue(d_arg, kern, globalDimX, globalDimY, globalDimZ, blockDimX,
            blockDimY, blockDimZ, nullptr);
    return hipSuccess;
}

hipError_t HsaComputeContext::Barrier() {
    barrier_->Set(1);
    hsa_barrier_and_packet_t pkt = {
        0,
    };
    pkt.header = kInvalidAql;
    pkt.completion_signal.handle = barrier_->GetHandle();

    queue_->DispatchAQLPacket(&pkt, kBarrierPacketHeader, 0);
    barrier_->Barrier();
    return hipSuccess;
}

hipError_t HsaComputeContext::ReleaseDeviceMemoryFence() {
    hipError_t err = hipSuccess;
    if (pending_dispatch_) {
        err = Barrier();
        pending_dispatch_ = false;
    }
    return err;
}

void HsaComputeContext::Enqueue(hsa::gpu_addr_t arg_buffer, const Kernel *kern,
                                unsigned globalDimX, unsigned globalDimY,
                                unsigned globalDimZ, unsigned blockDimX,
                                unsigned blockDimY, unsigned blockDimZ,
                                hsa::Signal *barrier) {
    auto ki = kern->GetKernelInfo();
    hsa_kernel_dispatch_packet_t dispatchPacket;
    memset(&dispatchPacket, 0, sizeof(dispatchPacket));

    dispatchPacket.header = kInvalidAql;
    dispatchPacket.kernel_object =
        kern->GetParent()->GetVMABase() + ki->desc_vma_offset;

    // dispatchPacket.header = aqlHeader_;
    // dispatchPacket.setup |= sizes.dimensions() <<
    // HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    dispatchPacket.grid_size_x = globalDimX;
    dispatchPacket.grid_size_y = globalDimY;
    dispatchPacket.grid_size_z = globalDimZ;

    dispatchPacket.workgroup_size_x = blockDimX;
    dispatchPacket.workgroup_size_y = blockDimY;
    dispatchPacket.workgroup_size_z = blockDimZ;

    dispatchPacket.kernarg_address = reinterpret_cast<void *>(arg_buffer);
    dispatchPacket.group_segment_size = ki->lds_size;
    dispatchPacket.private_segment_size = ki->private_segment_fixed_size;

    if (barrier) {
        struct hsa_signal_s h = {
            .handle = barrier->GetHandle(),
        };
        dispatchPacket.completion_signal = h;
    }

    // set acquire to system scope
    auto dispatch_header = kDispatchPacketAgentScopeHeader;
    if (system_scope_acquire_) {
        dispatch_header = kDispatchPacketSystemScopeHeader;
        system_scope_acquire_ = false;
    }
    /*
     * NOTE: actually more processing can be done for the AQL header
     * for a better performance, you can see processMemObjects
     * in ROCclr for more details
     * e.g. when a kernel involves no global memory access
     * or only needs to access some queue/sampler, then no
     * barrier is needed
     */

    // Seems like that original HIP implementation always activates all three
    // dimensions.
    auto index = queue_->DispatchAQLPacket(
        &dispatchPacket, dispatch_header,
        3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);

    // each index corresponds to a KernArgAllocate
    // we must Free it according to the rptr of the AQL queue
    kernel_indices_.push_back(index);

    // set pending dispatch flag for memcpy from device
    pending_dispatch_ = true;
}

hipError_t HsaComputeContext::hipModuleUnload(hipModule_t module) {
    delete reinterpret_cast<Module *>(module);
    return hipSuccess;
}
std::unique_ptr<HipComputeContextBase>
HipComputeContextBase::NewHsaComputeContext(DeviceContext *parent,
                                            hsa::Signal *barrier) {
    return std::unique_ptr<HipComputeContextBase>(
        new HsaComputeContext(parent, barrier));
}

} // namespace ocl::hip
