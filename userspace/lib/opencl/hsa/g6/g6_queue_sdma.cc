#include "g6_ioctl.h"
#include "g6_queue.h"
#include "opencl/hsa/utils.h"

#include "amdgpu/asic_reg/gc/gc_10_1_0_sh_mask.h"
#include "amdgpu/v10_structs.h"

namespace ocl::hsa {

#define SDMA_RLC_DUMMY_DEFAULT 0xf

absl::Status
G6SDMAQueue::RegisterQueue(struct kfd_ioctl_create_queue_args *args) {
    int kfd_fd = Platform::Instance().GetKFDFD();

    kfd_ioctl_allocate_queue_ctx_args alloc_args = {
        .gpu_id = args->gpu_id,
        .type = args->queue_type,
    };

    int err = kmtIoctl(kfd_fd, AMDKFD_IOC_ALLOCATE_QUEUE_CTX, &alloc_args);
    if (err) {
        return absl::InvalidArgumentError("Cannot create queue");
    }

    // Looks like the VMID is always 0
    unsigned vmid = 0;

    v10_sdma_mqd mqd = {
        0,
    };
    mqd.sdmax_rlcx_rb_cntl =
        (ffs(args->ring_size / sizeof(unsigned int)) - 1)
            << SDMA0_RLC0_RB_CNTL__RB_SIZE__SHIFT |
        vmid << SDMA0_RLC0_RB_CNTL__RB_VMID__SHIFT |
        1 << SDMA0_RLC0_RB_CNTL__RPTR_WRITEBACK_ENABLE__SHIFT |
        6 << SDMA0_RLC0_RB_CNTL__RPTR_WRITEBACK_TIMER__SHIFT;

    mqd.sdmax_rlcx_rb_base = lower_32_bits(args->ring_base_address >> 8);
    mqd.sdmax_rlcx_rb_base_hi = upper_32_bits(args->ring_base_address >> 8);
    mqd.sdmax_rlcx_rb_rptr_addr_lo =
        lower_32_bits((uint64_t)args->read_pointer_address);
    mqd.sdmax_rlcx_rb_rptr_addr_hi =
        upper_32_bits((uint64_t)args->read_pointer_address);
    mqd.sdmax_rlcx_doorbell_offset =
        alloc_args.doorbell_dw_offset_gpu_addr
        << SDMA0_RLC0_DOORBELL_OFFSET__OFFSET__SHIFT;

    mqd.sdma_engine_id = alloc_args.sdma_id % kNumSDMAEngine;
    mqd.sdma_queue_id = alloc_args.sdma_id / kNumSDMAEngine;
    mqd.sdmax_rlcx_dummy_reg = SDMA_RLC_DUMMY_DEFAULT;

    kfd_ioctl_install_queue_mqd_args mqd_args = {
        .gpu_id = alloc_args.gpu_id,
        .queue_id = alloc_args.queue_id,
        .mqd_user_ptr = (unsigned long)&mqd,
        .mqd_size = sizeof(mqd),
        .write_pointer_address = args->write_pointer_address,
    };
    err = kmtIoctl(kfd_fd, AMDKFD_IOC_INSTALL_QUEUE_MQD, &mqd_args);
    if (err) {
        return absl::InvalidArgumentError("Cannot create queue");
    }

    queue_id_ = args->queue_id;
    return InitializeDoorbell(alloc_args.doorbell_offset);
}

G6SDMAQueue::G6SDMAQueue(Device *dev) : SDMAQueue(dev) {}

} // namespace ocl::hsa
