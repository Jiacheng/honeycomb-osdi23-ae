#include "g6_queue.h"
#include "g6_ioctl.h"
#include "opencl/hsa/utils.h"

#include "amdgpu/asic_reg/gc/gc_10_1_0_sh_mask.h"
#include "amdgpu/v10_structs.h"

namespace ocl::hsa {

static void init_mqd(struct v10_compute_mqd *m, unsigned long addr,
                     const kfd_ioctl_create_queue_args *args,
                     unsigned long doorbell_off, unsigned vmid) {
    static const bool cwsr_enabled = true;
    memset(m, 0, sizeof(struct v10_compute_mqd));

    m->header = 0xC0310800;
    m->compute_pipelinestat_enable = 1;
    m->compute_static_thread_mgmt_se0 = 0xFFFFFFFF;
    m->compute_static_thread_mgmt_se1 = 0xFFFFFFFF;
    m->compute_static_thread_mgmt_se2 = 0xFFFFFFFF;
    m->compute_static_thread_mgmt_se3 = 0xFFFFFFFF;

    m->cp_hqd_persistent_state =
        CP_HQD_PERSISTENT_STATE__PRELOAD_REQ_MASK |
        0x53 << CP_HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT;

    m->cp_mqd_control = 1 << CP_MQD_CONTROL__PRIV_STATE__SHIFT;

    m->cp_mqd_base_addr_lo = lower_32_bits(addr);
    m->cp_mqd_base_addr_hi = upper_32_bits(addr);

    m->cp_hqd_quantum = 1 << CP_HQD_QUANTUM__QUANTUM_EN__SHIFT |
                        1 << CP_HQD_QUANTUM__QUANTUM_SCALE__SHIFT |
                        1 << CP_HQD_QUANTUM__QUANTUM_DURATION__SHIFT;

    m->cp_hqd_aql_control = 1 << CP_HQD_AQL_CONTROL__CONTROL0__SHIFT;

    if (cwsr_enabled) {
        m->cp_hqd_persistent_state |=
            (1 << CP_HQD_PERSISTENT_STATE__QSWITCH_MODE__SHIFT);
        m->cp_hqd_ctx_save_base_addr_lo =
            lower_32_bits(args->ctx_save_restore_address);
        m->cp_hqd_ctx_save_base_addr_hi =
            upper_32_bits(args->ctx_save_restore_address);
        m->cp_hqd_ctx_save_size = args->ctx_save_restore_size;
        m->cp_hqd_cntl_stack_size = args->ctl_stack_size;
        m->cp_hqd_cntl_stack_offset = args->ctl_stack_size;
        m->cp_hqd_wg_state_offset = args->ctl_stack_size;
    }

    m->cp_hqd_pq_control = 5 << CP_HQD_PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT;
    m->cp_hqd_pq_control |= ffs(args->ring_size / sizeof(unsigned int)) - 1 - 1;

    m->cp_hqd_pq_base_lo =
        lower_32_bits((uint64_t)args->ring_base_address >> 8);
    m->cp_hqd_pq_base_hi =
        upper_32_bits((uint64_t)args->ring_base_address >> 8);

    m->cp_hqd_pq_rptr_report_addr_lo =
        lower_32_bits((uint64_t)args->read_pointer_address);
    m->cp_hqd_pq_rptr_report_addr_hi =
        upper_32_bits((uint64_t)args->read_pointer_address);
    m->cp_hqd_pq_wptr_poll_addr_lo =
        lower_32_bits((uint64_t)args->write_pointer_address);
    m->cp_hqd_pq_wptr_poll_addr_hi =
        upper_32_bits((uint64_t)args->write_pointer_address);

    m->cp_hqd_pq_doorbell_control =
        doorbell_off << CP_HQD_PQ_DOORBELL_CONTROL__DOORBELL_OFFSET__SHIFT;

    m->cp_hqd_ib_control = 3 << CP_HQD_IB_CONTROL__MIN_IB_AVAIL_SIZE__SHIFT;

    /*
     * HW does not clamp this field correctly. Maximum EOP queue size
     * is constrained by per-SE EOP done signal count, which is 8-bit.
     * Limit is 0xFF EOP entries (= 0x7F8 dwords). CP will not submit
     * more than (EOP entry count - 1) so a queue size of 0x800 dwords
     * is safe, giving a maximum field value of 0xA.
     */
    m->cp_hqd_eop_control = std::min(
        0xA, ffs(args->eop_buffer_size / sizeof(unsigned int)) - 1 - 1);
    m->cp_hqd_eop_base_addr_lo = lower_32_bits(args->eop_buffer_address >> 8);
    m->cp_hqd_eop_base_addr_hi = upper_32_bits(args->eop_buffer_address >> 8);

    m->cp_hqd_iq_timer = 0;

    m->cp_hqd_vmid = vmid;

    /* GC 10 removed WPP_CLAMP from PQ Control */
    m->cp_hqd_pq_control |= CP_HQD_PQ_CONTROL__NO_UPDATE_RPTR_MASK |
                            2 << CP_HQD_PQ_CONTROL__SLOT_BASED_WPTR__SHIFT |
                            1 << CP_HQD_PQ_CONTROL__QUEUE_FULL_EN__SHIFT;
    m->cp_hqd_pq_doorbell_control |=
        1 << CP_HQD_PQ_DOORBELL_CONTROL__DOORBELL_BIF_DROP__SHIFT;

    if (cwsr_enabled) {
        m->cp_hqd_ctx_save_control = 0;
    }

    static_assert(offsetof(v10_compute_mqd, cp_mqd_base_addr_lo) == 512);
    m->cp_hqd_pipe_priority = KFD_PIPE_PRIORITY_CS_MEDIUM;
    m->cp_hqd_queue_priority = args->queue_priority;
}

absl::Status
G6AQLQueue::RegisterQueue(struct kfd_ioctl_create_queue_args *args) {
    int kfd_fd = Platform::Instance().GetKFDFD();

    kfd_ioctl_allocate_queue_ctx_args alloc_args = {
        .gpu_id = args->gpu_id,
        .type = args->queue_type,
    };

    int err = kmtIoctl(kfd_fd, AMDKFD_IOC_ALLOCATE_QUEUE_CTX, &alloc_args);
    if (err) {
        return absl::InvalidArgumentError("Cannot create queue");
    }

    args->queue_id = alloc_args.queue_id;
    v10_compute_mqd mqd;

    init_mqd(&mqd, alloc_args.mqd_gpu_addr, args,
             alloc_args.doorbell_dw_offset_gpu_addr, 0);

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

G6AQLQueue::G6AQLQueue(Device *dev) : AQLQueue(dev) {}

} // namespace ocl::hsa
