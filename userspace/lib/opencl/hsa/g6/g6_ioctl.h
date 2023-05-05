#pragma once

#include <hsa/kfd_ioctl.h>

namespace ocl::hsa {

enum KFD_PIPE_PRIORITY {
    KFD_PIPE_PRIORITY_CS_LOW = 0,
    KFD_PIPE_PRIORITY_CS_MEDIUM,
    KFD_PIPE_PRIORITY_CS_HIGH
};

/* GPU ID hash width in bits */
#define KFD_GPU_ID_HASH_WIDTH 16

/* Use upper bits of mmap offset to store KFD driver specific information.
 * BITS[63:62] - Encode MMAP type
 * BITS[61:46] - Encode gpu_id. To identify to which GPU the offset belongs to
 * BITS[45:0]  - MMAP offset value
 *
 * NOTE: struct vm_area_struct.vm_pgoff uses offset in pages. Hence, these
 *  defines are w.r.t to PAGE_SIZE
 */
#define KFD_MMAP_TYPE_SHIFT 62
#define KFD_MMAP_TYPE_MASK (0x3ULL << KFD_MMAP_TYPE_SHIFT)
#define KFD_MMAP_TYPE_DOORBELL (0x3ULL << KFD_MMAP_TYPE_SHIFT)
#define KFD_MMAP_TYPE_EVENTS (0x2ULL << KFD_MMAP_TYPE_SHIFT)
#define KFD_MMAP_TYPE_RESERVED_MEM (0x1ULL << KFD_MMAP_TYPE_SHIFT)
#define KFD_MMAP_TYPE_MMIO (0x0ULL << KFD_MMAP_TYPE_SHIFT)

#define KFD_MMAP_GPU_ID_SHIFT 46
#define KFD_MMAP_GPU_ID_MASK                                                   \
    (((1ULL << KFD_GPU_ID_HASH_WIDTH) - 1) << KFD_MMAP_GPU_ID_SHIFT)
#define KFD_MMAP_GPU_ID(gpu_id)                                                \
    ((((uint64_t)gpu_id) << KFD_MMAP_GPU_ID_SHIFT) & KFD_MMAP_GPU_ID_MASK)
#define KFD_MMAP_GET_GPU_ID(offset)                                            \
    ((offset & KFD_MMAP_GPU_ID_MASK) >> KFD_MMAP_GPU_ID_SHIFT)

enum { kNumSDMAEngine = 2 };

struct kfd_ioctl_allocate_queue_ctx_args {
    __u32 gpu_id;
    __u32 type;
    __u32 sdma_id;
    __u64 doorbell_offset;
    __u64 doorbell_dw_offset_gpu_addr;
    __u64 mqd_gpu_addr;
    __u32 queue_id;
};

struct kfd_ioctl_install_queue_mqd_args {
    __u32 gpu_id;
    __u32 queue_id;
    __u64 mqd_user_ptr;
    __u32 mqd_size;
    __u64 write_pointer_address;
};

struct kfd_ioctl_dump_page_table_args {
    __u32 gpu_id;
    __u32 entry;         /* from / to KFD */
    __u64 metadata_ptr;  /* to KFD */
    __u64 pagetable_ptr; /* to KFD */
};

struct kfd_ioctl_dump_page_table_metadata {
    __u64 gpu_addr;
};

struct kfd_ioctl_dump_addr_space_map_info {
    __u64 va_start;
    __u64 gpu_phys_addr;
    __u64 size;
    __u32 flags;
    __u32 dma_addr_offset;
};

struct kfd_ioctl_dump_addr_space_args {
    __u32 gpu_id;
    __u64 doorbell_phys_addr_cpu;
    __u64 doorbell_size;
    __u32 mapping_entries;
    __u32 dma_addr_entries;
    __u64 mapping_info_ptr;
    __u64 dma_addr_ptr;
};

struct kfd_ioctl_set_root_pgt_args {
    __u32 gpu_id;
    __u64 root_pgt_ptr;
};

#define KFD_IOCTL_G6_RESERVED_PGT_REGION_PAGES 16

struct kfd_ioctl_set_pdb_args {
	__u32 gpu_id;
	__u64 pdb_phys_addr;
};

struct kfd_ioctl_acquire_vm_g6_args {
    __u32 gpu_id;
    __u64 pgt_region_gpu_addr[KFD_IOCTL_G6_RESERVED_PGT_REGION_PAGES];
    __u64 doorbell_gpu_addr;
    __u32 doorbell_slice_size;
    __u64 user_vram_gpu_addr;
    __u64 gtt_region_gpu_addr;
    __u64 gtt_region_size;
};

struct kfd_ioctl_set_pgt_args {
    __u32 gpu_id;
    __u32 pgt_id;
    __u64 src;
};

struct kfd_ioctl_set_event_page_args {
    __u32 gpu_id;
    __u64 gtt_region_offset;
};

#define AMDKFD_IOC_INSTALL_QUEUE_MQD                                           \
    AMDKFD_IOWR(0x22, struct kfd_ioctl_install_queue_mqd_args)

#define AMDKFD_IOC_ALLOCATE_QUEUE_CTX                                          \
    AMDKFD_IOWR(0x23, struct kfd_ioctl_allocate_queue_ctx_args)

#define AMDKFD_IOC_DUMP_PAGE_TABLE                                             \
    AMDKFD_IOWR(0x24, struct kfd_ioctl_dump_page_table_args)

#define AMDKFD_IOC_DUMP_ADDR_SPACE                                             \
    AMDKFD_IOWR(0x25, struct kfd_ioctl_dump_addr_space_args)

#define AMDKFD_IOC_SET_ROOT_PGT                                                \
    AMDKFD_IOW(0x26, struct kfd_ioctl_set_root_pgt_args)

#define AMDKFD_IOC_SET_PDB \
		AMDKFD_IOW(0x27, struct kfd_ioctl_set_pdb_args)

#define AMDKFD_IOC_ACQUIRE_VM_G6                                               \
    AMDKFD_IOWR(0x28, struct kfd_ioctl_acquire_vm_g6_args)

#define AMDKFD_IOC_SET_PGT AMDKFD_IOW(0x29, struct kfd_ioctl_set_pgt_args)

#define AMDKFD_IOC_SET_EVENT_PAGE                                              \
    AMDKFD_IOW(0x2a, struct kfd_ioctl_set_event_page_args)

} // namespace ocl::hsa