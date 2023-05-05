#pragma once

#include <cassert>
#include <cstdint>

namespace ocl::hsa {
// https://github.com/ROCm-Developer-Tools/ROCm-ComputeABI-Doc/blob/master/AMDGPU-ABI.md#amd-kernel-code-object-amd_kernel_code_t
// https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor-for-gfx6-gfx10
#pragma pack(push, 1)

union KernelCodeProperties {
    struct {
        unsigned enable_sgpr_private_segment_buffer : 1; // use private buffer
        unsigned enable_sgpr_dispatch_ptr : 1;           // use setup
        unsigned enable_sgpr_queue_ptr : 1;
        unsigned enable_sgpr_kernarg_segment_ptr : 1; // use args
        unsigned enable_sgpr_dispatch_id : 1;
        unsigned enable_sgpr_flat_scratch_init : 1; // use flat scratch
        unsigned enable_sgpr_private_segment_size : 1;
        unsigned reserved0 : 3;
        unsigned enable_wavefront_size32 : 1;
        unsigned reserved1 : 5;
    } bits;
    uint16_t value;
};

static_assert(sizeof(KernelCodeProperties) == 2, "");

// https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-kernel-descriptor
struct CodeObjectV3KernelDescriptor {
    uint32_t group_segment_fixed_size;
    uint32_t private_segment_fixed_size;
    uint32_t kern_arg_size;
    uint32_t reserved0;
    int64_t kernel_code_entry_byte_offset;
    uint32_t reserved1[5];
    uint32_t pgm_rsrc3;
    uint32_t pgm_rsrc1;
    uint32_t pgm_rsrc2;
    KernelCodeProperties properties;
    uint8_t reserved2[6];
};
#pragma pack(pop)

static_assert(sizeof(CodeObjectV3KernelDescriptor) == 64, "");
} // namespace ocl::hsa