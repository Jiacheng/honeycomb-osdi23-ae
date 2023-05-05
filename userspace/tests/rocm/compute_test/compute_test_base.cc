#include <absl/types/span.h>
#include <algorithm>
#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include "compute_test_base.h"

#include <cstring>
#include <fstream>

using namespace ocl::hsa;
using absl::Status;

static constexpr uint16_t kInvalidAql =
    (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);
static const auto kAqlHeader =
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);

void ComputeTestBase::HSAEnqueue(void *arg_buffer, uintptr_t kernel_code_handle,
                                 Signal *signal, uint32_t group_segment_size) {
    hsa_kernel_dispatch_packet_t dispatchPacket;
    memset(&dispatchPacket, 0, sizeof(dispatchPacket));

    dispatchPacket.header = kInvalidAql;
    dispatchPacket.kernel_object = kernel_code_handle;

    // dispatchPacket.header = aqlHeader_;
    // dispatchPacket.setup |= sizes.dimensions() <<
    // HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    dispatchPacket.grid_size_x = 1;
    dispatchPacket.grid_size_y = 1;
    dispatchPacket.grid_size_z = 1;

    dispatchPacket.workgroup_size_x = 1;
    dispatchPacket.workgroup_size_y = 1;
    dispatchPacket.workgroup_size_z = 1;

    dispatchPacket.kernarg_address = arg_buffer;
    dispatchPacket.group_segment_size = group_segment_size;
    dispatchPacket.private_segment_size = 0;

    if (signal) {
        struct hsa_signal_s h = {
            .handle = signal->GetHandle(),
        };
        dispatchPacket.completion_signal = h;
    }

    // Pass the header accordingly
    queue_->DispatchAQLPacket(&dispatchPacket, kAqlHeader,
                              1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
}

using namespace ocl::hsa;

struct Arguments {
    uint64_t gpu_addr;
    uint32_t val;
    uint32_t padding[15];
};

static_assert(sizeof(Arguments) == 0x48, "");

enum { kCodeEnd = 0xbf9f0000 };

static void SetupAssign(absl::Span<char> base,
                        absl::Span<const uint32_t> code) {
#pragma pack(push, 1)
    struct Data {
        struct CodeObjectV3KernelDescriptor desc;
        unsigned padding[48];
        uint32_t code[64];
    };
#pragma pack(pop)

    struct Data *d = reinterpret_cast<Data *>(base.data());
    memset(d, 0, sizeof(*d));
    d->desc.kern_arg_size = sizeof(Arguments);
    d->desc.kernel_code_entry_byte_offset = offsetof(Data, code);
    static_assert(offsetof(Data, code) % 256 == 0, "");
    d->desc.pgm_rsrc1 = 0x60af0000;
    d->desc.pgm_rsrc2 = 0x8c;
    d->desc.properties.value = 0x0409;

    std::fill_n(d->code, 64, kCodeEnd);
    std::copy(code.begin(), code.end(), d->code);
}

void ComputeTestBase::TestLaunchKernel() {
    enum { kCodeSize = 512 };

    // s_clause 0x1
    // s_load_dword s2, s[4:5], 0x8
    // s_load_dwordx2 s[0:1], s[4:5], 0x0
    // v_mov_b32_e32 v0, 0
    // s_waitcnt lgkmcnt(0)
    // v_mov_b32_e32 v1, s2
    // global_store_dword v0, v1, s[0:1]
    // s_endpgm
    static const uint32_t kMachineCodeAssign[] = {
        0xbfa10001, 0xf4000082, 0xfa000008, 0xf4040002, 0xfa000000, 0x7e000280,
        0xbf8cc07f, 0x7e020202, 0xdc708000, 0x00000100, 0xbf810000};

    auto mem = mm_->NewGTTMemory(1 << 20, false);
    auto base = (char *)mem->GetBuffer();
    SetupAssign(absl::Span<char>(base, 512),
                absl::MakeSpan(kMachineCodeAssign));

    auto args = (Arguments *)(base + kCodeSize);
    auto ptr = (unsigned *)(base + kCodeSize + sizeof(*args));
    args->gpu_addr = (uint64_t)ptr;
    args->val = 42;
    auto signal = signals_->GetSignal();
    signal->Set(1);
    HSAEnqueue(args, (uintptr_t)base, signal, 0);
    signal->Barrier();
    signals_->PutSignal(signal);

    unsigned result = *ptr;
    ASSERT_EQ(42, result);
}

void ComputeTestBase::TestSharedMemory() {
    enum { kCodeSize = 512 };
    // s_load_dword s0, s[4:5], 0x8
    // v_mov_b32_e32 v0, 0
    // s_waitcnt lgkmcnt(0)
    // v_mov_b32_e32 v1, s0
    // s_load_dwordx2 s[0:1], s[4:5], 0x0
    // ds_write_b32 v0, v1
    // s_waitcnt vmcnt(0) lgkmcnt(0)
    // s_waitcnt_vscnt null, 0x0
    // s_barrier
    // s_waitcnt vmcnt(0) lgkmcnt(0)
    // s_waitcnt_vscnt null, 0x0
    // buffer_gl0_inv
    // ds_read_b32 v1, v0
    // s_waitcnt lgkmcnt(0)
    // global_store_dword v0, v1, s[0:1]
    // s_endpgm
    // s_code_end
    static const uint32_t kMachineCodeShm[] = {
        0xf4000002, 0xfa000008, 0x7e000280, 0xbf8cc07f, 0x7e020200, 0xf4040002,
        0xfa000000, 0xd8340000, 0x00000100, 0xbf8c0070, 0xbbfd0000, 0xbf8a0000,
        0xbf8c0070, 0xbbfd0000, 0xe1c40000, 0x00000000, 0xd8d80000, 0x01000000,
        0xbf8cc07f, 0xdc708000, 0x00000100, 0xbf810000,
    };

    // Allocate new memory to avoid explicitly flushing the i-cache
    auto mem = mm_->NewGTTMemory(1 << 20, false);
    auto base = (char *)mem->GetBuffer();
    SetupAssign(absl::Span<char>(base, 512), absl::MakeSpan(kMachineCodeShm));

    auto args = (Arguments *)(base + kCodeSize);
    auto ptr = (unsigned *)(base + kCodeSize + sizeof(*args));
    args->gpu_addr = (uint64_t)ptr;
    args->val = 42;
    auto signal = signals_->GetSignal();
    signal->Set(1);
    HSAEnqueue(args, (uintptr_t)base, signal, 4);
    signal->Barrier();
    signals_->PutSignal(signal);

    unsigned result = *ptr;
    ASSERT_EQ(42, result);
}
