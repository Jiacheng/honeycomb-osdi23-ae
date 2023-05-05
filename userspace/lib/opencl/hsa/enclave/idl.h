#pragma once

#include "opencl/hsa/assert.h"

#include <cstddef>
#include <cstdint>

namespace ocl::hsa::enclave::idl {

enum RPCType {
    kRPCCreateQueueRequest,
    kRPCCreateQueueResponse,
    kRPCDestroyQueueRequest,
    kRPCDestroyQueueResponse,
    kRPCUpdateDoorbellRequest,
    kRPCUpdateDoorbellResponse,
    kRPCAllocateGPUMemoryRequest,
    kRPCAllocateGPUMemoryResponse,
    kRPCMapGPUMemoryRequest,
    kRPCMapGPUMemoryResponse,
    kRPCUnmapGPUMemoryRequest,
    kRPCUnmapGPUMemoryResponse,
    kRPCDeallocateGPUMemoryRequest,
    kRPCDeallocateGPUMemoryResponse,
    kRPCCreateEventRequest,
    kRPCCreateEventResponse,
    kRPCWaitEventRequest,
    kRPCWaitEventResponse,
    kRPCDestroyEventRequest,
    kRPCDestroyEventResponse,
};

enum QueueType {
    kQueueTypeSDMA = 1,
    kQueueTypeAQL = 2,
};

struct CreateQueueRequest {
    uintptr_t ring_buffer_base;
    size_t ring_buffer_size;
    uintptr_t dispatch_base;
    QueueType type;
    uintptr_t eop_buffer_address;
    size_t eop_buffer_size;
    uintptr_t ctx_save_restore_address;
    size_t ctx_save_restore_size;
    size_t ctl_stack_size;
};

struct CreateQueueResponse {
    unsigned queue_id;
    unsigned long doorbell_offset;
};

struct UpdateDoorbellRequest {
    unsigned long doorbell_offset;
    unsigned long value;
};

struct AllocateGPUMemoryRequest {
    unsigned flag;
    uintptr_t va_addr;
    size_t size;
    unsigned long mmap_offset;
};

struct CreateEventRequest {
    unsigned long event_page_handle;
    unsigned event_type;
};

struct CreateEventResponse {
    unsigned event_id;
    unsigned long mailbox_ptr;
    unsigned long hw_data3;
};

struct WaitEventRequest {
    unsigned event_id;
    unsigned timeout;
};

typedef unsigned DestroyEventRequest;

typedef unsigned long DestroyQueueRequest;
typedef unsigned long MapGPUMemoryRequest;
typedef unsigned long UnmapGPUMemoryRequest;
typedef unsigned long DeallocateGPUMemoryRequest;

/*
 * Ensure that all RPCs are aligned on the dword boundaries.
 * That way the 4-byte tag never wraps around.
 */
typedef unsigned NullResponse;

typedef NullResponse DestroyQueueResponse;
typedef NullResponse UpdateDoorbellResponse;
typedef NullResponse MapGPUMemoryResponse;
typedef NullResponse UnmapGPUMemoryResponse;
typedef NullResponse DeallocateGPUMemoryResponse;
typedef unsigned long AllocateGPUMemoryResponse;
typedef NullResponse WaitEventResponse;
typedef NullResponse DestroyEventResponse;

static inline size_t GetPayloadSize(RPCType type) {
    switch (type) {
    case kRPCCreateQueueRequest:
        return sizeof(CreateQueueRequest);
    case kRPCCreateQueueResponse:
        return sizeof(CreateQueueResponse);
    case kRPCDestroyQueueRequest:
        return sizeof(DestroyQueueRequest);
    case kRPCDestroyQueueResponse:
        return sizeof(DestroyQueueResponse);
    case kRPCUpdateDoorbellRequest:
        return sizeof(UpdateDoorbellRequest);
    case kRPCUpdateDoorbellResponse:
        return sizeof(UpdateDoorbellResponse);
    case kRPCAllocateGPUMemoryRequest:
        return sizeof(AllocateGPUMemoryRequest);
    case kRPCAllocateGPUMemoryResponse:
        return sizeof(AllocateGPUMemoryResponse);
    case kRPCMapGPUMemoryRequest:
        return sizeof(MapGPUMemoryRequest);
    case kRPCMapGPUMemoryResponse:
        return sizeof(MapGPUMemoryResponse);
    case kRPCUnmapGPUMemoryRequest:
        return sizeof(UnmapGPUMemoryRequest);
    case kRPCUnmapGPUMemoryResponse:
        return sizeof(UnmapGPUMemoryResponse);
    case kRPCDeallocateGPUMemoryRequest:
        return sizeof(DeallocateGPUMemoryRequest);
    case kRPCDeallocateGPUMemoryResponse:
        return sizeof(DeallocateGPUMemoryResponse);
    case kRPCCreateEventRequest:
        return sizeof(CreateEventRequest);
    case kRPCCreateEventResponse:
        return sizeof(CreateEventResponse);
    case kRPCWaitEventRequest:
        return sizeof(WaitEventRequest);
    case kRPCWaitEventResponse:
        return sizeof(WaitEventResponse);
    case kRPCDestroyEventRequest:
        return sizeof(DestroyEventRequest);
    case kRPCDestroyEventResponse:
        return sizeof(DestroyEventResponse);
    }
    HSA_ASSERT(0 && "Unreachable");
    return 0;
}

struct HostConfiguration {
    uintptr_t gtt_vaddr;
    size_t gtt_size;
    uintptr_t vram_vaddr;
    size_t vram_size;
    unsigned node_id;
    unsigned gpu_id;
};

/*
 * Shared memory for the host agent.
 *
 * Layout of the configuration space:
 * 0 - 4k: the HostConfiguration struct (to be replaced with MMIO)
 * 4k - 4k + 8: rptr and wptr for RX ring buffer
 * 4k + 8 - 4k + 16: rptr and wptr for TX ring buffer
 * 8k - 16k: RX ring buffer for host: host <- guest
 * 16k - 24k: TX ring buffer for host: host -> guest
 */
struct ConfigurationSpaceLayout {
    struct Watermark {
        unsigned long rptr;
        unsigned long wptr;
    };
    enum {
        kConfigurationSpaceSize = 2 * 1024 * 1024,
        kTransmitBufferSize = 8 * 1024,
        kRXBufferWatermarkOffset = 4096,
        kTXBufferWatermarkOffset = 4096 + sizeof(Watermark),
        kRXBufferOffset = 8192,
        kTXBufferOffset = 16384,
    };
};

} // namespace ocl::hsa::enclave::idl
