#include "guest_rpc_client.h"
#include "guest_platform.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/enclave/idl.h"
#include <absl/types/span.h>
#include <sched.h>

namespace ocl::hsa::enclave {
using namespace ocl::hsa::enclave::idl;

template <class Request, class Response>
void GuestRPCClient::RPC(idl::RPCType req_type, const Request &req,
                         idl::RPCType resp_type, Response *resp) {
    auto &tx = dev_->tx_;
    auto &rx = dev_->rx_;

    tx.Push(req_type, absl::MakeConstSpan(reinterpret_cast<const char *>(&req),
                                          sizeof(req)));
    dev_->NotifyHostAgent();
    auto wptr = rx.GetWptr()->load();
    auto rptr = rx.GetRptr()->load();
    while (rptr == wptr) {
        sched_yield();
        wptr = rx.GetWptr()->load();
    }

    char tmp[TransmitBuffer::kMaxRPCSize];
    auto size = rx.GetBuffer().size();
    while (rptr != wptr) {
        RPCType ty;
        size_t payload_size;
        auto pkt =
            rx.ReadPacketAt(rptr, &ty, &payload_size, absl::MakeSpan(tmp));
        HSA_ASSERT(pkt && ty == resp_type);
        rptr = (rptr + sizeof(RPCType) + payload_size) & (size - 1);
        // This is a FIFO queue since there is no supports for streams yet.
        HSA_ASSERT(rptr == wptr);
        *resp = *reinterpret_cast<const Response *>(pkt);
    }
    rx.GetRptr()->store(rptr);
}

GuestRPCClient::GuestRPCClient(EnclaveGuestDevice *dev) : dev_(dev) {}

idl::CreateQueueResponse GuestRPCClient::CreateQueue(
    uintptr_t ring_buffer_base, size_t ring_buffer_size,
    uintptr_t dispatch_base, idl::QueueType type, uintptr_t eop_buffer_address,
    size_t eop_buffer_size, uintptr_t ctx_save_restore_address,
    size_t ctx_save_restore_size, size_t ctl_stack_size) {
    struct CreateQueueRequest req = {
        .ring_buffer_base = ring_buffer_base,
        .ring_buffer_size = ring_buffer_size,
        .dispatch_base = dispatch_base,
        .type = type,
        .eop_buffer_address = eop_buffer_address,
        .eop_buffer_size = eop_buffer_size,
        .ctx_save_restore_address = ctx_save_restore_address,
        .ctx_save_restore_size = ctx_save_restore_size,
        .ctl_stack_size = ctl_stack_size,
    };
    struct CreateQueueResponse resp;
    RPC(RPCType::kRPCCreateQueueRequest, req, RPCType::kRPCCreateQueueResponse,
        &resp);
    return resp;
}

idl::CreateQueueResponse
GuestRPCClient::CreateQueue(const CreateQueueRequest &req) {
    struct CreateQueueResponse resp;
    RPC(RPCType::kRPCCreateQueueRequest, req, RPCType::kRPCCreateQueueResponse,
        &resp);
    return resp;
}

void GuestRPCClient::DestroyQueue(unsigned long queue_id) {
    NullResponse resp;
    RPC(RPCType::kRPCDestroyQueueRequest, queue_id,
        RPCType::kRPCDestroyQueueResponse, &resp);
}

void GuestRPCClient::UpdateDoorbell(unsigned long doorbell_offset,
                                    unsigned long value) {
    struct UpdateDoorbellRequest req = {
        .doorbell_offset = doorbell_offset,
        .value = value,
    };
    NullResponse resp;
    RPC(RPCType::kRPCUpdateDoorbellRequest, req,
        RPCType::kRPCUpdateDoorbellResponse, &resp);
}

void GuestRPCClient::AllocateGPUMemory(unsigned flag, uintptr_t va_addr,
                                       size_t size, unsigned long mmap_offset,
                                       unsigned long *resp) {
    struct AllocateGPUMemoryRequest req = {
        .flag = flag,
        .va_addr = va_addr,
        .size = size,
        .mmap_offset = mmap_offset,
    };
    RPC(RPCType::kRPCAllocateGPUMemoryRequest, req,
        RPCType::kRPCAllocateGPUMemoryResponse, resp);
}

void GuestRPCClient::MapGPUMemory(unsigned long handle) {
    NullResponse resp;
    RPC(RPCType::kRPCMapGPUMemoryRequest, handle,
        RPCType::kRPCMapGPUMemoryResponse, &resp);
}

void GuestRPCClient::UnmapGPUMemory(unsigned long handle) {
    NullResponse resp;
    RPC(RPCType::kRPCUnmapGPUMemoryRequest, handle,
        RPCType::kRPCUnmapGPUMemoryResponse, &resp);
}

void GuestRPCClient::DeallocateGPUMemory(unsigned long handle) {
    NullResponse resp;
    RPC(RPCType::kRPCDeallocateGPUMemoryRequest, handle,
        RPCType::kRPCDeallocateGPUMemoryResponse, &resp);
}

CreateEventResponse
GuestRPCClient::CreateEvent(unsigned event_type,
                            unsigned long event_page_handle) {
    CreateEventRequest req = {
        .event_page_handle = event_page_handle,
        .event_type = event_type,
    };
    CreateEventResponse resp;
    RPC(RPCType::kRPCCreateEventRequest, req, RPCType::kRPCCreateEventResponse,
        &resp);
    return resp;
}

void GuestRPCClient::WaitEvent(unsigned event_id, unsigned timeout) {
    WaitEventRequest req = {.event_id = event_id, .timeout = timeout};
    NullResponse resp;
    RPC(RPCType::kRPCWaitEventRequest, req, RPCType::kRPCWaitEventResponse,
        &resp);
}

void GuestRPCClient::DestroyEvent(unsigned event_id) {
    NullResponse resp;
    RPC(RPCType::kRPCDestroyEventRequest, event_id,
        RPCType::kRPCDestroyEventResponse, &resp);
}

} // namespace ocl::hsa::enclave
