#include "guest_rpc_client.h"
#include "guest_platform.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/enclave/idl.h"
#include "utils/align.h"
#include <absl/types/span.h>
#include <cstdint>
#include <cstring>
#include <sched.h>

namespace ocl::hsa::enclave {
using namespace ocl::hsa::enclave::idl;

template <class T>
void GuestRPCClient::RPC(idl::RPCType req_type, const char *req,
                         unsigned request_size, idl::RPCType resp_type,
                         T get_response) {
    auto &tx = dev_->tx_;
    auto &rx = dev_->rx_;
    unsigned type_tag = req_type | (request_size << 8);

    tx.Push(type_tag, absl::MakeConstSpan(req, request_size));
    dev_->NotifyHostAgent();
    auto wptr = rx.GetWptr()->load();
    auto rptr = rx.GetRptr()->load();
    while (rptr == wptr) {
        sched_yield();
        wptr = rx.GetWptr()->load();
    }

    thread_local std::vector<char> tmp(TransmitBuffer::kMaxRPCSize);
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
        get_response(pkt, payload_size);
    }
    rx.GetRptr()->store(rptr);
}

GuestRPCClient::GuestRPCClient(EnclaveGuestDevice *dev) : dev_(dev) {}

idl::CreateQueueResponse
GuestRPCClient::CreateQueue(const CreateQueueRequest &req) {
    struct CreateQueueResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCCreateQueueRequest, req,
                        RPCType::kRPCCreateQueueResponse, &resp);
    return resp;
}

void GuestRPCClient::DestroyQueue(unsigned long queue_id) {
    NullResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCDestroyQueueRequest, queue_id,
                        RPCType::kRPCDestroyQueueResponse, &resp);
}

void GuestRPCClient::UpdateDoorbell(unsigned long doorbell_offset,
                                    unsigned long value) {
    struct UpdateDoorbellRequest req = {
        .doorbell_offset = doorbell_offset,
        .value = value,
    };
    NullResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCUpdateDoorbellRequest, req,
                        RPCType::kRPCUpdateDoorbellResponse, &resp);
}

void GuestRPCClient::AllocateGPUMemory(unsigned request_pfn, unsigned flag,
                                       uintptr_t va_addr, size_t size,
                                       unsigned long mmap_offset,
                                       AllocateGPUMemoryResponse *resp) {
    struct AllocateGPUMemoryRequest req = {
        .request_pfn = request_pfn,
        .flag = flag,
        .va_addr = va_addr,
        .size = size,
        .mmap_offset = mmap_offset,
    };

    RPC(RPCType::kRPCAllocateGPUMemoryRequest,
        reinterpret_cast<const char *>(&req), sizeof(req),
        RPCType::kRPCAllocateGPUMemoryResponse,
        [resp](const char *pkt, size_t size) {
            auto h =
                reinterpret_cast<const AllocateGPUMemoryResponseHeader *>(pkt);
            resp->handle = h->handle;
            resp->pfns.resize(h->num_pages);
            auto pfn = reinterpret_cast<const uintptr_t *>(h + 1);
            for (unsigned i = 0; i < h->num_pages; ++i) {
                resp->pfns[i] = pfn[i];
            }
        });
}

void GuestRPCClient::MapGPUMemory(unsigned long handle) {
    NullResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCMapGPUMemoryRequest, handle,
                        RPCType::kRPCMapGPUMemoryResponse, &resp);
}

void GuestRPCClient::UnmapGPUMemory(unsigned long handle) {
    NullResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCUnmapGPUMemoryRequest, handle,
                        RPCType::kRPCUnmapGPUMemoryResponse, &resp);
}

void GuestRPCClient::DeallocateGPUMemory(unsigned long handle) {
    NullResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCDeallocateGPUMemoryRequest, handle,
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
    FixedRequestSizeRPC(RPCType::kRPCCreateEventRequest, req,
                        RPCType::kRPCCreateEventResponse, &resp);
    return resp;
}

void GuestRPCClient::WaitEvent(unsigned event_id, unsigned timeout) {
    WaitEventRequest req = {.event_id = event_id, .timeout = timeout};
    NullResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCWaitEventRequest, req,
                        RPCType::kRPCWaitEventResponse, &resp);
}

void GuestRPCClient::DestroyEvent(unsigned event_id) {
    NullResponse resp;
    FixedRequestSizeRPC(RPCType::kRPCDestroyEventRequest, event_id,
                        RPCType::kRPCDestroyEventResponse, &resp);
}
} // namespace ocl::hsa::enclave