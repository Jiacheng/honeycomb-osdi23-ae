#pragma once

#include "idl.h"

namespace ocl::hsa::enclave {

class EnclaveGuestDevice;

class GuestRPCClient {
  public:
    explicit GuestRPCClient(EnclaveGuestDevice *dev);
    idl::CreateQueueResponse CreateQueue(const idl::CreateQueueRequest &req);
    void DestroyQueue(unsigned long queue_id);
    void UpdateDoorbell(unsigned long doorbell_offset, unsigned long value);
    void AllocateGPUMemory(unsigned request_pfn, unsigned flag,
                           uintptr_t va_addr, size_t size,
                           unsigned long mmap_offset,
                           idl::AllocateGPUMemoryResponse *resp);
    void MapGPUMemory(unsigned long handle);
    void UnmapGPUMemory(unsigned long handle);
    void DeallocateGPUMemory(unsigned long handle);
    idl::CreateEventResponse CreateEvent(unsigned event_type,
                                         unsigned long event_page_handle);
    void WaitEvent(unsigned event_id, unsigned timeout);
    void DestroyEvent(unsigned event_id);

  private:
    template <class T>
    void RPC(idl::RPCType req_type, const char *req, unsigned req_size,
             idl::RPCType resp_type, T get_response);
    template <class Request, class Response>
    void FixedRequestSizeRPC(idl::RPCType req_type, const Request &req,
                             idl::RPCType resp_type, Response *resp) {
        RPC(req_type, reinterpret_cast<const char *>(&req), sizeof(Request),
            resp_type, [resp](const char *pkt, size_t size) {
                *resp = *reinterpret_cast<const Response *>(pkt);
            });
    }
    EnclaveGuestDevice *dev_;
};
} // namespace ocl::hsa::enclave
