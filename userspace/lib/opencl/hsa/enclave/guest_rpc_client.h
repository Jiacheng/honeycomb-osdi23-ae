#pragma once

#include "idl.h"

namespace ocl::hsa::enclave {

class EnclaveGuestDevice;

class GuestRPCClient {
  public:
    explicit GuestRPCClient(EnclaveGuestDevice *dev);
    idl::CreateQueueResponse CreateQueue(const idl::CreateQueueRequest &req);
    idl::CreateQueueResponse
    CreateQueue(uintptr_t ring_buffer_base, size_t ring_buffer_size,
                uintptr_t dispatch_base, idl::QueueType type,
                uintptr_t eop_buffer_address, size_t eop_buffer_size,
                uintptr_t ctx_save_restore_address,
                size_t ctx_save_restore_size, size_t ctl_stack_size);
    void DestroyQueue(unsigned long queue_id);
    void UpdateDoorbell(unsigned long doorbell_offset, unsigned long value);
    void AllocateGPUMemory(unsigned flag, uintptr_t va_addr, size_t size,
                           unsigned long mmap_offset, unsigned long *handle);
    void MapGPUMemory(unsigned long handle);
    void UnmapGPUMemory(unsigned long handle);
    void DeallocateGPUMemory(unsigned long handle);
    idl::CreateEventResponse CreateEvent(unsigned event_type,
                                         unsigned long event_page_handle);
    void WaitEvent(unsigned event_id, unsigned timeout);
    void DestroyEvent(unsigned event_id);

  private:
    template <class Request, class Response>
    void RPC(idl::RPCType req_type, const Request &req, idl::RPCType resp_type,
             Response *resp);
    EnclaveGuestDevice *dev_;
};
} // namespace ocl::hsa::enclave
