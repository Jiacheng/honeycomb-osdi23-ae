#pragma once

#include "opencl/hsa/enclave/idl.h"
#include "opencl/hsa/enclave/transmit_buffer.h"
#include "opencl/hsa/kfd/kfd_memory.h"
#include "opencl/hsa/kfd_event.h"
#include "opencl/hsa/platform.h"
#include "opencl/hsa/ring_allocator.h"

#include <absl/types/span.h>
#include <atomic>
#include <functional>
#include <map>
#include <memory>

namespace ocl::hsa::enclave {
class HostRequestHandler {
  public:
    explicit HostRequestHandler(Device *dev, TransmitBuffer *tx,
                                TransmitBuffer *rx);
    absl::Status Initialize();
    void ProcessRequests();

  private:
    void Dispatch(idl::RPCType ty, char *payload, size_t payload_size);
    void OnCreateQueue(const idl::CreateQueueRequest *req);
    void OnDestroyQueue(const idl::DestroyQueueRequest *req);
    void OnUpdateDoorbell(const idl::UpdateDoorbellRequest *req);
    void OnAllocGPUMemory(const idl::AllocateGPUMemoryRequest *req);
    void OnMapGPUMemory(const idl::MapGPUMemoryRequest *req);
    void OnUnmapGPUMemory(const idl::UnmapGPUMemoryRequest *req);
    void OnFreeGPUMemory(const idl::DeallocateGPUMemoryRequest *req);
    void OnCreateEvent(const idl::CreateEventRequest *req);
    void OnWaitEvent(const idl::WaitEventRequest *req);
    void OnDestroyEvent(const idl::DestroyEventRequest *req);

    void PushResponse(idl::RPCType type, absl::Span<const char> payload);
    uintptr_t GetPhysicalPageFrameNumber(uintptr_t va_addr);

    Device *dev_;
    TransmitBuffer *tx_;
    TransmitBuffer *rx_;
    //
    // XXX: The maps can be removed as the handler is stateless.
    // Keeping the maps for debugging for now.
    std::map<unsigned long, std::unique_ptr<KFDMemory>> mem_;
    std::map<unsigned, std::unique_ptr<Event>> events_;
    std::atomic<uint64_t> *doorbell_base_;

    int page_map_fd_;
};

} // namespace ocl::hsa::enclave
