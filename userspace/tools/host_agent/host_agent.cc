#include "host_agent.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/enclave/idl.h"
#include "opencl/hsa/kfd/kfd_memory.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/utils.h"

#include <absl/types/span.h>
#include <atomic>
#include <memory>

#include <hsa/kfd_ioctl.h>
#include <sched.h>

namespace ocl::hsa::enclave {

using namespace ocl::hsa::enclave::idl;

static absl::Span<const char> NullResponseSpan() {
    static const NullResponse kNullResponse = 0;
    return absl::MakeConstSpan(reinterpret_cast<const char *>(&kNullResponse),
                               sizeof(NullResponse));
}

template <class T> static inline absl::Span<const char> AsSpan(const T &v) {
    return absl::MakeConstSpan(reinterpret_cast<const char *>(&v), sizeof(T));
}

HostRequestHandler::HostRequestHandler(Device *dev, TransmitBuffer *tx,
                                       TransmitBuffer *rx)
    : dev_(dev), tx_(tx), rx_(rx) {}

void HostRequestHandler::ProcessRequests() {
    auto buf = rx_->GetBuffer();
    auto rptr = rx_->GetRptr()->load();
    auto wptr = rx_->GetWptr()->load();
    auto size = buf.size();

    while (rptr == wptr) {
        sched_yield();
        wptr = rx_->GetWptr()->load();
    }

    char tmp[TransmitBuffer::kMaxRPCSize];
    while (rptr != wptr) {
        RPCType ty;
        size_t payload_size;
        auto pkt =
            rx_->ReadPacketAt(rptr, &ty, &payload_size, absl::MakeSpan(tmp));
        HSA_ASSERT(pkt);
        Dispatch(ty, pkt);
        rptr = (rptr + sizeof(RPCType) + payload_size) & (size - 1);
    }
    rx_->GetRptr()->store(rptr);
}

void HostRequestHandler::Dispatch(RPCType ty, const char *req) {
    switch (ty) {
    case kRPCCreateQueueRequest:
        OnCreateQueue(reinterpret_cast<const CreateQueueRequest *>(req));
        break;
    case kRPCDestroyQueueRequest:
        OnDestroyQueue(reinterpret_cast<const DestroyQueueRequest *>(req));
        break;
    case kRPCUpdateDoorbellRequest:
        OnUpdateDoorbell(reinterpret_cast<const UpdateDoorbellRequest *>(req));
        break;
    case kRPCAllocateGPUMemoryRequest:
        OnAllocGPUMemory(
            reinterpret_cast<const AllocateGPUMemoryRequest *>(req));
        break;
    case kRPCMapGPUMemoryRequest:
        OnMapGPUMemory(reinterpret_cast<const MapGPUMemoryRequest *>(req));
        break;
    case kRPCUnmapGPUMemoryRequest:
        OnUnmapGPUMemory(reinterpret_cast<const UnmapGPUMemoryRequest *>(req));
        break;
    case kRPCDeallocateGPUMemoryRequest:
        OnFreeGPUMemory(
            reinterpret_cast<const DeallocateGPUMemoryRequest *>(req));
        break;
    case kRPCCreateEventRequest:
        OnCreateEvent(reinterpret_cast<const CreateEventRequest *>(req));
        break;
    case kRPCWaitEventRequest:
        OnWaitEvent(reinterpret_cast<const WaitEventRequest *>(req));
        break;
    case kRPCDestroyEventRequest:
        OnDestroyEvent(reinterpret_cast<const DestroyEventRequest *>(req));
        break;
    default:
        HSA_ASSERT(0 && "unreachable");
        break;
    }
}

void HostRequestHandler::OnCreateQueue(const idl::CreateQueueRequest *req) {
    struct kfd_ioctl_create_queue_args args = {0};

    auto q = reinterpret_cast<DeviceQueue::SharedQueueWatermark *>(
        req->dispatch_base);
    args.gpu_id = dev_->GetGPUID();
    args.queue_type = req->type;
    args.read_pointer_address = (uintptr_t)&q->rptr;
    args.write_pointer_address = (uintptr_t)&q->wptr;
    args.ring_base_address = req->ring_buffer_base;
    args.ring_size = req->ring_buffer_size;
    args.queue_percentage = 100;
    args.queue_priority = 7;

    args.eop_buffer_address = req->eop_buffer_address;
    args.eop_buffer_size = req->eop_buffer_size;
    args.ctx_save_restore_address = req->ctx_save_restore_address;
    args.ctx_save_restore_size = req->ctx_save_restore_size;
    args.ctl_stack_size = req->ctl_stack_size;

    int kfd_fd = Platform::Instance().GetKFDFD();
    int err = kmtIoctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, &args);
    HSA_ASSERT(err == 0 && "Cannot create queue");

    auto queue_id = args.queue_id;
    auto doorbell_offset = args.doorbell_offset;

    const uint64_t doorbell_page_size = dev_->GetDoorbellPageSize();
    /* On SOC15 chips, the doorbell offset within the
     * doorbell page is included in the doorbell offset
     * returned by KFD. This allows CP queue doorbells to be
     * allocated dynamically (while SDMA queue doorbells fixed)
     * rather than based on the its process queue ID.
     */
    uint64_t doorbell_mmap_offset = doorbell_offset & ~(doorbell_page_size - 1);
    doorbell_offset = doorbell_offset & (dev_->GetDoorbellPageSize() - 1);

    doorbell_base_ = reinterpret_cast<std::atomic<uint64_t> *>(
        dev_->GetOrInitializeDoorbell(doorbell_mmap_offset));
    HSA_ASSERT(doorbell_base_);
    CreateQueueResponse resp = {
        .queue_id = queue_id,
        .doorbell_offset = doorbell_offset,
    };
    PushResponse(RPCType::kRPCCreateQueueResponse, AsSpan(resp));
}

void HostRequestHandler::OnDestroyQueue(const idl::DestroyQueueRequest *req) {
    auto queue_id = *req;
    struct kfd_ioctl_destroy_queue_args args = {0};
    args.queue_id = queue_id;

    int kfd_fd = Platform::Instance().GetKFDFD();
    int err = kmtIoctl(kfd_fd, AMDKFD_IOC_DESTROY_QUEUE, &args);
    HSA_ASSERT(err == 0);
    PushResponse(RPCType::kRPCDestroyQueueResponse, NullResponseSpan());
}

void HostRequestHandler::OnUpdateDoorbell(
    const idl::UpdateDoorbellRequest *req) {
    HSA_ASSERT(req->doorbell_offset < dev_->GetDoorbellPageSize());
    auto ptr = doorbell_base_ + req->doorbell_offset / sizeof(uint64_t);
    ptr->store(req->value, std::memory_order_release);
    PushResponse(RPCType::kRPCUpdateDoorbellResponse, NullResponseSpan());
}

void HostRequestHandler::OnAllocGPUMemory(
    const idl::AllocateGPUMemoryRequest *req) {
    std::unique_ptr<KFDMemory> m(
        new KFDMemory(reinterpret_cast<void *>(req->va_addr), req->size));
    auto stat =
        m->AllocGPUMemory(dev_->GetGPUID(), req->flag, req->mmap_offset);
    HSA_ASSERT(stat.ok());
    m->SetResourceScopeFlag(KFDMemory::kUnmanagedMmap);
    unsigned long handle = m->GetHandle();
    mem_.insert(std::make_pair<unsigned long, std::unique_ptr<KFDMemory>>(
        std::move(handle), std::move(m)));

    PushResponse(RPCType::kRPCAllocateGPUMemoryResponse, AsSpan(handle));
}

void HostRequestHandler::OnMapGPUMemory(const idl::MapGPUMemoryRequest *req) {
    auto it = mem_.find(*req);
    HSA_ASSERT(it != mem_.end());
    auto stat = it->second->MapGPUMemory();
    HSA_ASSERT(stat.ok());
    PushResponse(RPCType::kRPCMapGPUMemoryResponse, NullResponseSpan());
}

void HostRequestHandler::OnUnmapGPUMemory(
    const idl::UnmapGPUMemoryRequest *req) {
    auto it = mem_.find(*req);
    HSA_ASSERT(it != mem_.end());
    // UnmapFromGPU will be called by onFreeGPUMemory
    // auto stat = it->second->UnmapFromGPU();
    // HSA_ASSERT(stat.ok());
    PushResponse(RPCType::kRPCUnmapGPUMemoryResponse, NullResponseSpan());
}

void HostRequestHandler::OnFreeGPUMemory(
    const idl::DeallocateGPUMemoryRequest *req) {
    auto it = mem_.find(*req);
    HSA_ASSERT(it != mem_.end());
    auto stat = it->second->Destroy();
    HSA_ASSERT(stat.ok());
    mem_.erase(it);
    PushResponse(RPCType::kRPCDeallocateGPUMemoryResponse, NullResponseSpan());
}

void HostRequestHandler::OnCreateEvent(const idl::CreateEventRequest *req) {
    auto event = KFDEvent::New(req->event_type, req->event_page_handle);
    HSA_ASSERT(event && "Failed to create event");
    unsigned id = event->GetEventID();
    CreateEventResponse resp = {
        .event_id = event->GetEventID(),
        .mailbox_ptr = event->GetEventMailboxPtr(),
        .hw_data3 = event->GetHWData3(),
    };
    events_.insert(std::make_pair(std::move(id), std::move(event)));
    PushResponse(RPCType::kRPCCreateEventResponse, AsSpan(resp));
}

void HostRequestHandler::OnWaitEvent(const idl::WaitEventRequest *req) {
    auto it = events_.find(req->event_id);
    HSA_ASSERT(it != events_.end());
    auto _ = it->second->Wait(req->timeout);
    PushResponse(RPCType::kRPCWaitEventResponse, NullResponseSpan());
}

void HostRequestHandler::OnDestroyEvent(const idl::DestroyEventRequest *req) {
    auto it = events_.find(*req);
    HSA_ASSERT(it != events_.end());
    auto _ = it->second->Destroy();
    events_.erase(it);
    PushResponse(RPCType::kRPCDestroyEventResponse, NullResponseSpan());
}

void HostRequestHandler::PushResponse(idl::RPCType type,
                                      absl::Span<const char> payload) {
    tx_->Push(type, payload);
}

} // namespace ocl::hsa::enclave
