#include "host_agent.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/enclave/idl.h"
#include "opencl/hsa/kfd/kfd_memory.h"
#include "opencl/hsa/queue.h"
#include "opencl/hsa/utils.h"
#include "utils/align.h"

#include <absl/status/status.h>
#include <absl/types/span.h>
#include <algorithm>
#include <atomic>
#include <fcntl.h>
#include <memory>

#include <hsa/kfd_ioctl.h>
#include <sched.h>

namespace ocl::hsa::enclave {

enum { kPageSize = 4096 };
struct PageMapEntry {
    uint64_t pfn : 55;
    uint64_t reserved : 8;
    uint8_t present : 1;
};

static_assert(sizeof(PageMapEntry) == 8, "");

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
    : dev_(dev), tx_(tx), rx_(rx), page_map_fd_(-1) {}

absl::Status HostRequestHandler::Initialize() {
    rx_->GetRptr()->store(0);
    rx_->GetWptr()->store(0);
    tx_->GetRptr()->store(0);
    tx_->GetWptr()->store(0);
    page_map_fd_ = open("/proc/self/pagemap", O_RDONLY);
    if (page_map_fd_ < 0) {
        return absl::InvalidArgumentError("Cannot read page map");
    }
    return absl::OkStatus();
}

void HostRequestHandler::ProcessRequests() {
    auto buf = rx_->GetBuffer();
    auto rptr = rx_->GetRptr()->load();
    auto wptr = rx_->GetWptr()->load();
    auto size = buf.size();

    while (rptr == wptr) {
        sched_yield();
        wptr = rx_->GetWptr()->load();
    }

    std::vector<char> tmp(TransmitBuffer::kMaxRPCSize);
    while (rptr != wptr) {
        RPCType ty;
        size_t payload_size;
        auto pkt =
            rx_->ReadPacketAt(rptr, &ty, &payload_size, absl::MakeSpan(tmp));
        HSA_ASSERT(pkt);
        Dispatch(ty, pkt, payload_size);
        rptr = (rptr + sizeof(RPCType) + payload_size) & (size - 1);
    }
    rx_->GetRptr()->store(rptr);
}

void HostRequestHandler::Dispatch(RPCType ty, char *req, size_t payload_size) {
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
    HSA_ASSERT(!req->request_pfn ||
               (req->flag & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR));
    std::unique_ptr<KFDMemory> m(
        new KFDMemory(reinterpret_cast<void *>(req->va_addr), req->size));
    auto stat =
        m->AllocGPUMemory(dev_->GetGPUID(), req->flag, req->mmap_offset);
    HSA_ASSERT(stat.ok());
    m->SetResourceScopeFlag(KFDMemory::kUnmanagedMmap);
    unsigned long handle = m->GetHandle();
    mem_.insert(std::make_pair<unsigned long, std::unique_ptr<KFDMemory>>(
        std::move(handle), std::move(m)));

    std::vector<char> response;
    size_t num_pages = req->request_pfn
                           ? gpumpc::AlignUp(req->size, kPageSize) / kPageSize
                           : 0;

    response.resize(sizeof(AllocateGPUMemoryResponseHeader) +
                    num_pages * sizeof(uint64_t));

    auto header =
        reinterpret_cast<AllocateGPUMemoryResponseHeader *>(response.data());
    header->handle = handle;
    header->num_pages = num_pages;
    uintptr_t *pfn = reinterpret_cast<uintptr_t *>(header + 1);
    for (size_t i = 0; i < num_pages; ++i) {
        pfn[i] = GetPhysicalPageFrameNumber(req->va_addr + i * kPageSize);
    }

    PushResponse(RPCType::kRPCAllocateGPUMemoryResponse, response);
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
    unsigned size = (unsigned)payload.size();
    tx_->Push(type | (size << 8), payload);
}

uintptr_t HostRequestHandler::GetPhysicalPageFrameNumber(uintptr_t va_addr) {
    uintptr_t vpfn = reinterpret_cast<uintptr_t>(va_addr) / kPageSize;
    PageMapEntry entry;
    int ret = pread64(page_map_fd_, &entry, sizeof(entry),
                      vpfn * sizeof(PageMapEntry));
    if (ret != sizeof(entry)) {
        return 0;
    }
    uintptr_t pfn = entry.pfn;
    auto physical_addr = pfn;
    return entry.present ? physical_addr : 0;
}

} // namespace ocl::hsa::enclave
