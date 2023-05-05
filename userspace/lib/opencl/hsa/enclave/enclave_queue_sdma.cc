#include "enclave_queue.h"
#include "guest_platform.h"
#include "guest_rpc_client.h"
#include "opencl/hsa/enclave/idl.h"
#include "opencl/hsa/queue.h"
#include <absl/status/status.h>
#include <atomic>

namespace ocl::hsa::enclave {

EnclaveGuestSDMAQueue::EnclaveGuestSDMAQueue(Device *dev) : SDMAQueue(dev) {}

absl::Status EnclaveGuestSDMAQueue::Register() {
    auto rpc = static_cast<EnclaveGuestDevice *>(dev_)->GetRPCClient();
    auto q = dispatch_region_->GetBuffer();
    idl::CreateQueueRequest req = {
        .ring_buffer_base = reinterpret_cast<uintptr_t>(ring_->GetBuffer()),
        .ring_buffer_size = ring_->GetSize(),
        .dispatch_base = reinterpret_cast<uintptr_t>(q),
        .type = idl::QueueType::kQueueTypeSDMA,
        .eop_buffer_address = 0,
        .eop_buffer_size = 0,
        .ctx_save_restore_address = 0,
        .ctx_save_restore_size = 0,
        .ctl_stack_size = 0};
    auto resp = rpc->CreateQueue(req);
    queue_id_ = resp.queue_id;
    doorbell_offset_ = resp.doorbell_offset;
    return absl::OkStatus();
}

absl::Status EnclaveGuestSDMAQueue::UnregisterQueue() {
    if (queue_id_ >= 0) {
        auto rpc = static_cast<EnclaveGuestDevice *>(dev_)->GetRPCClient();
        rpc->DestroyQueue(queue_id_);
        queue_id_ = kInvalidQueueID;
    }
    return absl::OkStatus();
}

void EnclaveGuestSDMAQueue::UpdateDoorbell(uint64_t new_index,
                                           std::memory_order) {
    auto rpc = static_cast<EnclaveGuestDevice *>(dev_)->GetRPCClient();
    rpc->UpdateDoorbell(doorbell_offset_, new_index);
}

} // namespace ocl::hsa::enclave
