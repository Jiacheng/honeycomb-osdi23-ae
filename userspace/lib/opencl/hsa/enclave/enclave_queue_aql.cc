#include "enclave_queue.h"
#include "guest_platform.h"
#include "guest_rpc_client.h"
#include "opencl/hsa/enclave/idl.h"
#include "opencl/hsa/queue.h"
#include <absl/status/status.h>
#include <atomic>

#include <hsa/hsa.h>
#include <hsa/hsakmttypes.h>
#include <hsa/kfd_ioctl.h>

namespace ocl::hsa::enclave {

EnclaveGuestAQLQueue::EnclaveGuestAQLQueue(Device *dev) : AQLQueue(dev) {}

absl::Status EnclaveGuestAQLQueue::CreateQueue() {
    struct kfd_ioctl_create_queue_args args = {0};
    auto stat = AllocateCtxBuffer(&args);
    if (!stat.ok()) {
        return stat;
    }

    auto rpc = static_cast<EnclaveGuestDevice *>(dev_)->GetRPCClient();
    auto q = dispatch_region_->GetBuffer();
    idl::CreateQueueRequest req = {
        .ring_buffer_base = reinterpret_cast<uintptr_t>(ring_->GetBuffer()),
        .ring_buffer_size = ring_->GetSize(),
        .dispatch_base = reinterpret_cast<uintptr_t>(q),
        .type = idl::QueueType::kQueueTypeAQL,
        .eop_buffer_address = args.eop_buffer_address,
        .eop_buffer_size = args.eop_buffer_size,
        .ctx_save_restore_address = args.ctx_save_restore_address,
        .ctx_save_restore_size = args.ctx_save_restore_size,
        .ctl_stack_size = args.ctl_stack_size};
    auto resp = rpc->CreateQueue(req);
    queue_id_ = resp.queue_id;
    doorbell_offset_ = resp.doorbell_offset;
    return absl::OkStatus();
}

absl::Status EnclaveGuestAQLQueue::UnregisterQueue() {
    if (queue_id_ >= 0) {
        auto rpc = static_cast<EnclaveGuestDevice *>(dev_)->GetRPCClient();
        rpc->DestroyQueue(queue_id_);
        queue_id_ = kInvalidQueueID;
    }
    return absl::OkStatus();
}

void EnclaveGuestAQLQueue::UpdateDoorbell(uint64_t new_index,
                                          std::memory_order) {
    auto rpc = static_cast<EnclaveGuestDevice *>(dev_)->GetRPCClient();
    rpc->UpdateDoorbell(doorbell_offset_, new_index);
}

} // namespace ocl::hsa::enclave
