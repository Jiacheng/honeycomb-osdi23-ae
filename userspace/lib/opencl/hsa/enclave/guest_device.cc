#include "enclave_queue.h"
#include "guest_platform.h"
#include "guest_rpc_client.h"

#include "opencl/hsa/assert.h"
#include "opencl/hsa/kfd/kfd_device.h"
#include "opencl/hsa/queue.h"
#include <absl/status/status.h>
#include <memory>

namespace ocl::hsa::enclave {

EnclaveGuestDevice::EnclaveGuestDevice(unsigned node_id, unsigned gpu_id,
                                       TransmitBuffer &&tx, TransmitBuffer &&rx)
    : node_id_(node_id), gpu_id_(gpu_id), tx_(std::move(tx)),
      rx_(std::move(rx)) {}

const Device::Properties &EnclaveGuestDevice::GetProperties() const {
    return KFDDevice::GetHardCodedProperties();
}

std::unique_ptr<DeviceQueue> EnclaveGuestDevice::CreateSDMAQueue() {
    return std::unique_ptr<DeviceQueue>(new EnclaveGuestSDMAQueue(this));
}

std::unique_ptr<DeviceQueue> EnclaveGuestDevice::CreateAQLQueue() {
    return std::unique_ptr<DeviceQueue>(new EnclaveGuestAQLQueue(this));
}

absl::Status EnclaveGuestDevice::Initialize() {
    client_.reset(new GuestRPCClient(this));
    return absl::OkStatus();
}

void EnclaveGuestDevice::NotifyHostAgent() {
    auto stat =
        static_cast<EnclaveGuestPlatform &>(EnclaveGuestPlatform::Instance())
            .NotifyHostAgent();
    HSA_ASSERT(stat.ok());
}

absl::Status EnclaveGuestDevice::Close() { return absl::OkStatus(); }

} // namespace ocl::hsa::enclave
