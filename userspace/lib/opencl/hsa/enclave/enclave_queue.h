#pragma once

#include "opencl/hsa/queue.h"
#include <atomic>

namespace ocl::hsa::enclave {
class EnclaveGuestDevice;

class EnclaveGuestSDMAQueue : public SDMAQueue {
  protected:
    friend class EnclaveGuestDevice;
    friend class SDMAQueue;
    friend class EnclaveGuestDevice;
    virtual absl::Status Register() override;
    virtual absl::Status UnregisterQueue() override;
    virtual void UpdateDoorbell(uint64_t new_index, std::memory_order) override;
    explicit EnclaveGuestSDMAQueue(Device *dev);
};

class EnclaveGuestAQLQueue : public AQLQueue {
  protected:
    friend class EnclaveGuestDevice;
    friend class AQLQueue;
    // TODO: use Register instead of CreateQueue
    virtual absl::Status CreateQueue() override;
    virtual absl::Status UnregisterQueue() override;
    virtual void UpdateDoorbell(uint64_t new_index, std::memory_order) override;
    explicit EnclaveGuestAQLQueue(Device *dev);
};

} // namespace ocl::hsa::enclave
