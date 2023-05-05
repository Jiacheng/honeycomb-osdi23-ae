#include "guest_platform.h"
#include "guest_rpc_client.h"

#include "opencl/hsa/assert.h"
#include "opencl/hsa/event.h"
#include <absl/status/status.h>
#include <memory>

namespace ocl::hsa::enclave {

class EnclaveGuestEvent : public Event {
  public:
    virtual absl::Status Destroy() override;
    virtual absl::Status Wait(unsigned long ms) override;
    virtual absl::Status Notify() override {
        HSA_ASSERT(0 && "Unimplemented");
        return absl::OkStatus();
    }

    explicit EnclaveGuestEvent(EnclaveGuestDevice *dev, unsigned event_id,
                               unsigned long hw_data2, unsigned long hw_data3);
    ~EnclaveGuestEvent();

  private:
    EnclaveGuestDevice *dev_;
};

std::unique_ptr<Event> NewEnclaveGuestEvent(EnclaveGuestDevice *dev, int type,
                                            uint64_t event_page_handle) {
    auto rpc = dev->GetRPCClient();
    auto resp = rpc->CreateEvent(type, event_page_handle);
    return std::unique_ptr<Event>(new EnclaveGuestEvent(
        dev, resp.event_id, resp.mailbox_ptr, resp.hw_data3));
}

absl::Status EnclaveGuestEvent::Destroy() {
    if (!event_id_) {
        return absl::OkStatus();
    }

    auto rpc = dev_->GetRPCClient();
    rpc->DestroyEvent(event_id_);
    event_id_ = 0;
    return absl::OkStatus();
}

absl::Status EnclaveGuestEvent::Wait(unsigned long ms) {
    auto rpc = dev_->GetRPCClient();
    rpc->WaitEvent(event_id_, ms);
    return absl::OkStatus();
}

EnclaveGuestEvent::EnclaveGuestEvent(EnclaveGuestDevice *dev, unsigned event_id,
                                     unsigned long hw_data2,
                                     unsigned long hw_data3)
    : Event(event_id, hw_data2, hw_data3), dev_(dev) {}

EnclaveGuestEvent::~EnclaveGuestEvent() { auto _ = Destroy(); }

} // namespace ocl::hsa::enclave