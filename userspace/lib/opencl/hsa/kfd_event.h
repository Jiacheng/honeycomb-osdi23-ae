#pragma once

#include "event.h"
#include "opencl/hsa/g6/g6_platform.h"
#include "opencl/hsa/kfd/kfd_memory.h"
#include "opencl/hsa/platform.h"

namespace ocl::hsa {

namespace enclave {
class HostRequestHandler;
}

class Device;
class KFDPlatform;

// Wrapper for the ioctl interface of KFD events
class KFDEvent : public Event {
  public:
    virtual ~KFDEvent();
    virtual absl::Status Destroy() override;
    virtual absl::Status Wait(unsigned long ms) override;
    virtual absl::Status Notify() override;

  private:
    friend class KFDPlatform;
    friend class G6Platform;
    friend class enclave::HostRequestHandler;
    explicit KFDEvent(unsigned event_id, unsigned long hw_data2,
                      unsigned long hw_data3);
    static std::unique_ptr<Event> New(int type, uint64_t event_page_handle);
    static std::unique_ptr<Event> NewSignalEvent();
};
} // namespace ocl::hsa