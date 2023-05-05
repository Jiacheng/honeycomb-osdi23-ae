#pragma once

#include "opencl/hsa/platform.h"
#include <absl/status/status.h>

namespace ocl::hsa {

class Event;

class PlatformResource {
  public:
    friend class Platform;
    Memory *GetEventPage() { return event_page_.get(); };
    void Destroy();

    PlatformResource(const PlatformResource &) = delete;
    PlatformResource &operator=(const PlatformResource &) = delete;

    PlatformResource() = default;
    absl::Status Initialize(Device *dev);
    absl::Status InitializeEventPage(Device *dev);

    //
    // The signal BO (i.e. the event page) is allocated on the
    // host side and mapped into all devices. The host requires no accesses of
    // the memory. The GPU writes to the GPU address space (which is identical
    // to the host VA) and signals the interrupt. Therefore it requires a device
    // to initialize the BO properly.
    //
    // There is no need to explicitly free the memory. The driver frees it
    // when the process terminates.
    std::unique_ptr<Memory> event_page_;

    std::unique_ptr<Event> vm_fault_event_;
    std::unique_ptr<Event> vm_fault_signal_event_;
};

class KFDPlatform : public Platform {
  public:
    friend class Platform;
    virtual int GetKFDFD() const override { return kfd_fd_; }
    virtual const std::vector<Device *> &GetDevices() const override {
        return devices_;
    }
    virtual gpu_addr_t GetEventPageBase() override;

    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;
    virtual std::unique_ptr<Event>
    NewEvent(int type, uint64_t event_page_handle) override;

  protected:
    KFDPlatform();
    static Platform &Instance();
    absl::Status EnumerateDevices();
    virtual Device *NewDevice(unsigned node_id, unsigned gpu_id);

    bool initialized_;
    int kfd_fd_;
    std::vector<Device *> devices_;
    PlatformResource resource_;
};

} // namespace ocl::hsa