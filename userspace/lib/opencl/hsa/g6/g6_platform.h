#pragma once

#include "opencl/hsa/kfd/kfd_platform.h"
#include "opencl/hsa/platform.h"

namespace ocl::hsa {

class G6Platform : public KFDPlatform {
  public:
    virtual gpu_addr_t GetEventPageBase() override;

    //
    // Initialize the platform. Doorbells / signals works
    // only after the user flushes the page table of the GPU.
    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;
    virtual std::unique_ptr<Event>
    NewEvent(int type, uint64_t event_page_handle) override;

  protected:
    friend class Platform;
    static Platform &Instance();
    explicit G6Platform();

    // Temporary hack
    virtual Device *NewDevice(unsigned node_id, unsigned gpu_id) override;

    absl::Status InitializeSignalBO(Device *dev);
    gpu_addr_t signal_va_;
    std::unique_ptr<Memory> signal_bo_;
};

} // namespace ocl::hsa