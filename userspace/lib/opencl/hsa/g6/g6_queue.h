#pragma once

#include "opencl/hsa/queue.h"

namespace ocl::hsa {
class G6Device;

class G6SDMAQueue : public SDMAQueue {
  protected:
    friend class SDMAQueue;
    friend class G6Device;
    virtual absl::Status
    RegisterQueue(struct kfd_ioctl_create_queue_args *args) override;
    explicit G6SDMAQueue(Device *dev);
};

class G6AQLQueue : public AQLQueue {
  protected:
    friend class AQLQueue;
    friend class G6Device;
    virtual absl::Status
    RegisterQueue(struct kfd_ioctl_create_queue_args *args) override;
    explicit G6AQLQueue(Device *dev);
};

} // namespace ocl::hsa