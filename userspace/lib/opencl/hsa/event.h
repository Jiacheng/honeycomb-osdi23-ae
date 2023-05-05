#pragma once

#include <absl/status/status.h>

namespace ocl::hsa {

// Event provides a synchronization primitive to implement GPU / host barriers.
class Event {
  public:
    virtual ~Event() = default;
    virtual absl::Status Destroy() = 0;
    Event &operator=(const Event &) = delete;
    Event(const Event &) = delete;
    unsigned GetEventID() const { return event_id_; }
    unsigned long GetEventMailboxPtr() const { return hw_data2_; }
    unsigned long GetHWData3() const { return hw_data3_; }
    virtual absl::Status Wait(unsigned long ms) = 0;
    virtual absl::Status Notify() = 0;

  protected:
    explicit Event(unsigned event_id, unsigned long hw_data2,
                   unsigned long hw_data3);
    unsigned event_id_;
    unsigned long hw_data2_;
    unsigned long hw_data3_;
};
} // namespace ocl::hsa