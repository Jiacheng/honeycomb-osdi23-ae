#include "event.h"

namespace ocl::hsa {

Event::Event(unsigned event_id, unsigned long hw_data2, unsigned long hw_data3)
    : event_id_(event_id), hw_data2_(hw_data2), hw_data3_(hw_data3) {}

} // namespace ocl::hsa