#pragma once

#include "api.h"
#include "ref_counted_object.h"

namespace crater::opencl {

class Event : public CLObject<_cl_event>, public RefCountedObject<Event> {
  public:
    ~Event() = default;
    static cl_int clGetEventInfo(cl_event event, cl_event_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret);
    static cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info param_name,
                                          size_t param_value_size, void *param_value,
                                          size_t *param_value_size_ret);
    static cl_int clReleaseEvent(cl_event event);
    static cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list);
    static cl_int clEuqueueMarker(cl_command_queue command_queue, cl_event *event);
    static Event *New();

  private:
    Event() = default;
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;

};

} // namespace crater::opencl
