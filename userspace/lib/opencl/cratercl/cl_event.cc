#include "cl_event.h"
#include "api.h"
#include <cstdio>

namespace crater::opencl {

cl_int Event::clGetEventInfo(cl_event event, cl_event_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret) {
    return CL_SUCCESS;
}

cl_int Event::clGetEventProfilingInfo(cl_event, cl_profiling_info param_name,
                                      size_t param_value_size, void *param_value,
                                      size_t *param_value_size_ret) {
    return CL_SUCCESS;
}

cl_int Event::clReleaseEvent(cl_event event) {
    auto self = static_cast<Event *>(event);
    self->Release();
    return CL_SUCCESS;
}

cl_int Event::clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
    return CL_SUCCESS;
}

cl_int Event::clEuqueueMarker(cl_command_queue command_queue, cl_event *event) {
    if (event) {
        *event = Event::New();
    }
    return CL_SUCCESS;
}

Event *Event::New() {
    return new Event();
}

} // namespace crater::opencl
