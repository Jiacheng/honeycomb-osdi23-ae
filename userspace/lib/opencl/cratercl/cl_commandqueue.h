#pragma once

#include "absl/types/span.h"
#include "api.h"
#include "cl_context.h"
#include "cl_memobj.h"
#include "ref_counted_object.h"

#include <mutex>

namespace crater::opencl {

class Kernel;
class CommandQueue : public CLObject<_cl_command_queue>,
                     public RefCountedObject<CommandQueue> {
  public:
    enum {
        kMaxPendingCommands = 16,
    };
    static cl_command_queue
    clCreateCommandQueue(cl_context context, cl_device_id device,
                         cl_command_queue_properties properties,
                         cl_int *errcode_ret);

    static cl_int clRetainCommandQueue(cl_command_queue command_queue);
    static cl_int clReleaseCommandQueue(cl_command_queue command_queue);
    static cl_int clFlush(cl_command_queue command_queue);
    static cl_int clFinish(cl_command_queue command_queue);
    static cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                                      cl_mem buffer, cl_bool blocking_read,
                                      size_t offset, size_t size, void *ptr,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event *event_wait_list,
                                      cl_event *event);

    static cl_int
    clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                         cl_bool blocking_write, size_t offset, size_t size,
                         const void *ptr, cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list, cl_event *event);

    static cl_int
    clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer,
                        cl_mem dst_buffer, size_t src_offset, size_t dst_offset,
                        size_t size, cl_uint num_events_in_wait_list,
                        const cl_event *event_wait_list, cl_event *event);

    static cl_int clEnqueueNDRangeKernel(
        cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
        const size_t *global_work_offset, const size_t *global_work_size,
        const size_t *local_work_size, cl_uint num_events_in_wait_list,
        const cl_event *event_wait_list, cl_event *event);

    ~CommandQueue();

  private:
    friend class Context;
    explicit CommandQueue(Context *ctx);
    cl_int EnqueueReadBuffer(Memory *buffer, bool blocking, size_t offset,
                             size_t size, void *ptr);
    cl_int EnqueueWriteBuffer(Memory *buffer, bool blocking, size_t offset,
                              size_t size, const void *ptr);
    cl_int EnqueueCopyBuffer(Memory *src_buffer, Memory *dst_buffer,
                             size_t src_offset, size_t dst_offset, size_t size);
    cl_int EnqueueNDRangeKernel(Kernel *kernel, cl_uint work_dim,
                                const size_t *global_work_offset,
                                const size_t *global_work_size,
                                const size_t *local_work_size);

    Context *ctx_;
    // The Command Queue APIs are supposed to be thread-safe
    std::recursive_mutex mutex_;
};

} // namespace crater::opencl
