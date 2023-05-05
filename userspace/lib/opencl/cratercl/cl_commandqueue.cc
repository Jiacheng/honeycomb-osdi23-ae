#include "cl_commandqueue.h"
#include "api.h"
#include "cl_kernel.h"
#include "cl_event.h"

#include "opencl/hsa/runtime_options.h"

namespace crater::opencl {
CommandQueue::CommandQueue(Context *ctx) : ctx_(ctx) {}

CommandQueue::~CommandQueue() { ctx_->Release(); }

cl_command_queue
CommandQueue::clCreateCommandQueue(cl_context context, cl_device_id device,
                                   cl_command_queue_properties properties,
                                   cl_int *errcode_ret) {
    auto ctx = static_cast<Context *>(context);
    auto dev = static_cast<Device *>(device);
    if (!ctx->GetDevices().size() || ctx->GetDevices().front() != dev) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }
    ctx->Retain();
    if (errcode_ret) {
        *errcode_ret = CL_SUCCESS;
    }
    return new CommandQueue(ctx);
}

cl_int CommandQueue::clRetainCommandQueue(cl_command_queue command_queue) {
    auto self = static_cast<CommandQueue *>(command_queue);
    std::lock_guard<std::recursive_mutex> lk(self->mutex_);
    self->Retain();
    return CL_SUCCESS;
}

cl_int CommandQueue::clReleaseCommandQueue(cl_command_queue command_queue) {
    auto self = static_cast<CommandQueue *>(command_queue);
    std::lock_guard<std::recursive_mutex> lk(self->mutex_);
    self->Release();
    return CL_SUCCESS;
}

cl_int CommandQueue::clFlush(cl_command_queue command_queue) {
    return CL_SUCCESS;
}

cl_int CommandQueue::clFinish(cl_command_queue command_queue) {
    auto self = static_cast<CommandQueue *>(command_queue);
    std::lock_guard<std::recursive_mutex> lk(self->mutex_);
    return CL_SUCCESS;
}

cl_int CommandQueue::clEnqueueReadBuffer(cl_command_queue command_queue,
                                         cl_mem buffer, cl_bool blocking_read,
                                         size_t offset, size_t size, void *ptr,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event *event_wait_list,
                                         cl_event *event) {
    auto self = static_cast<CommandQueue *>(command_queue);
    auto mem = static_cast<Memory *>(buffer);
    HSA_ASSERT(num_events_in_wait_list == 0);
    if (event) {
        *event = Event::New();
    }
    return self->EnqueueReadBuffer(mem, blocking_read, offset, size, ptr);
}

cl_int CommandQueue::clEnqueueWriteBuffer(cl_command_queue command_queue,
                                          cl_mem buffer, cl_bool blocking_write,
                                          size_t offset, size_t size,
                                          const void *ptr,
                                          cl_uint num_events_in_wait_list,
                                          const cl_event *event_wait_list,
                                          cl_event *event) {
    auto self = static_cast<CommandQueue *>(command_queue);
    auto mem = static_cast<Memory *>(buffer);
    HSA_ASSERT(num_events_in_wait_list == 0);
    if (event) {
        *event = Event::New();
    }
    return self->EnqueueWriteBuffer(mem, blocking_write, offset, size, ptr);
}

cl_int CommandQueue::clEnqueueCopyBuffer(cl_command_queue command_queue,
                                         cl_mem src_buffer, cl_mem dst_buffer,
                                         size_t src_offset, size_t dst_offset,
                                         size_t size,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event *event_wait_list,
                                         cl_event *event) {
    auto self = static_cast<CommandQueue *>(command_queue);
    auto src_mem = static_cast<Memory *>(src_buffer);
    auto dst_mem = static_cast<Memory *>(dst_buffer);
    HSA_ASSERT(num_events_in_wait_list == 0);
    if (event) {
        *event = Event::New();
    }
    return self->EnqueueCopyBuffer(src_mem, dst_mem, src_offset, dst_offset,
                                   size);
}

cl_int CommandQueue::clEnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {

    // Unimplemented
    if (num_events_in_wait_list) {
        return CL_INVALID_VALUE;
    }

    auto self = static_cast<CommandQueue *>(command_queue);
    auto cl_kernel = static_cast<Kernel *>(kernel);
    if (event) {
        *event = Event::New();
    }
    return self->EnqueueNDRangeKernel(cl_kernel, work_dim, global_work_offset,
                                      global_work_size, local_work_size);
}

static inline bool IsSecureMemoryEnabled() {
    auto opt = ocl::GetRuntimeOptions();
    return opt && opt->IsSecureMemcpy();
}

cl_int CommandQueue::EnqueueReadBuffer(Memory *buffer, bool blocking,
                                       size_t offset, size_t size, void *ptr) {
    std::lock_guard<std::recursive_mutex> lk(mutex_);
    auto hip = ctx_->GetCtx();
    if (IsSecureMemoryEnabled()) {
        auto stat = hip->GetSecureMemcpy()->MemcpyDtoH(
            ptr, buffer->GetGPUAddress() + offset, size);
        return stat.ok() ? CL_SUCCESS : CL_INVALID_VALUE;
    } else {
        auto error = hip->GetMemoryManager()->hipMemcpyDtoH(
            ptr, buffer->GetGPUAddress() + offset, size);
        return error == hipSuccess ? CL_SUCCESS : CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

cl_int CommandQueue::EnqueueWriteBuffer(Memory *buffer, bool blocking,
                                        size_t offset, size_t size,
                                        const void *ptr) {
    std::lock_guard<std::recursive_mutex> lk(mutex_);
    auto hip = ctx_->GetCtx();
    if (IsSecureMemoryEnabled()) {
        auto stat = hip->GetSecureMemcpy()->MemcpyHtoD(
            buffer->GetGPUAddress() + offset, const_cast<void *>(ptr), size);
        return stat.ok() ? CL_SUCCESS : CL_INVALID_VALUE;
    } else {
        auto error = hip->GetMemoryManager()->hipMemcpyHtoD(
            buffer->GetGPUAddress() + offset, ptr, size);
        return error == hipSuccess ? CL_SUCCESS : CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

cl_int CommandQueue::EnqueueCopyBuffer(Memory *src_buffer, Memory *dst_buffer,
                                       size_t src_offset, size_t dst_offset,
                                       size_t size) {
    std::lock_guard<std::recursive_mutex> lk(mutex_);
    auto hip = ctx_->GetCtx();
    auto error = hip->GetMemoryManager()->hipMemcpyDtoD(
        dst_buffer->GetGPUAddress() + dst_offset,
        src_buffer->GetGPUAddress() + src_offset, size);
    return error == hipSuccess ? CL_SUCCESS : CL_INVALID_VALUE;
}

cl_int CommandQueue::EnqueueNDRangeKernel(Kernel *kernel, cl_uint work_dim,
                                          const size_t *global_work_offset,
                                          const size_t *global_work_size,
                                          const size_t *local_work_size) {
    std::lock_guard<std::recursive_mutex> lk(mutex_);
    if (work_dim > 3 || !work_dim) {
        return CL_INVALID_WORK_DIMENSION;
    }

    if (global_work_offset) {
        // NOT SUPPORTED
        return CL_INVALID_WORK_DIMENSION;
    }

    auto hip = ctx_->GetCtx();
    auto error = hip->GetComputeContext()->hipModuleLaunchKernel(
        kernel->GetImplementation(),
        (0 < work_dim && global_work_size) ? global_work_size[0] : 1,
        (1 < work_dim && global_work_size) ? global_work_size[1] : 1,
        (2 < work_dim && global_work_size) ? global_work_size[2] : 1,
        (0 < work_dim && local_work_size) ? local_work_size[0] : 1,
        (1 < work_dim && local_work_size) ? local_work_size[1] : 1,
        (2 < work_dim && local_work_size) ? local_work_size[2] : 1,
        kernel->GetArgsBuffer(), kernel->GetArgsBufferSize());
    return error == hipSuccess ? CL_SUCCESS : CL_INVALID_VALUE;
}

} // namespace crater::opencl
