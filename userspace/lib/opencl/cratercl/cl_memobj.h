#pragma once

#include "api.h"
#include "cl_context.h"
#include "cl_device.h"
#include "ref_counted_object.h"

#include <memory>

namespace crater::opencl {
class Memory : public CLObject<_cl_mem>, public RefCountedObject<Memory> {
  public:
    static cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags,
                                 size_t size, void *host_ptr,
                                 cl_int *errcode_ret);
    static cl_mem clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags,
                                    cl_buffer_create_type buffer_create_type,
                                    const void *buffer_create_info,
                                    cl_int *errcode_ret);
    static cl_int clRetainMemObject(cl_mem memory_obj);
    static cl_int clReleaseMemObject(cl_mem memory_obj);
    ocl::hsa::gpu_addr_t GetGPUAddress() const { return mem_; }
    ocl::hip::DeviceContext *GetCtx() const { return ctx_; }

    ~Memory();

  private:
    Memory(ocl::hsa::gpu_addr_t mem, ocl::hip::DeviceContext *ctx,
           bool is_child);
    ocl::hsa::gpu_addr_t mem_;
    ocl::hip::DeviceContext *ctx_;
    bool is_child_;
};
} // namespace crater::opencl
