#include "cl_memobj.h"

namespace crater::opencl {
Memory::Memory(ocl::hsa::gpu_addr_t mem, ocl::hip::DeviceContext *ctx,
               bool is_child)
    : mem_(mem), ctx_(ctx), is_child_(is_child) {}

Memory::~Memory() {
    if (!is_child_) {
        auto _ = ctx_->GetMemoryManager()->hipFree(mem_);
        (void)_;
    }
}

cl_mem Memory::clCreateBuffer(cl_context context, cl_mem_flags flags,
                              size_t size, void *host_ptr,
                              cl_int *errcode_ret) {
    auto ctx = static_cast<Context *>(context);
    HSA_ASSERT(((flags & CL_MEM_USE_HOST_PTR) == 0) && "DMA not supported");
    HSA_ASSERT(((flags & CL_MEM_ALLOC_HOST_PTR) == 0) && "GTT not supported");
    auto hip = ctx->GetCtx();
    ocl::hsa::gpu_addr_t mem;
    auto error = hip->GetMemoryManager()->hipMalloc(&mem, size);
    if (errcode_ret) {
        *errcode_ret =
            (error == hipSuccess) ? CL_SUCCESS : CL_INVALID_BUFFER_SIZE;
    }
    auto ptr = new Memory(mem, hip, false);
    if ((flags & CL_MEM_COPY_HOST_PTR) != 0 && host_ptr != NULL) {
        auto queue = ctx->GetDefaultCommandQueue();
        auto ret = clEnqueueWriteBuffer((cl_command_queue)queue, ptr, CL_TRUE,
                                        0, size, host_ptr, 0, NULL, NULL);
        Nullable(errcode_ret).Assign(ret);
    }
    return ptr;
}

//
// TODO: Implement nested sub buffer and refcounting for the parent
//
cl_mem Memory::clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags,
                                 cl_buffer_create_type buffer_create_type,
                                 const void *buffer_create_info,
                                 cl_int *errcode_ret) {
    auto parent = static_cast<Memory *>(buffer);
    HSA_ASSERT(((flags & CL_MEM_USE_HOST_PTR) == 0) && "DMA not supported");
    HSA_ASSERT(((flags & CL_MEM_ALLOC_HOST_PTR) == 0) && "GTT not supported");
    if (buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }
    auto info = reinterpret_cast<const _cl_buffer_region *>(buffer_create_info);
    auto m = parent->GetGPUAddress();
    if (!m) {
        Nullable(errcode_ret).Assign(CL_INVALID_VALUE);
        return nullptr;
    }
    auto r = m + info->origin;
    if (!r) {
        Nullable(errcode_ret).Assign(CL_INVALID_VALUE);
        return nullptr;
    }
    Nullable(errcode_ret).Assign(CL_SUCCESS);
    return new Memory(r, parent->GetCtx(), true);
}

cl_int Memory::clRetainMemObject(cl_mem memory_obj) {
    auto self = static_cast<Memory *>(memory_obj);
    self->Retain();
    return CL_SUCCESS;
}

cl_int Memory::clReleaseMemObject(cl_mem memory_obj) {
    auto self = static_cast<Memory *>(memory_obj);
    self->Release();
    return CL_SUCCESS;
}

} // namespace crater::opencl
