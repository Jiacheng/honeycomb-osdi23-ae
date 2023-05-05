#pragma once

#include "absl/types/span.h"
#include "api.h"
#include "opencl/hip/device_context.h"
#include "opencl/hip/module.h"
#include "ref_counted_object.h"

#include <mutex>
#include <vector>

namespace crater::opencl {
class Program;

class Kernel : public CLObject<_cl_kernel>, public RefCountedObject<Kernel> {
  public:
    ~Kernel();
    static cl_kernel clCreateKernel(cl_program program, const char *kernel_name,
                                    cl_int *errcode_ret);
    static cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index,
                                 size_t arg_size, const void *arg_value);
    static cl_int clRetainKernel(cl_kernel kernel);
    static cl_int clReleaseKernel(cl_kernel kernel);
    hipFunction_t GetImplementation() const { return kernel_; }
    char *GetArgsBuffer() const { return args_buffer_; }
    size_t GetArgsBufferSize() const { return args_buffer_size_; }

  private:
    explicit Kernel(Program *program, hipFunction_t kernel_);
    void ComputeArgumentPosition();
    cl_int SetKernelArg(cl_uint arg_index, size_t arg_size,
                        const void *arg_value);

    Program *parent_;
    hipFunction_t kernel_;
    char *args_buffer_;
    size_t args_buffer_size_;
    std::vector<absl::Span<char>> args_;
    std::recursive_mutex mutex_;
};
} // namespace crater::opencl
