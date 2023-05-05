#pragma once

#include "api.h"
#include "cl_context.h"

namespace crater::opencl {

class Program : public CLObject<_cl_program>, public RefCountedObject<Program> {
  public:
    static cl_program clCreateProgramWithSource(cl_context context,
                                                cl_uint count,
                                                const char **strings,
                                                const size_t *lengths,
                                                cl_int *errcode_ret);

    static cl_program clCreateProgramWithBinary(cl_context context,
                                                cl_uint num_devices,
                                                const cl_device_id *device_list,
                                                const size_t *lengths,
                                                const unsigned char **binaries,
                                                cl_int *binary_status,
                                                cl_int *errcode_ret);

    static cl_int clBuildProgram(
        cl_program program, cl_uint num_devices,
        const cl_device_id *device_list, const char *options,
        void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
        void *user_data);
    static cl_int clRetainProgram(cl_program program);
    static cl_int clReleaseProgram(cl_program program);
    static cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device,
                                        cl_program_build_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret);

    ~Program();
    hipModule_t GetImpl() const { return module_; }
    Context *GetContext() const { return ctx_; }

  private:
    explicit Program(Context *ctx, hipModule_t module);
    Context *ctx_;
    hipModule_t module_;
};
} // namespace crater::opencl
