#pragma once

#include "api.h"
#include "cl_device.h"
#include "ref_counted_object.h"

#include <memory>
#include <vector>

namespace crater::opencl {

class CommandQueue;

class Context : public CLObject<_cl_context>, public RefCountedObject<Context> {
  public:
    static cl_context
    clCreateContext(const cl_context_properties *properties,
                    cl_uint num_devices, const cl_device_id *devices,
                    void(CL_CALLBACK *pfn_notify)(const char *, const void *,
                                                  size_t, void *),
                    void *user_data, cl_int *errcode_ret);
    static cl_context clCreateContextFromType(
        const cl_context_properties *properties, cl_device_type device_type,
        void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t,
                                      void *),
        void *user_data, cl_int *errcode_ret);
    static cl_int clGetContextInfo(cl_context context,
                                   cl_context_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret);

    static cl_int clRetainContext(cl_context context);
    static cl_int clReleaseContext(cl_context context);
    const std::vector<Device *> &GetDevices() const { return devices_; }
    ocl::hip::DeviceContext *GetCtx() {
        HSA_ASSERT(devices_.size() != 0);
        return devices_[0]->GetImpl();
    }
    CommandQueue *GetDefaultCommandQueue() const { return default_queue_; }

    ~Context();
  private:
    Context(std::vector<Device *> &&devices);
    cl_int GetContextInfo(cl_context_info param_name, size_t param_value_size,
                          void *param_value, size_t *param_value_size_ret);
    std::vector<Device *> devices_;
    CommandQueue *default_queue_;
};

} // namespace crater::opencl
