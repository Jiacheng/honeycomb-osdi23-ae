#pragma once

#include "api.h"
#include "opencl/hip/device_context.h"
#include "opencl/hsa/assert.h"

#include <memory>
#include <string>
#include <vector>

namespace crater::opencl {
class Platform;
class Device : public CLObject<_cl_device_id> {
  public:
    friend class Platform;
    Platform *GetPlatform() const;
    cl_int GetInfo(cl_device_info param_name, size_t param_value_size,
                   void *param_value, size_t *param_value_size_ret);
    static cl_int clRetainDevice(cl_device_id device);
    static cl_int clReleaseDevice(cl_device_id device);
    static cl_int clGetDeviceInfo(cl_device_id device,
                                  cl_device_info param_name,
                                  size_t param_value_size, void *param_value,
                                  size_t *param_value_size_ret);
    ocl::hip::DeviceContext *GetImpl() { return impl_; }

  private:
    explicit Device(ocl::hip::DeviceContext *impl);
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    ocl::hip::DeviceContext *impl_;
    static const char kExtensions[];
    static const char kDeviceName[];
    static const char kDeviceProfile[];
    static const size_t kMaxWorkItemSizes[];
    static const char kVendor[];
    static const char kDeviceVersion[];
    static const char kDriverVersion[];
    static const char kDeviceOpenCLCVersion[];
};

class Platform : public CLObject<_cl_platform_id> {
  public:
    static Platform &Instance();
    const std::vector<Device *> &GetDevices() const { return devices_; }

    static cl_int clGetPlatformIDs(cl_uint num_entries,
                                   cl_platform_id *platforms,
                                   cl_uint *num_platforms);
    static cl_int clGetPlatformInfo(cl_platform_id platform,
                                    cl_platform_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret);
    static cl_int clGetDeviceIDs(cl_platform_id platform,
                                 cl_device_type device_type,
                                 cl_uint num_entries, cl_device_id *devices,
                                 cl_uint *num_devices);

    ~Platform();

  private:
    Platform();
    void Initialize();
    void EnumerateGPUs();
    std::vector<Device *> devices_;
};

} // namespace crater::opencl
