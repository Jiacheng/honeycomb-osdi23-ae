#include "api.h"
#include "cl_device.h"

namespace crater::opencl {

static const char kAMDPlatformName[] =
    "AMD Accelerated Parallel Processing (GPU-MPC)";

Platform &Platform::Instance() {
    static Platform inst;
    return inst;
}

Platform::Platform() { Initialize(); }
Platform::~Platform() {
    // DO NOT delete the device yet as the destructor of the OpenCL library
    // might come after
}

void Platform::Initialize() { EnumerateGPUs(); }
void Platform::EnumerateGPUs() {
    auto ctx = ocl::hip::GetCurrentDeviceContext();
    devices_.emplace_back(new Device(ctx));
}

cl_int Platform::clGetPlatformIDs(cl_uint num_entries,
                                  cl_platform_id *platforms,
                                  cl_uint *num_platforms) {
    if (((num_entries > 0 || !num_platforms) && !platforms) ||
        (!num_entries && platforms)) {
        return CL_INVALID_VALUE;
    }
    if (!ocl::hip::GetCurrentDeviceContext()) {
        return CL_INVALID_PLATFORM;
    }

    if (num_platforms && !platforms) {
        *num_platforms = 1;
        return CL_SUCCESS;
    }

    *platforms = &Platform::Instance();
    if (num_platforms) {
        *num_platforms = 1;
    }

    return CL_SUCCESS;
}

cl_int Platform::clGetPlatformInfo(cl_platform_id platform,
                                   cl_platform_info param_name,
                                   size_t param_value_size, void *param_value,
                                   size_t *param_value_size_ret) {
    if (platform && platform != &Platform::Instance()) {
        return CL_INVALID_PLATFORM;
    }

    const char *value = nullptr;
    switch (param_name) {
    case CL_PLATFORM_PROFILE:
        value = "EMBEDDED_PROFILE";
        break;
    case CL_PLATFORM_VERSION:
        value = "OpenCL 1.2 AMD";
        break;
    case CL_PLATFORM_NAME:
        value = kAMDPlatformName;
        break;
    case CL_PLATFORM_VENDOR:
        value = "Advanced Micro Devices, Inc.";
        break;
    case CL_PLATFORM_EXTENSIONS:
        value = "cl_khr_icd";
        break;
    case CL_PLATFORM_ICD_SUFFIX_KHR:
        value = "AMD";
        break;
    default:
        break;
    }

    if (value) {
        return clGetInfo(value, param_value_size, param_value,
                         param_value_size_ret);
    }

    return CL_INVALID_VALUE;
}

cl_int Platform::clGetDeviceIDs(cl_platform_id platform,
                                cl_device_type device_type, cl_uint num_entries,
                                cl_device_id *devices, cl_uint *num_devices) {
    auto plat = &Platform::Instance();
    if (platform && platform != plat) {
        return CL_INVALID_PLATFORM;
    }

    if (((num_entries > 0 || !num_devices) && !devices) ||
        (!num_entries && devices)) {
        return CL_INVALID_VALUE;
    }

    // Get all available devices
    // FIXME: handle flags
    auto n = std::min<cl_uint>(num_entries, (cl_uint)plat->GetDevices().size());
    for (size_t i = 0; i < n; ++i) {
        devices[i] = plat->GetDevices()[i];
    }
    if (num_devices) {
        *num_devices = (cl_uint)plat->GetDevices().size();
    }

    return CL_SUCCESS;
}

} // namespace crater::opencl
