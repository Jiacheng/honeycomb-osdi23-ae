#include "cl_context.h"
#include "api.h"
#include "cl_commandqueue.h"

#include <algorithm>

namespace crater::opencl {

Context::Context(std::vector<Device *> &&devices)
    : devices_(std::move(devices)) {
    default_queue_ = new CommandQueue(this);
}

Context::~Context() { default_queue_->Release(); }

cl_int Context::clRetainContext(cl_context context) {
    static_cast<Context *>(context)->Retain();
    return CL_SUCCESS;
}

cl_int Context::clReleaseContext(cl_context context) {
    static_cast<Context *>(context)->Release();
    return CL_SUCCESS;
}

cl_context Context::clCreateContext(
    const cl_context_properties *properties, cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data, cl_int *errcode_ret) {
    if (!num_devices || !devices) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    HSA_ASSERT(num_devices == 1 && "CLContext only support 1 device");
    std::vector<Device *> d;
    for (cl_uint i = 0; i < num_devices; ++i) {
        d.push_back(static_cast<Device *>(devices[i]));
    }

    if (errcode_ret) {
        *errcode_ret = CL_SUCCESS;
    }
    return new Context(std::move(d));
}

cl_context Context::clCreateContextFromType(
    const cl_context_properties *properties, cl_device_type device_type,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data, cl_int *errcode_ret) {
    std::vector<Device *> res;
    const auto &d = Platform::Instance().GetDevices();

    for (auto it = d.begin(), end = d.end(); it != end; ++it) {
        if (device_type == CL_DEVICE_TYPE_GPU) {
            res.push_back(*it);
        }
    }
    if (res.size() != 1) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }
    return new Context(std::move(res));
}

cl_int Context::clGetContextInfo(cl_context context, cl_context_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
    return static_cast<Context *>(context)->GetContextInfo(
        param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int Context::GetContextInfo(cl_context_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
    switch (param_name) {
    case CL_CONTEXT_REFERENCE_COUNT: {
        cl_uint count = ref_count_;
        return clGetInfo(count, param_value_size, param_value,
                         param_value_size_ret);
    }
    case CL_CONTEXT_NUM_DEVICES: {
        cl_uint num_devices = (cl_uint)devices_.size();
        return clGetInfo(num_devices, param_value_size, param_value,
                         param_value_size_ret);
    }
    case CL_CONTEXT_DEVICES: {
        auto value_size = devices_.size() * sizeof(cl_device_id *);
        if (param_value && param_value_size < value_size) {
            return CL_INVALID_VALUE;
        }
        if (param_value_size_ret) {
            *param_value_size_ret = value_size;
        }
        if (param_value) {
            cl_device_id *device_list = (cl_device_id *)param_value;
            for (const auto &it : devices_) {
                *device_list++ = it;
            }
        }
        return CL_SUCCESS;
    }
    default:
        break;
    }

    return CL_INVALID_VALUE;
}
} // namespace crater::opencl
