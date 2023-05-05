#include "api.h"
#include "cl_device.h"

extern "C" {
using namespace crater::opencl;

cl_int clIcdGetPlatformIDsKHR(cl_uint num_entries, cl_platform_id *platforms,
                              cl_uint *num_platforms) {

    if (((num_entries > 0 || !num_platforms) && !platforms) ||
        (!num_entries && platforms)) {
        return CL_INVALID_VALUE;
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

__attribute__((visibility("default"))) void *
clGetExtensionFunctionAddress(const char *func_name)

{
#define CL_EXTENSION_ENTRYPOINT_CHECK(name)                                    \
    if (!strcmp(func_name, #name))                                             \
        return reinterpret_cast<void *>(name);

    CL_EXTENSION_ENTRYPOINT_CHECK(clIcdGetPlatformIDsKHR);

    return nullptr;
#undef CL_EXTENSION_ENTRY_POINT_TEST
}
}
