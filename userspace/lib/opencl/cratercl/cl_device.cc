#include "cl_device.h"
#include "api.h"
#include <CL/cl_ext.h>

#include <fcntl.h>
#include <unistd.h>

typedef union {
    struct {
        cl_uint type;
        cl_uint data[5];
    } raw;
    struct {
        cl_uint type;
        cl_char unused[17];
        cl_char bus;
        cl_char device;
        cl_char function;
    } pcie;
} cl_device_topology_amd;

#define CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD 1

namespace crater::opencl {
static const cl_uint kAddressBits = 64;
static const cl_bool kDeviceAvailable = true;
static const cl_bool kCompilerAvailable = false;
static const cl_device_fp_config kDoubleFPConfig =
    CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO |
    CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM;
static const cl_bool kEndianLittle = true;
static const cl_bool kErrorCorrectionSupport = false;
static const cl_device_exec_capabilities kExecutionCapabilities =
    CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
static const cl_ulong kGlobalMemCacheSize = 16384;
static const cl_device_mem_cache_type kGlobalMemCacheType = CL_READ_WRITE_CACHE;
static const cl_uint kGlobalMemCachelineSize = 64;
//   static const cl_device_fp_config kHalfFPConfig = 0;
static const cl_bool kImageSupport = false;
static const size_t kImage2DMaxHeight = 0;
static const size_t kImage2DMaxWidth = 0;
static const size_t kImage3DMaxDepth = 0;
static const size_t kImage3DMaxHeight = 0;
static const size_t kImage3DMaxWidth = 0;
static const cl_ulong kLocalMemSize = 65536;
static const cl_device_local_mem_type kLocalMemType = CL_LOCAL;
static const cl_uint kMaxClockFrequency = 1590;
static const cl_uint kMaxConstantArgs = 8;
static const cl_ulong kMaxConstantBufferSize = 6732ul * 1024 * 1024;
static const size_t kMaxParameterSize = 1024;
static const cl_uint kMaxReadImageArgs = 0;
static const cl_uint kMaxSamplers = 0;
static const size_t kMaxWorkGroupSize = 1024;
static const cl_uint kMaxWorkItemDimensions = 3;
static const cl_uint kMaxWriteImageArgs = 0;
static const cl_uint kMemBaseAddrAlign = 2048;
static const cl_uint kMinDataTypeAlignSize = 128;
static const cl_uint kPreferredVectorWidthChar = 4;
static const cl_uint kPreferredVectorWidthShort = 2;
static const cl_uint kPreferredVectorWidthInt = 1;
static const cl_uint kPreferredVectorWidthLong = 1;
static const cl_uint kPreferredVectorWidthFloat = 1;
static const cl_uint kPreferredVectorWidthDouble = 1;
static const size_t kProfilingTimerResolution = 1;
static const cl_command_queue_properties kQueueProperties =
    CL_QUEUE_PROFILING_ENABLE;
static const cl_device_fp_config kSingleFPConfig =
    CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
static const cl_uint kVendorID = 0x1002;
static const cl_bool kHostUnifiedMemory = false;

const char Device::kExtensions[] =
    "cl_khr_fp64 cl_khr_global_int32_base_atomics "
    "cl_khr_global_int32_extended_atomics cl_khr_local_int32_base_atomics "
    "cl_khr_local_int32_extended_atomics";

const char Device::kDeviceName[] = "gfx900";
const char Device::kDeviceProfile[] = "EMBEDDED_PROFILE";
const char Device::kVendor[] = "Advanced Micro Devices, Inc.";
const size_t Device::kMaxWorkItemSizes[] = {1024, 1024, 1024};
const char Device::kDeviceVersion[] = "OpenCL 1.2";
const char Device::kDriverVersion[] = "Crater 0.1";
const char Device::kDeviceOpenCLCVersion[] = "OpenCL 1.2 AMD";

Device::Device(ocl::hip::DeviceContext *impl) : impl_(impl) {}

cl_int Device::GetInfo(cl_device_info param_name, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret) {

    // FIXME: They are pre-determined values from VEGA
#define CASE(param_name, field_name)                                           \
    case param_name:                                                           \
        return clGetInfo(field_name, param_value_size, param_value,            \
                         param_value_size_ret);

    switch (param_name) {
    case CL_DEVICE_TYPE: {
        cl_device_type device_type = CL_DEVICE_TYPE_GPU;
        return clGetInfo(device_type, param_value_size, param_value,
                         param_value_size_ret);
    }
    case CL_DEVICE_PLATFORM: {
        cl_platform_id plat = &Platform::Instance();
        return clGetInfo(plat, param_value_size, param_value,
                         param_value_size_ret);
    }
    case CL_DEVICE_TOPOLOGY_AMD: {
        auto id = 44; // FIXME
        cl_device_topology_amd r;
        r.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
        r.pcie.bus = id >> 8;
        r.pcie.device = (id >> 3) & 0x1f;
        r.pcie.function = id & 0x7;
        return clGetInfo(r, param_value_size, param_value,
                         param_value_size_ret);
    }
    case CL_DEVICE_MAX_COMPUTE_UNITS: {
        cl_int max_compute_units = 40; // FIXME
        return clGetInfo(max_compute_units, param_value_size, param_value,
                         param_value_size_ret);
    }

    case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
    case CL_DEVICE_GLOBAL_MEM_SIZE: {
        // 12G
        cl_ulong total_heap_size = 12 * 1024 * 1024 * 1024UL;
        return clGetInfo<cl_ulong>(total_heap_size, param_value_size,
                                   param_value, param_value_size_ret);
    }

    case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD: {
        // cl_ulong r = info.usable_heap_size - info.used_heap_size;
        //  12G
        cl_ulong r = 12 * 1024 * 1024 * 1024UL;
        return clGetInfo(r, param_value_size, param_value,
                         param_value_size_ret);
    }

    case CL_DEVICE_NAME: {
        auto r = "gfx1030";
        return clGetInfo(r, param_value_size, param_value,
                         param_value_size_ret);
    }

        CASE(CL_DEVICE_ADDRESS_BITS, kAddressBits);
        CASE(CL_DEVICE_AVAILABLE, kDeviceAvailable);
        CASE(CL_DEVICE_COMPILER_AVAILABLE, kCompilerAvailable);
        CASE(CL_DEVICE_DOUBLE_FP_CONFIG, kDoubleFPConfig);
        CASE(CL_DEVICE_ENDIAN_LITTLE, kEndianLittle);
        CASE(CL_DEVICE_ERROR_CORRECTION_SUPPORT, kErrorCorrectionSupport);
        CASE(CL_DEVICE_EXECUTION_CAPABILITIES, kExecutionCapabilities);
        CASE(CL_DEVICE_EXTENSIONS, kExtensions);
        CASE(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, kGlobalMemCacheSize);
        CASE(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, kGlobalMemCacheType);
        CASE(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, kGlobalMemCachelineSize);
        // CASE(CL_DEVICE_HALF_FP_CONFIG, kHalfFPConfig);
        CASE(CL_DEVICE_IMAGE_SUPPORT, kImageSupport);
        CASE(CL_DEVICE_IMAGE2D_MAX_HEIGHT, kImage2DMaxHeight);
        CASE(CL_DEVICE_IMAGE2D_MAX_WIDTH, kImage2DMaxWidth);
        CASE(CL_DEVICE_IMAGE3D_MAX_DEPTH, kImage3DMaxDepth);
        CASE(CL_DEVICE_IMAGE3D_MAX_HEIGHT, kImage3DMaxHeight);
        CASE(CL_DEVICE_IMAGE3D_MAX_WIDTH, kImage3DMaxWidth);
        CASE(CL_DEVICE_LOCAL_MEM_SIZE, kLocalMemSize);
        CASE(CL_DEVICE_LOCAL_MEM_TYPE, kLocalMemType);
        CASE(CL_DEVICE_MAX_CLOCK_FREQUENCY, kMaxClockFrequency);
        CASE(CL_DEVICE_MAX_CONSTANT_ARGS, kMaxConstantArgs);
        CASE(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, kMaxConstantBufferSize);
        CASE(CL_DEVICE_MAX_PARAMETER_SIZE, kMaxParameterSize);
        CASE(CL_DEVICE_MAX_READ_IMAGE_ARGS, kMaxReadImageArgs);
        CASE(CL_DEVICE_MAX_SAMPLERS, kMaxSamplers);
        CASE(CL_DEVICE_MAX_WORK_GROUP_SIZE, kMaxWorkGroupSize);
        CASE(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, kMaxWorkItemDimensions);
        CASE(CL_DEVICE_MAX_WORK_ITEM_SIZES, kMaxWorkItemSizes);
        CASE(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, kMaxWriteImageArgs);
        CASE(CL_DEVICE_MEM_BASE_ADDR_ALIGN, kMemBaseAddrAlign);
        CASE(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, kMinDataTypeAlignSize);
        CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, kPreferredVectorWidthChar);
        CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
             kPreferredVectorWidthShort);
        CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, kPreferredVectorWidthInt);
        CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, kPreferredVectorWidthLong);
        CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
             kPreferredVectorWidthFloat);
        CASE(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
             kPreferredVectorWidthDouble);
        CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, kPreferredVectorWidthChar);
        CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, kPreferredVectorWidthShort);
        CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, kPreferredVectorWidthInt);
        CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, kPreferredVectorWidthLong);
        CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, kPreferredVectorWidthFloat);
        CASE(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, kPreferredVectorWidthDouble);
        CASE(CL_DEVICE_PROFILE, kDeviceProfile);
        CASE(CL_DEVICE_PROFILING_TIMER_RESOLUTION, kProfilingTimerResolution);
        CASE(CL_DEVICE_QUEUE_PROPERTIES, kQueueProperties);
        CASE(CL_DEVICE_SINGLE_FP_CONFIG, kSingleFPConfig);
        CASE(CL_DEVICE_VENDOR, kVendor);
        CASE(CL_DEVICE_VENDOR_ID, kVendorID);
        CASE(CL_DEVICE_VERSION, kDeviceVersion);
        CASE(CL_DRIVER_VERSION, kDriverVersion);
        CASE(CL_DEVICE_HOST_UNIFIED_MEMORY, kHostUnifiedMemory);
        CASE(CL_DEVICE_OPENCL_C_VERSION, kDeviceOpenCLCVersion);

    default:
        break;
    }
#undef CASE

    return CL_INVALID_VALUE;
}

cl_int Device::clRetainDevice(cl_device_id device) { return CL_SUCCESS; }
cl_int Device::clReleaseDevice(cl_device_id device) { return CL_SUCCESS; }

cl_int Device::clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
    if (!device) {
        return CL_INVALID_DEVICE;
    }

    auto impl = static_cast<crater::opencl::Device *>(device);
    return impl->GetInfo(param_name, param_value_size, param_value,
                         param_value_size_ret);
}

} // namespace crater::opencl
