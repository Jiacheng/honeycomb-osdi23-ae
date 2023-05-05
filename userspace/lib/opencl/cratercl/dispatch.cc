#include "api.h"
#include "cl_commandqueue.h"
#include "cl_context.h"
#include "cl_device.h"
#include "cl_kernel.h"
#include "cl_memobj.h"
#include "cl_program.h"
#include "cl_event.h"

#include <cstring>

namespace crater::opencl {
const _cl_icd_dispatch kCLDispatchTable = []() {
    _cl_icd_dispatch p;
    memset(&p, 0, sizeof(p));
#ifdef CRATER_EMBEDED_LIBRARY
    p.clGetExtensionFunctionAddress = crater_clGetExtensionFunctionAddress;
#else
    p.clGetExtensionFunctionAddress = clGetExtensionFunctionAddress;
#endif
    p.clGetPlatformIDs = &Platform::clGetPlatformIDs;
    p.clGetPlatformInfo = &Platform::clGetPlatformInfo;
    p.clGetDeviceIDs = &Platform::clGetDeviceIDs;
    p.clGetDeviceInfo = &Device::clGetDeviceInfo;
    p.clRetainDevice = &Device::clRetainDevice;
    p.clReleaseDevice = &Device::clReleaseDevice;
    p.clCreateContext = &Context::clCreateContext;
    p.clCreateContextFromType = &Context::clCreateContextFromType;
    p.clGetContextInfo = &Context::clGetContextInfo;
    p.clRetainContext = &Context::clRetainContext;
    p.clReleaseContext = &Context::clReleaseContext;
    p.clCreateProgramWithSource = &Program::clCreateProgramWithSource;
    p.clCreateProgramWithBinary = &Program::clCreateProgramWithBinary;
    p.clBuildProgram = &Program::clBuildProgram;
    p.clRetainProgram = &Program::clRetainProgram;
    p.clReleaseProgram = &Program::clReleaseProgram;
    p.clGetProgramBuildInfo = &Program::clGetProgramBuildInfo;
    p.clCreateBuffer = &Memory::clCreateBuffer;
    p.clCreateSubBuffer = &Memory::clCreateSubBuffer;
    p.clRetainMemObject = &Memory::clRetainMemObject;
    p.clReleaseMemObject = &Memory::clReleaseMemObject;
    p.clCreateCommandQueue = &CommandQueue::clCreateCommandQueue;
    p.clRetainCommandQueue = &CommandQueue::clRetainCommandQueue;
    p.clReleaseCommandQueue = &CommandQueue::clReleaseCommandQueue;
    p.clEnqueueReadBuffer = &CommandQueue::clEnqueueReadBuffer;
    p.clEnqueueWriteBuffer = &CommandQueue::clEnqueueWriteBuffer;
    p.clEnqueueCopyBuffer = &CommandQueue::clEnqueueCopyBuffer;
    p.clEnqueueNDRangeKernel = &CommandQueue::clEnqueueNDRangeKernel;
    p.clFlush = &CommandQueue::clFlush;
    p.clFinish = &CommandQueue::clFinish;
    p.clCreateKernel = &Kernel::clCreateKernel;
    p.clSetKernelArg = &Kernel::clSetKernelArg;
    p.clRetainKernel = &Kernel::clRetainKernel;
    p.clReleaseKernel = &Kernel::clReleaseKernel;
    p.clEnqueueMarker = &Event::clEuqueueMarker;
    p.clGetEventInfo = &Event::clGetEventInfo;
    p.clGetEventProfilingInfo = &Event::clGetEventProfilingInfo;
    p.clReleaseEvent = &Event::clReleaseEvent;
    p.clWaitForEvents = &Event::clWaitForEvents;
    return p;
}();
} // namespace crater::opencl
