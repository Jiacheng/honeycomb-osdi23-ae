#pragma once

#ifdef __HIP_DEVICE_COMPILE__
#define GPUMPC_HOST_AND_DEVICE __device__ __host__
#define GPUMPC_CONSTANT __constant__
#else
#define GPUMPC_HOST_AND_DEVICE
#define GPUMPC_CONSTANT
#endif
