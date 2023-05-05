add_library(cratercl SHARED
  dispatch.cc
  cl_platform.cc
  cl_device.cc
  cl_context.cc
  cl_icd.cc
  cl_memobj.cc
  cl_program.cc
  cl_kernel.cc
  cl_commandqueue.cc
  cl_event.cc
)

get_target_property(OCL_INCLUDES OpenCL::OpenCL INTERFACE_INCLUDE_DIRECTORIES)
target_compile_definitions(cratercl PRIVATE -DCL_TARGET_OPENCL_VERSION=120)
target_include_directories(cratercl PRIVATE ${OCL_INCLUDES})
target_link_libraries(
    cratercl PRIVATE amdhip64_common hip_usm absl::status absl::span absl::strings hip::host crypto)