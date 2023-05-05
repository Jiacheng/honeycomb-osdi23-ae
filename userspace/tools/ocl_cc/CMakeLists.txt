include_directories(${CMAKE_SOURCE_DIR}/userspace/third_party)

add_executable(ocl-cc ocl_cc.cc)
target_link_libraries(ocl-cc PRIVATE OpenCL::OpenCL absl::status)

