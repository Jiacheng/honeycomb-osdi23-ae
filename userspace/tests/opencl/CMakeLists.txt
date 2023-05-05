include_directories(${CMAKE_SOURCE_DIR}/userspace/third_party)

add_library(opencl_test_utils opencl_test_utils.cc)
target_link_libraries(opencl_test_utils GTest::gtest OpenCL::OpenCL absl::status)

include_directories(.)
add_subdirectory(memcpy)
add_subdirectory(launch_kernel)
add_subdirectory(ring_allocator)
