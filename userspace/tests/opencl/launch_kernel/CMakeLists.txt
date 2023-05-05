add_executable(hip_launch_kernel_test launch_kernel_test.cc)
target_link_libraries(hip_launch_kernel_test hip::host GTest::gtest absl::status)

add_hip_bin(fill.bin ${CMAKE_CURRENT_BINARY_DIR}/fill.bin ${CMAKE_CURRENT_LIST_DIR}/fill.cu)
add_dependencies(hip_launch_kernel_test _hip_bin_fill.bin)

add_test(
  NAME hip_launch_kernel_test
  COMMAND hip_launch_kernel_test fill.bin)

