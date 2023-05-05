set(RUNTIME_GPU_SRC ${CMAKE_CURRENT_LIST_DIR}/channel.cu)
set(RUNTIME_GPU_DST ${CMAKE_CURRENT_BINARY_DIR}/gpu_runtime.bin)

add_hip_bin(runtime_gpu_code_gen ${RUNTIME_GPU_DST} "${RUNTIME_GPU_SRC}")

add_custom_target(runtime_gpu ALL)
add_dependencies(runtime_gpu _hip_bin_runtime_gpu_code_gen)

add_executable(dh_test dh_test.cc)
target_link_libraries(dh_test gflags GTest::gtest hip::host absl::status)