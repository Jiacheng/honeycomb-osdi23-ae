set(PINGPONG_CU_SRC ${CMAKE_CURRENT_LIST_DIR}/pingpong.cu)
set(PINGPONG_CU_DST ${CMAKE_BINARY_DIR}/experiments/data/pingpong.bin)

add_hip_bin(pingpong_codegen ${PINGPONG_CU_DST} ${PINGPONG_CU_SRC})

add_executable(pingpong_benchmark benchmark.cc)
add_dependencies(pingpong_benchmark _hip_bin_pingpong_codegen)
target_link_libraries(pingpong_benchmark hip::host absl::status gflags)
