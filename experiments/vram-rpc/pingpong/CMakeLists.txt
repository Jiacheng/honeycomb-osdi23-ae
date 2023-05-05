set(RPC_PINGPONG_CU_SRC ${CMAKE_CURRENT_LIST_DIR}/rpc_pingpong.cu)
set(RPC_PINGPONG_CU_DST ${CMAKE_BINARY_DIR}/experiments/data/rpc_pingpong.bin)

add_hip_bin(rpc_pingpong_codegen ${RPC_PINGPONG_CU_DST} ${RPC_PINGPONG_CU_SRC})

add_executable(rpc_pingpong rpc-pingpong.cc)
add_dependencies(rpc_pingpong _hip_bin_rpc_pingpong_codegen)
target_link_libraries(rpc_pingpong gflags experiments-vram-rpc)

add_executable(rpc_pingpong_cpu rpc-pingpong-cpu.cc)
add_dependencies(rpc_pingpong_cpu _hip_bin_rpc_pingpong_codegen)
target_link_libraries(rpc_pingpong_cpu rpc gflags experiments-vram-rpc OpenSSL::Crypto)
