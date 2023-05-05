set(RPC_RESNET18_CU_SRC ${CMAKE_CURRENT_LIST_DIR}/rpc_resnet18.cu)
set(RPC_RESNET18_CU_DST ${CMAKE_BINARY_DIR}/experiments/data/rpc_resnet18.bin)

add_hip_bin(rpc_resnet18_codegen ${RPC_RESNET18_CU_DST} ${RPC_RESNET18_CU_SRC})

add_executable(rpc_resnet18 rpc_resnet18.cc)
add_dependencies(rpc_resnet18 _hip_bin_rpc_resnet18_codegen)
target_link_libraries(rpc_resnet18 resnet_inference gflags experiments-vram-rpc)

add_executable(rpc_resnet18_cpu rpc_resnet18_cpu.cc)
add_dependencies(rpc_resnet18_cpu _hip_bin_rpc_resnet18_codegen)
target_link_libraries(rpc_resnet18_cpu resnet_inference gflags experiments-vram-rpc OpenSSL::Crypto)
