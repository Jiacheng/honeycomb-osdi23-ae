set(HELLO_WORLD_CU_SRCS ${CMAKE_CURRENT_LIST_DIR}/hello-world.cu)
set(HELLO_WORLD_CU_DST ${CMAKE_CURRENT_BINARY_DIR}/hello-world.bin)

add_hip_bin(hello_world_codegen ${HELLO_WORLD_CU_DST} ${HELLO_WORLD_CU_SRCS})

add_executable(hello_world main.cc)
add_dependencies(hello_world _hip_bin_hello_world_codegen)
target_link_libraries(hello_world hip::host gflags absl::status)