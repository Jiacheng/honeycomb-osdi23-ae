set(RESNET_CU_SRCS
  ${CMAKE_CURRENT_LIST_DIR}/elementwise.cu
)
set(EXPERIMENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../data/resnet/binary)
set(RESNET_CU_DST ${EXPERIMENT_BINARY_DIR}/elementwise.bin)

add_hip_bin(resnet_codegen ${RESNET_CU_DST} ${RESNET_CU_SRCS})
add_ocl_bin(kernel_replacement.bin ${EXPERIMENT_BINARY_DIR}/kernel_replacement.bin "${CMAKE_CURRENT_LIST_DIR}/kernel_replacement.cl")
add_ocl_bin(resnet_gen_kernels.bin ${EXPERIMENT_BINARY_DIR}/resnet_gen_kernels.bin "${CMAKE_CURRENT_LIST_DIR}/resnet_gen_kernels.cl")

add_library(resnet_inference resnet_inference.cc resnet1.cc resnet18.cc)
add_dependencies(resnet_inference _hip_bin_resnet_codegen _ocl_target_kernel_replacement.bin
_ocl_target_resnet_gen_kernels.bin)
target_link_libraries(resnet_inference experiments hip::host)

add_executable(resnet_test resnet_test.cc)
add_test(resnet_test resnet_test)
set_tests_properties(resnet_test PROPERTIES ENVIRONMENT "GPUMPC_RESOURCE_DIR=${CMAKE_CURRENT_BINARY_DIR}/..")
target_link_libraries(resnet_test resnet_inference GTest::gtest_main)

add_executable(winograd_test winograd_test.cc)
add_test(winograd_test winograd_test)
target_link_libraries(winograd_test GTest::gtest_main)

add_executable(resnet_benchmark benchmark.cc)
target_link_libraries(resnet_benchmark resnet_inference gflags)
