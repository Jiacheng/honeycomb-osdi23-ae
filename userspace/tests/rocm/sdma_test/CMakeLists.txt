include_directories(
  ${CMAKE_SOURCE_DIR}/userspace/third_party
)

add_library(sdma_test_base OBJECT sdma_test_base.cc)
target_link_libraries(sdma_test_base opencl_hsa)

add_executable(sdma_test sdma_test.cc)
target_link_libraries(sdma_test sdma_test_base GTest::gtest_main) 
add_test(sdma_test sdma_test)

add_executable(g6_sdma_test g6_sdma_test.cc)
target_link_libraries(g6_sdma_test sdma_test_base GTest::gtest_main)

add_executable(enclave_sdma_test enclave_sdma_test.cc)
target_link_libraries(
    enclave_sdma_test sdma_test_base GTest::gtest_main)