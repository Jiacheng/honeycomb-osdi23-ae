include_directories(${CMAKE_SOURCE_DIR}/userspace/third_party)

add_library(compute_test_base OBJECT compute_test_base.cc)
target_link_libraries(compute_test_base opencl_hsa)

add_executable(compute_test compute_test.cc)
target_link_libraries(compute_test compute_test_base GTest::gtest)

add_executable(enclave_compute_test enclave_compute_test.cc)
target_link_libraries(enclave_compute_test compute_test_base GTest::gtest)
