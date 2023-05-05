include_directories(
  ../../
  ${CMAKE_SOURCE_DIR}/userspace/third_party
)

add_executable(g6_uapi_test g6_uapi_test)
target_link_libraries(g6_uapi_test opencl_hsa GTest::gtest_main)
