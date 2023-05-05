add_executable(hip_memcpy_test memcpy_test.cc)
target_link_libraries(hip_memcpy_test GTest::gtest_main hip::host)
add_test(hip_memcpy_test hip_memcpy_test)
