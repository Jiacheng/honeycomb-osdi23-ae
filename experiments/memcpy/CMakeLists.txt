add_executable(memcpy_benchmark benchmark.cc)
target_link_libraries(memcpy_benchmark hip::host gflags)
