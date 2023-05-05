add_library(opencl_hip_parser STATIC
  amdgpu_program.cc
  msgpack.cc
  parser.cc
  llvm_elf_parser.cc
  code_object_v3_metadata_parser.cc
)
target_link_libraries(opencl_hip_parser absl::status)

add_executable(msgpack_test msgpack_test.cc)
target_link_libraries(msgpack_test opencl_hip_parser GTest::gtest_main)
add_test(msgpack_test msgpack_test)

add_executable(parser_test parser_test.cc)
target_link_libraries(parser_test opencl_hip_parser GTest::gtest_main)
add_dependencies(parser_test _hip_bin_fill.bin)
add_test(
  NAME parser_test
  COMMAND parser_test ${CMAKE_BINARY_DIR}/userspace/tests/opencl/launch_kernel/fill.bin
)
