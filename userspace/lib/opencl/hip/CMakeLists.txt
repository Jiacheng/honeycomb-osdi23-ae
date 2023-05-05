add_subdirectory(elf)
add_subdirectory(usm)

# common lib for both test and shared lib
add_library(amdhip64_common OBJECT
    device_context.cc
    hsa_mm.cc
    hsa_compute.cc
    module.cc
)

target_link_libraries(amdhip64_common
    PRIVATE absl::status hip::host opencl_hip_parser opencl_hsa hip_usm
)


add_library(amdhip64 SHARED
    bindings.cc
)

# The hip::host is an interface library so it will only bring in the header but not dynamically linking the .so
target_link_libraries(amdhip64 PRIVATE amdhip64_common hip_usm)

set_target_properties(amdhip64 PROPERTIES
  VERSION 5
  SOVERSION 5
  LINK_FLAGS "-Wl,--no-undefined")

add_executable(aes_device_test usm/aes_device_test.cc)
target_link_libraries(aes_device_test amdhip64_common hip_usm GTest::gtest_main)
add_test(aes_device_test aes_device_test)
set_tests_properties(aes_device_test PROPERTIES ENVIRONMENT "GPUMPC_RESOURCE_DIR=${CMAKE_BINARY_DIR}/experiments/")

add_dependencies(hip_memcpy_test amdhip64)
add_test(hip_hsa_memcpy_test ${CMAKE_BINARY_DIR}/userspace/tests/opencl/memcpy/hip_memcpy_test)
set_tests_properties(hip_hsa_memcpy_test PROPERTIES ENVIRONMENT
        "GPUMPC_RESOURCE_DIR=${CMAKE_BINARY_DIR}/experiments/;LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/")

add_test(hip_usm_memcpy_test ${CMAKE_BINARY_DIR}/userspace/tests/opencl/memcpy/hip_memcpy_test)
set_tests_properties(hip_usm_memcpy_test PROPERTIES ENVIRONMENT
        "GPUMPC_RESOURCE_DIR=${CMAKE_BINARY_DIR}/experiments/;LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/;GPUMPC_SECURE_MEMCPY=1")
