set(AES_CU_SRC ${CMAKE_CURRENT_LIST_DIR}/aes_ctr.cu)
set(AES_CU_DST ${CMAKE_BINARY_DIR}/experiments/data/aes_kernel.bin)

add_hip_bin(aes_codegen ${AES_CU_DST} ${AES_CU_SRC})

find_package(OpenSSL REQUIRED)
add_library(hip_usm OBJECT
  memcpy.cc
  aes_device.cc
  aes_buffer.cc
)
target_link_libraries(hip_usm hip::host absl::status OpenSSL::Crypto)
add_dependencies(hip_usm _hip_bin_aes_codegen)
