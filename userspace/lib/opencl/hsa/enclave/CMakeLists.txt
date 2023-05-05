add_library(hsa_enclave_common OBJECT
  transmit_buffer.cc
)

target_link_libraries(hsa_enclave_common absl::status)

add_library(hsa_enclave_guest OBJECT
  guest_device.cc
  guest_memory_manager.cc
  guest_platform.cc
  guest_memory.cc
  guest_rpc_client.cc
  guest_event.cc
  enclave_queue_sdma.cc
  enclave_queue_aql.cc
)

target_link_libraries(hsa_enclave_guest spdlog::spdlog fmt::fmt)
