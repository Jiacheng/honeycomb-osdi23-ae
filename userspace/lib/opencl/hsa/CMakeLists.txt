include_directories(
  ../../
  ${CMAKE_SOURCE_DIR}/userspace/third_party
)

add_subdirectory(g6)
add_subdirectory(enclave)

add_library(opencl_hsa
  platform.cc
  queue.cc queue_aql.cc queue_sdma.cc
  sdma_ops.cc
  memory_manager.cc
  signals.cc
  slab_allocator.cc
  bump_allocator.cc
  ring_allocator.cc
  event.cc
  kfd_event.cc
  kfd/kfd_device.cc
  kfd/gtt_memory.cc
  kfd/kfd_platform.cc
  kfd/kfd_memory_manager.cc
  runtime_options.cc
  $<TARGET_OBJECTS:hsa_g6>
  $<TARGET_OBJECTS:hsa_enclave_common>
  $<TARGET_OBJECTS:hsa_enclave_guest>
)

target_link_libraries(opencl_hsa absl::status spdlog::spdlog fmt::fmt)
