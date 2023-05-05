add_library(hsa_g6 OBJECT
  page_table.cc
  g6_queue_aql.cc
  g6_queue_sdma.cc
  g6_platform.cc
  g6_device.cc
  g6_platform.cc
  g6_memory.cc
  g6_memory_manager.cc
)

target_link_libraries(hsa_g6 absl::status spdlog::spdlog fmt::fmt)
