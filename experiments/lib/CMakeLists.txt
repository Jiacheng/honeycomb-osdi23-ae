include_directories(${CMAKE_SOURCE_DIR}/userspace/lib)

add_library(experiments
  experiments/linux/linux_platform.cc
)
target_link_libraries(experiments absl::status crypto)

add_library(experiments-vram-rpc
  experiments/vram-rpc/gpu_direct_dispatch.cc
  experiments/vram-rpc/cpu_dispatch.cc
)
target_link_libraries(experiments-vram-rpc rpc absl::status hip::host)
