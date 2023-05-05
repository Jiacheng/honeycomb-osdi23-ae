include_directories(
  ${CMAKE_SOURCE_DIR}/userspace/third_party
)

add_executable(host-agent
  main.cc
  host_agent.cc
  host_environment.cc
  $<TARGET_OBJECTS:hsa_enclave_common>
)
target_link_libraries(host-agent opencl_hsa gflags)
