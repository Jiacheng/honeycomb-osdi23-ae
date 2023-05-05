add_library(rpc hip_shared_buffer.cc)
target_link_libraries(rpc hip::host absl::status)