execute_process(
  COMMAND ${CMAKE_COMMAND} -E tar xf ${CMAKE_CURRENT_LIST_DIR}/resnet-data.tar.xz
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
