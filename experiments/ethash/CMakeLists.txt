#add_hip_bin(ethash_gfx1030.bin ${CMAKE_CURRENT_BINARY_DIR}/../data/ethash/ethash_gfx1030.bin ${CMAKE_CURRENT_LIST_DIR}/ethash.cu)
add_oclcc_bin(ethash_gfx1030.bin ${CMAKE_CURRENT_LIST_DIR}/ethash.cl)

add_library(ethash OBJECT
  dag_cache_manager_base.cc
  dag_cache_manager.cc
  ethash_solver.cc
)
target_link_libraries(ethash hip::host absl::status)
add_dependencies(ethash _ocl_target_ethash_gfx1030.bin)

add_executable(dag_cache_manager_test dag_cache_manager_test.cc)
target_link_libraries(dag_cache_manager_test ethash experiments GTest::gtest_main)
add_test(dag_cache_manager_test dag_cache_manager_test)

add_executable(ethash_test ethash_test.cc)
target_link_libraries(ethash_test ethash experiments GTest::gtest_main)
add_test(ethash_test ethash_test)
set_tests_properties(ethash_test PROPERTIES ENVIRONMENT "GPUMPC_RESOURCE_DIR=${CMAKE_CURRENT_BINARY_DIR}/..")

