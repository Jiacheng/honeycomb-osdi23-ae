# The experiment requires Clang 14 to compile our own custom code
# It requires <hip/hip_runtime.h> to provide the definitions of threadIdx.x / blockIdx.x
# TODO: Write a FindClang.cmake / FindROCM.cmake to do things properly 
find_program(HIP_DEVICE_CXX_EXECUTABLE clang
  HINTS /opt/rocm/llvm/bin
  DOC "The clang Compiler for HIP code" 
  REQUIRED
)

function(add_oclcc_bin output srcs)
  list(APPEND OCL_FLAGS -O3)

  get_property(additional_definitions SOURCE ${srcs} PROPERTY COMPILE_DEFINITIONS)
  list(APPEND OCL_FLAGS ${additional_definitions})

  get_directory_property(cmake_include_directories INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES cmake_include_directories)
  set(ocl_include_dirs)
  foreach(it ${cmake_include_directories})
    list(APPEND ocl_include_dirs "-I${it}")
  endforeach()
  list(APPEND ocl_include_dirs "-I${CMAKE_CURRENT_LIST_DIR}")

  add_custom_command(
    OUTPUT ${output}
    DEPENDS ${srcs} ocl-cc
    IMPLICIT_DEPENDS CXX ${srcs}
    COMMAND
    $<TARGET_FILE:ocl-cc> ${srcs} ${OCL_FLAGS} ${ocl_include_dirs} -o ${output}
  )
  add_custom_target(_ocl_target_${output} DEPENDS ${output})
endfunction()

function(add_ocl_bin target output srcs)
  list(APPEND OCL_FLAGS -O3)

  get_property(additional_definitions SOURCE ${srcs} PROPERTY COMPILE_DEFINITIONS)
  list(APPEND OCL_FLAGS ${additional_definitions})

  get_directory_property(cmake_include_directories INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES cmake_include_directories)
  set(ocl_include_dirs)
  foreach(it ${cmake_include_directories})
    list(APPEND ocl_include_dirs "-I${it}")
  endforeach()
  list(APPEND ocl_include_dirs "-I${CMAKE_CURRENT_LIST_DIR}")

  add_custom_command(
    OUTPUT ${output}
    DEPENDS ${srcs} ocl-cc
    IMPLICIT_DEPENDS CXX ${srcs}
    COMMAND
    $<TARGET_FILE:ocl-cc> ${srcs} ${OCL_FLAGS} ${ocl_include_dirs} -o ${output}
  )
  add_custom_target(_ocl_target_${target} DEPENDS ${output})
endfunction()

function(add_hip_bin target output srcs)
  list(APPEND CFLAGS -x hip --cuda-device-only --no-gpu-bundle-output -O2 --offload-arch=gfx1030)

  get_property(additional_definitions SOURCE ${srcs} PROPERTY COMPILE_DEFINITIONS)
  list(APPEND CFLAGS ${additional_definitions})

  get_directory_property(cmake_include_directories INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES cmake_include_directories)
  set(include_dirs)
  foreach(it ${cmake_include_directories})
    list(APPEND include_dirs "-I${it}")
  endforeach()

  add_custom_command(
    OUTPUT ${output}
    DEPENDS ${srcs}
    IMPLICIT_DEPENDS CXX ${srcs}
    COMMAND ${HIP_DEVICE_CXX_EXECUTABLE} ${CFLAGS} ${include_dirs} -c ${srcs} -o ${output}
  )
  add_custom_target(_hip_bin_${target} DEPENDS ${output})
endfunction()
