hunter_config(OpenSSL VERSION 1.1.1
  CONFIGURATION_TYPES Release
  CMAKE_ARGS
    ASM_SUPPORT=ON
)
hunter_config(abseil VERSION 20211102.0
  CONFIGURATION_TYPES Release
  CMAKE_ARGS
    CMAKE_CXX_STANDARD=17
    CMAKE_POSITION_INDEPENDENT_CODE=ON
)
hunter_config(spdlog VERSION 1.9.2-p0
  CONFIGURATION_TYPES Release
  CMAKE_ARGS
    CMAKE_POSITION_INDEPENDENT_CODE=ON
)
hunter_config(fmt VERSION 8.1.1
  CONFIGURATION_TYPES Release
  CMAKE_ARGS
    CMAKE_POSITION_INDEPENDENT_CODE=ON
)
