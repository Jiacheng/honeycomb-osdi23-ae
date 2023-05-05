cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HUNTER_PACKAGES OpenCL gflags GTest abseil fmt spdlog OpenSSL)
set(HUNTER_FILEPATH_CONFIG ${CMAKE_SOURCE_DIR}/cmake/hunter_config.cmake)

include(FetchContent)
FetchContent_Declare(SetupHunter GIT_REPOSITORY https://github.com/cpp-pm/gate)
FetchContent_MakeAvailable(SetupHunter)

project(gpumpc-userspace LANGUAGES CXX C ASM)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(BuildUtils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -fno-strict-aliasing")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -fno-strict-aliasing")

find_package(OpenCL CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

# For experiments and the HIP userspace library
find_package(hip REQUIRED)

enable_testing()

include_directories(lib)
add_subdirectory(lib)
add_subdirectory(userspace)
add_subdirectory(experiments)
