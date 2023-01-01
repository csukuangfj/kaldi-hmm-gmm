cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(kaldi-hmm-gmm)

set(KHG_VERSION "0.1")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(KHG_RPATH_ORIGIN "$ORIGIN")
else()
  set(CMAKE_MACOSX_RPATH ON)
  set(KHG_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${KHG_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${KHG_RPATH_ORIGIN})

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)
option(KHG_ENABLE_TESTS "Whether to build tests" ON)
option(KHG_BUILD_PYTHON "Whether to build Python" ON)

if(KHG_BUILD_PYTHON)
  include(pybind11)
endif()

# Note: Include torch first since it may set the C++ ABI flag
include(torch)

include(kaldi-native-io)
include(kaldifst)
if(KHG_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()


add_subdirectory(kaldi-hmm-gmm)
