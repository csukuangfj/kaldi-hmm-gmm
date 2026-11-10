if (CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
# Let normal variables shadow option() in dependency subprojects (e.g. the
# BUILD_SHARED_LIBS dance in cmake/kaldi-native-io.cmake).
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()
# kaldifst.cmake calls FetchContent_Populate directly (it must set
# CMAKE_MODULE_PATH between populate and add_subdirectory); allow it
# without the CMP0169 deprecation warning.  CMAKE_POLICY_DEFAULT_CMP0169
# covers the same pattern inside fetched subprojects (kaldifst's
# cmake/openfst.cmake), whose own cmake_minimum_required() would reset the
# policy to WARN.
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)
project(kaldi-hmm-gmm)

# Dependency builds (eigen/googletest/kaldi_native_io) still use old
# cmake_minimum_required / cmake_policy forms; silence their deprecation
# warnings like kaldi-native-io.cmake does locally.
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

set(KHG_VERSION "1.1.4")

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

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)
option(KHG_ENABLE_TESTS "Whether to build tests" ON)
option(KHG_BUILD_PYTHON "Whether to build Python" ON)
option(KHG_INSTALL_CORE_LIB "Whether to install the C++ libraries" ON)

# Static libraries are linked into the Python extension module; they must be
# position independent. (Same approach as sherpa-onnx.)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(BUILD_SHARED_LIBS AND MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(MSVC)
  # windows.h defines min/max macros that break std::min/std::max in the
  # vendored headers (e.g. openfst's randgen.h).
  add_compile_definitions(NOMINMAX)
endif()

if(NOT BUILD_SHARED_LIBS AND MSVC)
  # see https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html
  # https://stackoverflow.com/questions/14172856/compile-with-mt-instead-of-md-using-cmake
  if(MSVC)
      add_compile_options(
          $<$<CONFIG:>:/MT> #---------|
          $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
          $<$<CONFIG:Release>:/MT> #--|
      )
  endif()
endif()

if(KHG_BUILD_PYTHON)
  include(pybind11)
endif()

include(eigen)

include(kaldi-native-io)
include(kaldifst)
include(kaldi-decoder)
if(KHG_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()

add_subdirectory(kaldi-hmm-gmm)
