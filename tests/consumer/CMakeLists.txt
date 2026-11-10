# Minimal consumer project (sherpa-onnx-style usage spike): link both
# kaldi-hmm-gmm-core and kaldi-decoder-core from one C++ executable, with
# no Python involved.  Build:
#   cmake -S . -B build && cmake --build build && ./build/consumer_test
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(khg-consumer-spike CXX)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(KHG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(KHG_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KALDI_DECODER_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_DECODER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

# Consume kaldi-hmm-gmm exactly like sherpa-onnx will: add_subdirectory on a
# local checkout (its cmake/kaldi-decoder.cmake pulls in the decoder).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../..
                 ${CMAKE_BINARY_DIR}/kaldi-hmm-gmm-build
                 EXCLUDE_FROM_ALL)

add_executable(consumer_test main.cc)
target_link_libraries(consumer_test kaldi-hmm-gmm-core kaldi-decoder-core)
