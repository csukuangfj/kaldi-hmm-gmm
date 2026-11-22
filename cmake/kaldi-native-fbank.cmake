include_guard(GLOBAL)

# kaldi-native-fbank dependency (pattern follows kaldi-decoder.cmake in this
# directory and ../sherpa-onnx/cmake/kaldi-native-fbank.cmake).
#
# Resolution order:
#   1. KALDI_NATIVE_FBANK_SOURCE_DIR (explicit local checkout, for development),
#   2. the v1.23.3 release tarball (URL + SHA256; a pre-downloaded local copy
#      in possible_file_locations is preferred when present).
#
# kaldi-native-fbank provides the feature pipeline for the i-vector extractor
# (OnlineCmvn etc.; see docs/source/ivector.rst).
function(download_kaldi_native_fbank)
  if(TARGET kaldi-native-fbank-core)
    return()
  endif()

  set(kaldi_native_fbank_URL  "https://github.com/csukuangfj/kaldi-native-fbank/archive/refs/tags/v1.23.3.tar.gz")
  set(kaldi_native_fbank_HASH "SHA256=1f14b426da8701b6e14acf6c72e7517f42710d7239dc2868e6610b8fc46ae4b0")

  set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

  # If you don't have access to the Internet,
  # please pre-download kaldi-native-fbank-1.23.3.tar.gz
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-native-fbank-1.23.3.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-native-fbank-1.23.3.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-native-fbank-1.23.3.tar.gz
    /tmp/kaldi-native-fbank-1.23.3.tar.gz
    /star-fj/fangjun/download/github/kaldi-native-fbank-1.23.3.tar.gz
  )
  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_fbank_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_fbank_URL}" kaldi_native_fbank_URL)
      message(STATUS "Found local downloaded kaldi-native-fbank: ${kaldi_native_fbank_URL}")
      break()
    endif()
  endforeach()

  if(BUILD_SHARED_LIBS)
    set(_build_shared_libs_bak ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
  endif()

  set(_cmake_warn_deprecated_bak "${CMAKE_WARN_DEPRECATED}")
  set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

  if(KALDI_NATIVE_FBANK_SOURCE_DIR)
    message(STATUS "Using KALDI_NATIVE_FBANK_SOURCE_DIR: ${KALDI_NATIVE_FBANK_SOURCE_DIR}")
    add_subdirectory(${KALDI_NATIVE_FBANK_SOURCE_DIR}
                     ${CMAKE_BINARY_DIR}/_deps/kaldi_native_fbank-build
                     EXCLUDE_FROM_ALL)
    set(kaldi_native_fbank_SOURCE_DIR ${KALDI_NATIVE_FBANK_SOURCE_DIR})
    set(kaldi_native_fbank_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/kaldi_native_fbank-build)
  else()
    include(FetchContent)
    FetchContent_Declare(kaldi_native_fbank
      URL
        ${kaldi_native_fbank_URL}
      URL_HASH          ${kaldi_native_fbank_HASH}
    )
    message(STATUS "Downloading kaldi-native-fbank from ${kaldi_native_fbank_URL}")
    FetchContent_MakeAvailable(kaldi_native_fbank)
    message(STATUS "kaldi-native-fbank is downloaded to ${kaldi_native_fbank_SOURCE_DIR}")
    message(STATUS "kaldi-native-fbank's binary dir is ${kaldi_native_fbank_BINARY_DIR}")
  endif()

  set(CMAKE_WARN_DEPRECATED "${_cmake_warn_deprecated_bak}" CACHE BOOL "" FORCE)

  if(TARGET kissfft AND (CMAKE_C_COMPILER_ID MATCHES "Clang"))
    target_compile_options(kissfft PRIVATE -Wno-cast-align)
  endif()

  if(_build_shared_libs_bak)
    set_target_properties(kaldi-native-fbank-core
      PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    set(BUILD_SHARED_LIBS ON)
  endif()

  target_include_directories(kaldi-native-fbank-core
    INTERFACE
      ${kaldi_native_fbank_SOURCE_DIR}/
  )

  if(NOT BUILD_SHARED_LIBS)
    install(TARGETS kaldi-native-fbank-core kissfft DESTINATION lib)
  endif()
endfunction()

download_kaldi_native_fbank()
