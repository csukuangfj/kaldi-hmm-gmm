include_guard(GLOBAL)

# kaldi-decoder dependency (pattern follows ../sherpa-onnx/cmake/kaldi-decoder.cmake).
#
# Resolution order:
#   1. KALDI_DECODER_SOURCE_DIR (explicit local checkout, for development),
#   2. the v0.4.0 release tarball (URL + SHA256; a pre-downloaded local copy
#      in possible_file_locations is preferred when present).
#
# NOTE: the statically linked kaldi-decoder-core is the ONLY copy of the
# decoder engine in a kaldi-hmm-gmm process; there is no runtime dependency
# on the kaldi-decoder Python package.
#
# As a subproject kaldi-decoder's own cmake/ modules resolve to THIS repo's
# ./cmake (its CMakeLists uses CMAKE_SOURCE_DIR for CMAKE_MODULE_PATH), so
# kaldifst/eigen/googletest/pybind11 are fetched exactly once, from here.
function(download_kaldi_decoder)
  if(TARGET kaldi-decoder-core)
    return()
  endif()

  set(kaldi_decoder_URL "https://github.com/k2-fsa/kaldi-decoder/archive/refs/tags/v0.4.0.tar.gz")
  set(kaldi_decoder_HASH "SHA256=078f2ebb9927c19445096549f880b4ac6db14b7eeb263062f50697873901c955")

  # If you don't have access to the Internet,
  # please pre-download kaldi-decoder-0.4.0.tar.gz
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi-decoder-0.4.0.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldi-decoder-0.4.0.tar.gz
    ${CMAKE_BINARY_DIR}/kaldi-decoder-0.4.0.tar.gz
    /tmp/kaldi-decoder-0.4.0.tar.gz
    /star-fj/fangjun/download/github/kaldi-decoder-0.4.0.tar.gz
  )
  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_decoder_URL "${f}")
      file(TO_CMAKE_PATH "${kaldi_decoder_URL}" kaldi_decoder_URL)
      message(STATUS "Found local downloaded kaldi-decoder: ${kaldi_decoder_URL}")
      break()
    endif()
  endforeach()

  set(KALDI_DECODER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(KALDI_DECODER_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

  if(KALDI_DECODER_SOURCE_DIR)
    message(STATUS "Using KALDI_DECODER_SOURCE_DIR: ${KALDI_DECODER_SOURCE_DIR}")
    add_subdirectory(${KALDI_DECODER_SOURCE_DIR}
                     ${CMAKE_BINARY_DIR}/_deps/kaldi-decoder-build
                     EXCLUDE_FROM_ALL)
    set(kaldi_decoder_SOURCE_DIR ${KALDI_DECODER_SOURCE_DIR})
  else()
    include(FetchContent)
    message(STATUS "Downloading kaldi-decoder from ${kaldi_decoder_URL}")
    FetchContent_Declare(kaldi_decoder
      URL               ${kaldi_decoder_URL}
      URL_HASH          ${kaldi_decoder_HASH}
    )
    FetchContent_MakeAvailable(kaldi_decoder)
    message(STATUS "kaldi-decoder is downloaded to ${kaldi_decoder_SOURCE_DIR}")
  endif()

  target_include_directories(kaldi-decoder-core
    INTERFACE
      ${kaldi_decoder_SOURCE_DIR}/
  )
endfunction()

download_kaldi_decoder()
