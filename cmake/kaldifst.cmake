function(download_kaldifst)
  include(FetchContent)

  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.5.2.tar.gz")
  set(kaldifst_URL2 "https://huggingface.co/csukuangfj/kaldi-hmm-gmm-cmake-deps/resolve/main/kaldifst-1.5.2.tar.gz")
  set(kaldifst_HASH "SHA256=b8036431aa896bdefdba49616db21576bda04f8bc5be74de43a0c3a910828b27")

  # We will remove this branch "for-khg" to master later
  set(kaldifst_URL "https://github.com/csukuangfj/kaldifst/archive/5dfcb2614d812f92074e5c712327e75d952cfaf4.zip")
  set(kaldifst_HASH "SHA256=dbdec72a9c008c8b31a55ddd95dc4c6013a9af321bd6bc0328e0d3eca53dc893")


  # If you don't have access to the Internet,
  # please pre-download kaldi_native_io
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-5dfcb2614d812f92074e5c712327e75d952cfaf4.zip

    $ENV{HOME}/Downloads/kaldifst-1.5.2.tar.gz
    ${PROJECT_SOURCE_DIR}/kaldifst-1.5.2.tar.gz
    ${PROJECT_BINARY_DIR}/kaldifst-1.5.2.tar.gz
    /tmp/kaldifst-1.5.2.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.5.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldifst_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldifst_URL}" kaldifst_URL)
      set(kaldifst_URL2)
      break()
    endif()
  endforeach()

  set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldifst
    URL               ${kaldifst_URL}
    URL_HASH          ${kaldifst_HASH}
  )

  FetchContent_GetProperties(kaldifst)
  if(NOT kaldifst_POPULATED)
    message(STATUS "Downloading kaldifst ${kaldifst_URL}")
    FetchContent_Populate(kaldifst)
  endif()
  message(STATUS "kaldifst is downloaded to ${kaldifst_SOURCE_DIR}")
  message(STATUS "kaldifst's binary dir is ${kaldifst_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake)

  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR})

  target_include_directories(kaldifst_core
    PUBLIC
      ${kaldifst_SOURCE_DIR}/
  )

  target_include_directories(fst
    PUBLIC
      ${openfst_SOURCE_DIR}/src/include
  )

  set_target_properties(kaldifst_core PROPERTIES OUTPUT_NAME "kaldi-hmm-gmm-kaldi-fst-core")
  set_target_properties(fst PROPERTIES OUTPUT_NAME "kaldi-hmm-gmm-fst")
  set_target_properties(fstscript PROPERTIES OUTPUT_NAME "kaldi-hmm-gmm-fstscript")

  install(TARGETS kaldifst_core fst fstscript DESTINATION lib)
endfunction()

download_kaldifst()
