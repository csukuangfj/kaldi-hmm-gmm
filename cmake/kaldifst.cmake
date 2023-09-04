function(download_kaldifst)
  include(FetchContent)

  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.7.0.tar.gz")
  set(kaldifst_URL2 "https://huggingface.co/csukuangfj/kaldi-hmm-gmm-cmake-deps/resolve/main/kaldifst-1.7.0.tar.gz")
  set(kaldifst_HASH "SHA256=d5f4adbf7634e8cea57da00981e9f6424e777ad0396ab8f6f52baac0ceffb11b")

  set(kaldifst_URL "https://github.com/csukuangfj/kaldifst/archive/f330e9f86610bf5ad35f02541ae6c635d9e788c0.zip")
  set(kaldifst_URL2 "")
  set(kaldifst_HASH "")

  # If you don't have access to the Internet,
  # please pre-download kaldi_native_io
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-f330e9f86610bf5ad35f02541ae6c635d9e788c0.zip

    $ENV{HOME}/Downloads/kaldifst-1.7.0.tar.gz
    ${PROJECT_SOURCE_DIR}/kaldifst-1.7.0.tar.gz
    ${PROJECT_BINARY_DIR}/kaldifst-1.7.0.tar.gz
    /tmp/kaldifst-1.7.0.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.7.0.tar.gz
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
  set_target_properties(fstscript PROPERTIES OUTPUT_NAME "kaldi-hmm-gmm-fst-script")

  if(KHG_BUILD_PYTHON AND WIN32)
    install(TARGETS kaldifst_core fst fstscript DESTINATION ..)
  else()
    install(TARGETS kaldifst_core fst fstscript DESTINATION lib)
  endif()

endfunction()

download_kaldifst()
