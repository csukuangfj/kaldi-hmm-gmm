function(download_kaldifst)
  include(FetchContent)

  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.7.10.tar.gz")
  set(kaldifst_URL2 "https://huggingface.co/csukuangfj/kaldi-hmm-gmm-cmake-deps/resolve/main/kaldifst-1.7.10.tar.gz")
  set(kaldifst_HASH "SHA256=7f7b3173a6584a6b1987f65ae7af2ac453d66b845f875a9d31074b8d2cd0de54")

  # If you don't have access to the Internet,
  # please pre-download kaldifst
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-1.7.10.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldifst-1.7.10.tar.gz
    ${CMAKE_BINARY_DIR}/kaldifst-1.7.10.tar.gz
    /tmp/kaldifst-1.7.10.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.7.10.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldifst_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldifst_URL}" kaldifst_URL)
      message(STATUS "Found local downloaded kaldifst: ${kaldifst_URL}")
      set(kaldifst_URL2)
      break()
    endif()
  endforeach()

  set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  # NOTE: v1.9.0+ provides the zero-copy interop helpers
  # (_vector_fst_addr/_borrow/_take) and a matching openfst.  Keep in sync with
  # the pip requirement `kaldifst>=1.9.0` in setup.py.
  FetchContent_Declare(kaldifst
    URL               https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.9.0.tar.gz
    URL_HASH          SHA256=225d8377d1d262e3aedab23558fead106b4cbc83462fb29f1c5e7d9e3b0c14dd
  )

  # kaldifst's CMakeLists needs its own cmake/ dir on CMAKE_MODULE_PATH at
  # configure time (include(openfst)), so populate first, then add_subdirectory.
  FetchContent_Populate(kaldifst)
  message(STATUS "kaldifst is downloaded to ${kaldifst_SOURCE_DIR}")
  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake)
  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR})


  target_include_directories(kaldifst_core
    PUBLIC
      ${kaldifst_SOURCE_DIR}/
  )

  set_target_properties(kaldifst_core PROPERTIES OUTPUT_NAME "kaldi-hmm-gmm-kaldi-fst-core")

  if(KHG_INSTALL_CORE_LIB)
    install(TARGETS kaldifst_core fst DESTINATION lib)
  endif()

endfunction()

download_kaldifst()
