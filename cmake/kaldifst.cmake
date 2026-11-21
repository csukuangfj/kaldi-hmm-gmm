include_guard(GLOBAL)

function(download_kaldifst)
  include(FetchContent)

  set(kaldifst_URL  "https://github.com/k2-fsa/kaldifst/archive/refs/tags/v1.9.2.tar.gz")
  set(kaldifst_HASH "SHA256=8007cfde794f6c5e287d0e5e22753c1cf3503c9b02512fbcd6dc5d75e5b41fa2")

  # If you don't have access to the Internet,
  # please pre-download kaldifst
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldifst-1.9.2.tar.gz
    ${CMAKE_SOURCE_DIR}/kaldifst-1.9.2.tar.gz
    ${CMAKE_BINARY_DIR}/kaldifst-1.9.2.tar.gz
    /tmp/kaldifst-1.9.2.tar.gz
    /star-fj/fangjun/download/github/kaldifst-1.9.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldifst_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldifst_URL}" kaldifst_URL)
      message(STATUS "Found local downloaded kaldifst: ${kaldifst_URL}")
      break()
    endif()
  endforeach()

  set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  # NOTE: v1.9.0+ provides the zero-copy interop helpers
  # (_vector_fst_addr/_borrow/_take) and a matching openfst.  Keep in sync with
  # the pip requirement `kaldifst>=1.9.0` in setup.py.
  FetchContent_Declare(kaldifst
    URL               ${kaldifst_URL}
    URL_HASH          ${kaldifst_HASH}
  )

  # kaldifst's CMakeLists needs its own cmake/ dir on CMAKE_MODULE_PATH at
  # configure time (include(openfst)), so populate first, then add_subdirectory.
  FetchContent_Populate(kaldifst)
  message(STATUS "kaldifst is downloaded to ${kaldifst_SOURCE_DIR}")
  list(APPEND CMAKE_MODULE_PATH ${kaldifst_SOURCE_DIR}/cmake)
  add_subdirectory(${kaldifst_SOURCE_DIR} ${kaldifst_BINARY_DIR})

  # openfst's fst/extensions/ngram/nthbit.h calls __builtin_ctzll() and
  # __builtin_popcountll() but never includes <fst/compat.h>, where openfst
  # keeps its MSVC fallbacks for these GCC builtins.  MSVC therefore fails
  # to compile fstngram (C3861: '__builtin_ctzll': identifier not found);
  # other ngram sources get it only because their own headers happen to
  # include fst/compat.h first.  Patch the missing include into the
  # downloaded source; no-op wherever it is already present.
  FetchContent_GetProperties(openfst)
  if(openfst_POPULATED)
    set(nthbit_h ${openfst_SOURCE_DIR}/src/include/fst/extensions/ngram/nthbit.h)
    file(READ ${nthbit_h} nthbit_src)
    string(FIND "${nthbit_src}" "#include <fst/compat.h>" nthbit_compat_pos)
    if(nthbit_compat_pos EQUAL -1)
      string(REPLACE
        "#include <fst/log.h>"
        "#include <fst/compat.h>\n#include <fst/log.h>"
        nthbit_src "${nthbit_src}")
      file(WRITE ${nthbit_h} "${nthbit_src}")
      message(STATUS "Patched ${nthbit_h} to include fst/compat.h")
    endif()
  endif()

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
