include_directories(${PROJECT_SOURCE_DIR})

set(srcs
  context-dep.cc
  kaldi-hmm-gmm.cc
  tree-renderer.cc
)

pybind11_add_module(_kaldi_hmm_gmm ${srcs})
target_link_libraries(_kaldi_hmm_gmm PRIVATE kaldi-hmm-gmm-core)

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_kaldi_hmm_gmm PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()

if(NOT WIN32)
  target_link_libraries(_kaldi_hmm_gmm PRIVATE "-Wl,-rpath,${KHG_RPATH_ORIGIN}/kaldi_hmm_gmm/lib")
endif()


install(TARGETS _kaldi_hmm_gmm
  DESTINATION ../
)
