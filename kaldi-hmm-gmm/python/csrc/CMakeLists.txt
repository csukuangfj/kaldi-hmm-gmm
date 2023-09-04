include_directories(${PROJECT_SOURCE_DIR})

set(srcs
  add-self-loops.cc
  am-diag-gmm.cc
  cluster-utils.cc
  clusterable-classes.cc
  context-dep.cc
  decodable-am-diag-gmm.cc
  decodable-itf.cc
  decoder-wrappers.cc
  determinize-lattice-pruned.cc
  diag-gmm.cc
  eigen-test.cc
  event-map.cc
  hmm-topology.cc
  hmm-utils.cc
  kaldi-hmm-gmm.cc
  lattice-faster-decoder.cc
  lattice-simple-decoder.cc
  mle-am-diag-gmm.cc
  mle-diag-gmm.cc
  model-common.cc
  training-graph-compiler.cc
  transition-information.cc
  transition-model.cc
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
