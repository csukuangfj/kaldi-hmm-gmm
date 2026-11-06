include_directories(${PROJECT_SOURCE_DIR})

set(srcs
  add-self-loops.cc
  am-diag-gmm.cc
  cluster-utils.cc
  clusterable-classes.cc
  context-dep.cc
  decodable-am-diag-gmm.cc
  decodable-ctc.cc
  decodable-itf.cc
  decoder-wrappers.cc
  determinize-lattice-pruned.cc
  diag-gmm.cc
  event-map.cc
  faster-decoder.cc
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

set_target_properties(_kaldi_hmm_gmm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../kaldi_hmm_gmm/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../kaldi_hmm_gmm/lib"
)
# multi-config generators (MSVC) append /<Config> to the output dirs above;
# pin them so the module always lands in kaldi_hmm_gmm/lib/.
foreach(cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${cfg}" CFG)
  set_target_properties(_kaldi_hmm_gmm PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${CFG} "${CMAKE_CURRENT_SOURCE_DIR}/../kaldi_hmm_gmm/lib"
    RUNTIME_OUTPUT_DIRECTORY_${CFG} "${CMAKE_CURRENT_SOURCE_DIR}/../kaldi_hmm_gmm/lib"
  )
endforeach()

target_link_libraries(_kaldi_hmm_gmm PRIVATE kaldi-hmm-gmm-core)

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  if(PYTHON_SITE_PACKAGE_DIR STREQUAL "")
    message(WARNING "PYTHON_SITE_PACKAGE_DIR is empty!")
  else()
    target_link_libraries(_kaldi_hmm_gmm PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
  endif()
endif()

if(NOT WIN32)
  target_link_libraries(_kaldi_hmm_gmm PRIVATE "-Wl,-rpath,${KHG_RPATH_ORIGIN}/kaldi_hmm_gmm/lib")
endif()


install(TARGETS _kaldi_hmm_gmm
  DESTINATION lib
)
