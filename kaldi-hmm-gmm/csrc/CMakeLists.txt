include_directories(${CMAKE_SOURCE_DIR})

# Please keep the source files alphabetically sorted
set(srcs
  am-diag-gmm.cc
  build-tree-utils.cc
  cluster-utils.cc
  clusterable-classes.cc
  context-dep.cc
  decodable-am-diag-gmm.cc
  decodable-ctc.cc
  decoder-wrappers.cc
  diag-gmm-normal.cc
  diag-gmm.cc
  eigen.cc
  event-map.cc
  faster-decoder.cc
  hmm-topology.cc
  hmm-utils.cc
  kaldi-math.cc
  lattice-simple-decoder.cc
  lattice-faster-decoder.cc
  mle-am-diag-gmm.cc
  mle-diag-gmm.cc
  model-common.cc
  training-graph-compiler.cc
  transition-model.cc
  tree-renderer.cc
)
# The core library is always built as a static library (like
# kaldi-native-fbank-core); it is linked into the Python extension module
# and into the gtests.
add_library(kaldi-hmm-gmm-core STATIC ${srcs})
target_link_libraries(kaldi-hmm-gmm-core PUBLIC kaldi_native_io_core)
target_link_libraries(kaldi-hmm-gmm-core PUBLIC fst)
target_link_libraries(kaldi-hmm-gmm-core PUBLIC kaldifst_core)
target_link_libraries(kaldi-hmm-gmm-core PUBLIC Eigen3::Eigen)

if(KHG_ENABLE_TESTS)
  set(test_srcs
    am-diag-gmm-test.cc
    build-tree-utils-test.cc
    cluster-utils-test.cc
    context-dep-test.cc
    decodable-am-diag-gmm-test.cc
    diag-gmm-test.cc
    eigen-test.cc
    event-map-test.cc
    hash-list-test.cc
    hmm-topology-test.cc
    hmm-utils-test.cc
    kaldi-math-test.cc
    lattice-simple-decoder-test.cc
    mle-am-diag-gmm-test.cc
    mle-diag-gmm-test.cc
    model-common-test.cc
    transition-model-test.cc
  )

  function(khg_add_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name "${name}")
    add_executable(${target_name} ${source})
    target_link_libraries(${target_name}
      PRIVATE
      gtest
      gtest_main
      kaldi-hmm-gmm-core
    )
    add_test(NAME "Test.${target_name}"
      COMMAND
      $<TARGET_FILE:${target_name}>
    )
  endfunction()

  foreach(source IN LISTS test_srcs)
    khg_add_test(${source})
  endforeach()
endif()

if(KHG_INSTALL_CORE_LIB)
  install(TARGETS kaldi-hmm-gmm-core DESTINATION lib)
endif()
