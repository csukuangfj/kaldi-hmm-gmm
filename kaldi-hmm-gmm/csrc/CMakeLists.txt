include_directories(${CMAKE_SOURCE_DIR})

# Please keep the source files alphabetically sorted
set(srcs
  hmm-topology.cc
)
add_library(kaldi-hmm-gmm-core ${srcs})
target_link_libraries(kaldi-hmm-gmm-core PUBLIC kaldi_native_io_core)
