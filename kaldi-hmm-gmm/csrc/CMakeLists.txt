include_directories(${CMAKE_SOURCE_DIR})

# Please keep the source files alphabetically sorted
set(srcs
  hmm-topology.cc
)
add_library(kaldi-hmm-gmm-core ${srcs})
