// kaldi-hmm-gmm/csrc/cluster-utils.h
//
// Copyright 2012   Arnab Ghoshal
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CLUSTER_UTILS_H_
#define KALDI_HMM_GMM_CSRC_CLUSTER_UTILS_H_

#include <stdint.h>

namespace khg {

struct RefineClustersOptions {
  int32_t num_iters = 100;  // must be >= 0.  If zero, does nothing.
  int32_t top_n = 5;        // must be >= 2.
  RefineClustersOptions() = default;
  RefineClustersOptions(int32_t num_iters_in, int32_t top_n_in)
      : num_iters(num_iters_in), top_n(top_n_in) {}
};

struct ClusterKMeansOptions {
  RefineClustersOptions refine_cfg;
  int32_t num_iters = 20;
  int32_t num_tries = 2;  // if >1, try whole procedure >once and pick best.
  bool verbose = true;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTER_UTILS_H_
