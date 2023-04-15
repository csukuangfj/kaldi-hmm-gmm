// kaldi-hmm-gmm/csrc/event-map-test.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"

#include "gtest/gtest.h"

namespace khg {

TEST(DiagGmm, Case1) {
  int32_t nmix = 4;
  int32_t dim = 2;
  DiagGmm dgm(nmix, dim);

  torch::Tensor weights = torch::rand({nmix}, torch::kFloat);
  weights.div_(weights.sum());
  torch::Tensor means = torch::tensor(
      {
          {2, 2},
          {-10, -10},
          {1, 1},
          {-100, -100},
      },
      torch::kFloat);
  torch::Tensor vars = torch::rand({nmix, dim}, torch::kFloat);

  dgm.SetWeights(weights);
  dgm.SetMeans(means);
  dgm.SetInvVars(1 / vars);
  dgm.ComputeGconsts();  // essential!!

  dgm.MergeKmeans(3);  // TODO(fangjun): It aborts sometimes
#if 0
diag-gmm-test(47279,0x7ff85eb9c8c0) malloc: Incorrect checksum for freed object 0x7fd0318b4ac0: probably modified after being freed.
Corrupt value: 0x200000003f25532e
diag-gmm-test(47279,0x7ff85eb9c8c0) malloc: *** set a breakpoint in malloc_error_break to debug
Abort trap: 6
#endif

  KHG_LOG << "done\n";
}

}  // namespace khg
