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

  dgm.MergeKmeans(3);
}

}  // namespace khg
