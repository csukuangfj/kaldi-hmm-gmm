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

  FloatVector weights = FloatVector::Random(nmix).array() + 1;

  weights /= weights.sum();

  FloatMatrix means(nmix, dim);
  means << 2, 2, -10, -10, 1, 1, -100, -100;
  FloatMatrix vars = FloatMatrix::Random(nmix, dim).array() + 1;

  dgm.SetWeights(weights);
  dgm.SetMeans(means);
  dgm.SetInvVars(1 / vars.array());
  dgm.ComputeGconsts();  // essential!!

  dgm.MergeKmeans(3);
}

}  // namespace khg
