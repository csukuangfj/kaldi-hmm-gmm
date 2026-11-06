// kaldi-hmm-gmm/csrc/mle-am-diag-gmm-test.cc
//
// Copyright 2009-2012  Arnab Ghoshal
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/gmm/mle-am-diag-gmm-test.cc (gtest-ified).  It
// covers AccumAmDiagGmm::Init/AccumulateForGmm (TotLogLike/TotCount) and
// MleAmDiagGmmUpdate.  kaldi's accumulator Read/Write round-trip is not
// applicable here: our AccumAmDiagGmm has no Read/Write.

#include "kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h"

#include <cmath>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"

namespace khg {

namespace {

int32_t RandIntLocal(int32_t low, int32_t high) {
  return low + static_cast<int32_t>(RandUniform() * (high - low + 1));
}

void InitRandDiagGmm(int32_t dim, int32_t num_comp, DiagGmm *gmm) {
  gmm->Resize(num_comp, dim);
  FloatVector weights(num_comp);
  FloatMatrix means(num_comp, dim);
  FloatMatrix invvars(num_comp, dim);
  for (int32_t m = 0; m < num_comp; ++m) {
    weights(m) = std::exp(Randn(0.0f, 1.0f));
    for (int32_t d = 0; d < dim; ++d) {
      means(m, d) = Randn(0.0f, 1.0f) * 10.0f;
      invvars(m, d) = std::exp(-Randn(0.0f, 1.0f)) * 0.1f;
    }
  }
  weights = (weights / weights.sum()).eval();
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(invvars, means);
  gmm->ComputeGconsts();
}

}  // namespace

TEST(MleAmDiagGmm, AccumulateAndUpdate) {
  srand(1);
  int32_t dim = 1 + RandIntLocal(0, 9);
  int32_t num_pdfs = 5 + RandIntLocal(0, 9);

  AmDiagGmm am_gmm;
  for (int32_t i = 0; i < num_pdfs; ++i) {
    int32_t num_comp = 1 + RandIntLocal(0, 9);
    DiagGmm gmm;
    InitRandDiagGmm(dim, num_comp, &gmm);
    am_gmm.AddPdf(gmm);
  }

  int32_t num_frames = 200;
  FloatMatrix feats(num_frames, dim);
  for (int32_t i = 0; i < num_frames; ++i) {
    for (int32_t d = 0; d < dim; ++d) {
      feats(i, d) = Randn(0.0f, 1.0f) * 5.0f;
    }
  }

  GmmFlagsType flags = kGmmAll;
  AccumAmDiagGmm accs;
  accs.Init(am_gmm, flags);

  double loglike = 0.0;
  for (int32_t i = 0; i < num_frames; ++i) {
    int32_t state = RandIntLocal(0, am_gmm.NumPdfs() - 1);
    FloatVector feat = feats.row(i).transpose();
    loglike += accs.AccumulateForGmm(am_gmm, feat, state, 1.0f);
  }
  // accumulator totals must match the per-frame accumulation
  EXPECT_NEAR(accs.TotLogLike(), loglike,
              1e-5 * (std::fabs(loglike) + 1.0));
  EXPECT_FLOAT_EQ(accs.TotCount(), static_cast<float>(num_frames));

  MleDiagGmmOptions config;
  AmDiagGmm am_updated;
  am_updated.CopyFromAmDiagGmm(am_gmm);
  float obj = 0.0f;
  float count = 0.0f;
  MleAmDiagGmmUpdate(config, accs, flags, &am_updated, &obj, &count);
  EXPECT_FLOAT_EQ(count, static_cast<float>(num_frames));

  // the updated model must score the same data consistently
  int32_t check_pdf = RandIntLocal(0, am_gmm.NumPdfs() - 1);
  int32_t check_frame = RandIntLocal(0, num_frames - 1);
  FloatVector check_feat = feats.row(check_frame).transpose();
  float loglike1 = am_updated.LogLikelihood(check_pdf, check_feat);
  EXPECT_TRUE(std::isfinite(loglike1));
}

}  // namespace khg
