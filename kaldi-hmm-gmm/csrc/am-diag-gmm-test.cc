// kaldi-hmm-gmm/csrc/am-diag-gmm-test.cc
//
// Copyright 2009-2011  Saarland University
// Author:  Arnab Ghoshal
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/gmm/am-diag-gmm-test.cc (gtest-ified).  It covers
// AmDiagGmm::Read/Write (ASCII + binary), CopyFromAmDiagGmm, SplitByCount and
// likelihood computation.  kaldi's TestClustering (ClusterGaussiansToUbm) is
// not applicable here: that function is not part of this repo.

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"

#include <cmath>
#include <cstdio>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

namespace {

// Uniform integer in [low, high] (kaldi RandInt semantics; test-local).
int32_t RandIntLocal(int32_t low, int32_t high) {
  return low + static_cast<int32_t>(RandUniform() * (high - low + 1));
}

// Random well-conditioned diagonal GMM (kaldi unittest::InitRandDiagGmm).
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

float TotalLogLike(const AmDiagGmm &am, const FloatVector &feat) {
  float loglike = 0.0f;
  for (int32_t i = 0; i < am.NumPdfs(); ++i) {
    loglike += am.LogLikelihood(i, feat);
  }
  return loglike;
}

// Relative comparison.  kaldi AssertEqual/ApproxEqual is
// |a-b| <= tol * (|a| + |b|) (../kaldi/src/base/kaldi-math.h);
// we use the stricter |a-b| <= 0.5 * tol * (|a| + |b|).
void ExpectEqual(double a, double b, double tol) {
  EXPECT_LE(std::fabs(a - b), 0.5 * tol * (std::fabs(a) + std::fabs(b)))
      << "a = " << a << ", b = " << b;
}

// kaldi's TestAmDiagGmmIO.
void TestAmDiagGmmIO(const AmDiagGmm &am_gmm) {
  int32_t dim = am_gmm.Dim();
  FloatVector feat(dim);
  for (int32_t d = 0; d < dim; ++d) {
    feat(d) = Randn(0.0f, 1.0f);
  }

  float loglike = TotalLogLike(am_gmm, feat);

  // ASCII write/read
  const char *ascii_name = "tmp_am_diag_gmm_ascii";
  am_gmm.Write(kaldiio::Output(ascii_name, false).Stream(), false);
  AmDiagGmm am1;
  bool binary_in = false;
  {
    kaldiio::Input ki(ascii_name, &binary_in);
    am1.Read(ki.Stream(), binary_in);
  }
  ExpectEqual(loglike, TotalLogLike(am1, feat), 1e-4);

  // binary write/read
  const char *bin_name = "tmp_am_diag_gmm_bin";
  am1.Write(kaldiio::Output(bin_name, true).Stream(), true);
  AmDiagGmm am2;
  {
    kaldiio::Input ki(bin_name, &binary_in);
    am2.Read(ki.Stream(), binary_in);
  }
  ExpectEqual(loglike, TotalLogLike(am2, feat), 1e-4);

  std::remove(ascii_name);
  std::remove(bin_name);
}

// kaldi's TestSplitStates: splitting must not change the likelihoods much.
void TestSplitStates(const AmDiagGmm &am_gmm) {
  int32_t target_comp = 2 * am_gmm.NumGauss();
  FloatVector occs(am_gmm.NumPdfs());
  for (int32_t i = 0; i < occs.size(); ++i) {
    occs(i) = std::fabs(Randn(0.0f, 1.0f)) * (RandUniform() + 1) * 4;
  }
  AmDiagGmm am1;
  am1.CopyFromAmDiagGmm(am_gmm);
  am1.SplitByCount(occs, target_comp, 0.01f, 0.2f, 0.0f);

  int32_t dim = am_gmm.Dim();
  FloatVector feat(dim);
  for (int32_t d = 0; d < dim; ++d) {
    feat(d) = Randn(0.0f, 1.0f);
  }
  float loglike = am_gmm.LogLikelihood(0, feat);
  float loglike1 = am1.LogLikelihood(0, feat);
  ExpectEqual(loglike, loglike1, 1e-2);
}

}  // namespace

TEST(AmDiagGmm, IoAndSplit) {
  srand(1);
  for (int32_t trial = 0; trial < 5; ++trial) {
    int32_t dim = 1 + RandIntLocal(0, 9);
    int32_t num_pdfs = 5 + RandIntLocal(0, 9);

    AmDiagGmm am_gmm;
    for (int32_t i = 0; i < num_pdfs; ++i) {
      int32_t num_comp = 1 + RandIntLocal(0, 9);
      DiagGmm gmm;
      InitRandDiagGmm(dim, num_comp, &gmm);
      am_gmm.AddPdf(gmm);
    }

    TestAmDiagGmmIO(am_gmm);
    TestSplitStates(am_gmm);
  }
}

}  // namespace khg
