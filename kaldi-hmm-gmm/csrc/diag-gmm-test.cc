// kaldi-hmm-gmm/csrc/diag-gmm-test.cc
//
// Copyright 2009-2011  Microsoft Corporation;  Georg Stemmer;  Jan Silovsky;
//                      Saarland University
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/gmm/diag-gmm-test.cc (gtest-ified).

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/mle-diag-gmm.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

namespace {

// kaldi AssertEqual(a, b, tol) is relative: |a-b| <= 0.5 * tol * (|a| + |b|)
void ExpectEqual(double a, double b, double tol = 1.0e-6) {
  EXPECT_LE(std::fabs(a - b), 0.5 * tol * (std::fabs(a) + std::fabs(b)))
      << "a = " << a << ", b = " << b;
}

// Uniform integer in [low, high] (kaldi RandInt semantics; test-local).
int32_t RandIntLocal(int32_t low, int32_t high) {
  return low + static_cast<int32_t>(RandUniform() * (high - low + 1));
}

// Row r of matrix m as a column vector.
FloatVector RowVec(const FloatMatrix &m, int32_t r) {
  return m.row(r).transpose();
}

// Random well-conditioned GMM (kaldi's InitRandomGmm in diag-gmm-test.cc).
void InitRandomGmm(DiagGmm *gmm) {
  int32_t num_gauss = 10 + std::rand() % 5;
  int32_t dim = 10 + std::rand() % 10;
  gmm->Resize(num_gauss, dim);
  FloatMatrix inv_vars(num_gauss, dim);
  FloatMatrix means(num_gauss, dim);
  FloatVector weights(num_gauss);
  for (int32_t i = 0; i < num_gauss; ++i) {
    for (int32_t j = 0; j < dim; ++j) {
      inv_vars(i, j) = std::exp(Randn(0.0f, 1.0f) * (1.0f / (1 + j)));
      means(i, j) = Randn(0.0f, 1.0f) * (1.0f / (1 + j));
    }
    weights(i) = std::exp(Randn(0.0f, 1.0f));
  }
  weights = (weights / weights.sum()).eval();
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(inv_vars, means);
  gmm->Perturb(0.5f * RandUniform());
  gmm->ComputeGconsts();  // unnecessary; computed in Perturb (as in kaldi)
}

// Random data for the UnitTestDiagGmm-style cases: one random feature vector,
// random GMM parameters, and the manual per-component log-likelihoods computed
// with the textbook formula.
struct TestData {
  int32_t dim;
  int32_t nmix;
  FloatVector feat;
  FloatVector weights;
  FloatMatrix means;
  FloatMatrix vars;      // variances (not inverted)
  FloatVector loglikes;  // manual per-component log-likes
  float loglike;         // LogSumExp of loglikes
};

void MakeTestData(TestData *t) {
  t->dim = 1 + RandIntLocal(0, 5);
  t->nmix = 1 + RandIntLocal(0, 5);
  t->feat = FloatVector(t->dim);
  t->weights = FloatVector(t->nmix);
  t->means = FloatMatrix(t->nmix, t->dim);
  t->vars = FloatMatrix(t->nmix, t->dim);
  t->loglikes = FloatVector::Zero(t->nmix);
  for (int32_t d = 0; d < t->dim; ++d) {
    t->feat(d) = Randn(0.0f, 1.0f);
  }
  float tot_weight = 0.0f;
  for (int32_t m = 0; m < t->nmix; ++m) {
    t->weights(m) = RandUniform();
    for (int32_t d = 0; d < t->dim; ++d) {
      t->means(m, d) = Randn(0.0f, 1.0f);
      t->vars(m, d) = std::exp(Randn(0.0f, 1.0f)) + 1e-5f;
    }
    tot_weight += t->weights(m);
  }
  for (int32_t m = 0; m < t->nmix; ++m) {
    t->weights(m) /= tot_weight;
    for (int32_t d = 0; d < t->dim; ++d) {
      float diff = t->feat(d) - t->means(m, d);
      t->loglikes(m) += -0.5f * (M_LOG_2PI + std::log(t->vars(m, d)) +
                                 diff * diff / t->vars(m, d));
    }
    t->loglikes(m) += std::log(t->weights(m));
  }
  t->loglike = LogSumExp(t->loglikes);
}

// Builds a DiagGmm from the weights/means/vars of `t`.
void BuildGmm(const TestData &t, DiagGmm *gmm) {
  gmm->Resize(t.nmix, t.dim);
  FloatMatrix invvars = t.vars.cwiseInverse();
  gmm->SetWeights(t.weights);
  gmm->SetInvVarsAndMeans(invvars, t.means);
  gmm->ComputeGconsts();
}

// 1-component GMM from the first random component of `t` (kaldi builds this
// from means.Range(0, 1, 0, dim) etc.).
void MakeSingleGmm(const TestData &t, DiagGmm *gmm) {
  gmm->Resize(1, t.dim);
  FloatVector weights1 = FloatVector::Ones(1);
  FloatMatrix means1(1, t.dim);
  means1.row(0) = t.means.row(0);
  FloatMatrix vars1(1, t.dim);
  vars1.row(0) = t.vars.row(0);
  FloatMatrix invvars1 = vars1.cwiseInverse();
  gmm->SetWeights(weights1);
  gmm->SetInvVarsAndMeans(invvars1, means1);
  gmm->ComputeGconsts();
}

}  // namespace

// kaldi's manual per-component log-likelihood computation must agree with
// ComponentLogLikelihood, and the implied total (via LogSumExp) with the
// ComponentPosteriors return value.
TEST(DiagGmm, ComponentPosteriors) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm;
  BuildGmm(t, &gmm);

  for (int32_t m = 0; m < t.nmix; ++m) {
    ExpectEqual(t.loglikes(m), gmm.ComponentLogLikelihood(t.feat, m), 1e-3);
  }

  FloatVector post;
  float loglike1 = gmm.ComponentPosteriors(t.feat, &post);
  ExpectEqual(t.loglike, loglike1, 0.01);
  ExpectEqual(1.0, post.sum(), 0.01);
}

// GetMeans/GetVars + SetWeights/SetMeans/SetInvVars rebuild of the model must
// preserve the likelihood (kaldi's accessors/mutators block).
TEST(DiagGmm, RebuildFromAccessors) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm;
  BuildGmm(t, &gmm);
  float loglike1 = gmm.LogLikelihood(t.feat);

  FloatVector weights_bak = gmm.weights();
  FloatMatrix means_bak = gmm.GetMeans();
  FloatMatrix vars_bak = gmm.GetVars();
  FloatMatrix invvars_bak = vars_bak.cwiseInverse();

  DiagGmm gmm2(gmm.NumGauss(), gmm.Dim());
  gmm2.SetWeights(weights_bak);
  gmm2.SetMeans(means_bak);
  gmm2.SetInvVars(invvars_bak);
  gmm2.ComputeGconsts();
  float loglike2 = gmm2.LogLikelihood(t.feat);
  ExpectEqual(loglike1, loglike2, 1e-3);
}

// LogLikelihoods(vector) plus khg's LogSumExp must equal LogLikelihood.
TEST(DiagGmm, LogLikelihoods) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm;
  BuildGmm(t, &gmm);

  FloatVector loglikes;
  gmm.LogLikelihoods(t.feat, &loglikes);
  ExpectEqual(LogSumExp(loglikes), gmm.LogLikelihood(t.feat), 1e-3);
}

// LogLikelihoodsPreselect with the full index list must equal LogLikelihoods.
TEST(DiagGmm, LogLikelihoodsPreselect) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm;
  BuildGmm(t, &gmm);

  std::vector<int32_t> indices;
  for (int32_t i = 0; i < gmm.NumGauss(); ++i) {
    indices.push_back(i);
  }
  FloatVector loglikes_pre;
  gmm.LogLikelihoodsPreselect(t.feat, indices, &loglikes_pre);
  FloatVector loglikes;
  gmm.LogLikelihoods(t.feat, &loglikes);
  ASSERT_EQ(loglikes_pre.size(), loglikes.size());
  for (int32_t i = 0; i < loglikes.size(); ++i) {
    ExpectEqual(loglikes_pre(i), loglikes(i), 1e-3);
  }
  ExpectEqual(LogSumExp(loglikes_pre), gmm.LogLikelihood(t.feat), 1e-3);
}

// GetComponentMean/SetComponentMean and GetComponentVariance/SetComponentInvVar
// roundtrip must preserve the likelihood.
TEST(DiagGmm, ComponentMeanVarianceRoundtrip) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm;
  BuildGmm(t, &gmm);
  float loglike1 = gmm.LogLikelihood(t.feat);

  FloatMatrix means = gmm.GetMeans();
  FloatMatrix vars = gmm.GetVars();
  DiagGmm gmm3(gmm.NumGauss(), gmm.Dim());
  gmm3.SetWeights(gmm.weights());
  for (int32_t i = 0; i < gmm.NumGauss(); ++i) {
    FloatVector mean_i = gmm.GetComponentMean(i);
    FloatVector var_i = gmm.GetComponentVariance(i);
    for (int32_t d = 0; d < t.dim; ++d) {
      ExpectEqual(means(i, d), mean_i(d), 1e-6);
      ExpectEqual(vars(i, d), var_i(d), 1e-6);
    }
    // set the single component in the order the API recommends: inv-var first
    FloatVector invvar_i = var_i.cwiseInverse();
    gmm3.SetComponentInvVar(i, invvar_i);
    gmm3.SetComponentMean(i, mean_i);
  }
  gmm3.ComputeGconsts();
  float loglike3 = gmm3.LogLikelihood(t.feat);
  ExpectEqual(loglike1, loglike3, 0.01);
}

// Non-binary Write -> Read must preserve the likelihood.
TEST(DiagGmm, AsciiWriteRead) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm;
  BuildGmm(t, &gmm);
  float loglike = gmm.LogLikelihood(t.feat);

  const char *filename = "tmp_diag_gmm_test_ascii";
  gmm.Write(kaldiio::Output(filename, false).Stream(), false);
  DiagGmm gmm2;
  bool binary_in = false;
  {
    kaldiio::Input ki(filename, &binary_in);
    gmm2.Read(ki.Stream(), binary_in);
  }
  FloatVector post;
  float loglike2 = gmm2.ComponentPosteriors(t.feat, &post);
  ExpectEqual(loglike, loglike2, 0.01);
  std::remove(filename);
}

// Binary Write -> Read must preserve the likelihood.
TEST(DiagGmm, BinaryWriteRead) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm;
  BuildGmm(t, &gmm);
  float loglike = gmm.LogLikelihood(t.feat);

  const char *filename = "tmp_diag_gmm_test_bin";
  gmm.Write(kaldiio::Output(filename, true).Stream(), true);
  DiagGmm gmm2;
  bool binary_in = true;
  {
    kaldiio::Input ki(filename, &binary_in);
    gmm2.Read(ki.Stream(), binary_in);
  }
  FloatVector post;
  float loglike2 = gmm2.ComponentPosteriors(t.feat, &post);
  ExpectEqual(loglike, loglike2, 0.01);
  std::remove(filename);
}

// Split + Merge on a 1-component GMM (doesn't test the merge criterion): the
// log-likelihood must be preserved.
TEST(DiagGmm, SplitMerge) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm1;
  MakeSingleGmm(t, &gmm1);

  DiagGmm gmm2;
  gmm2.CopyFromDiagGmm(gmm1);
  gmm2.Split(2, 0.001f);
  gmm2.Merge(1);
  float loglike1 = gmm1.LogLikelihood(t.feat);
  float loglike2 = gmm2.LogLikelihood(t.feat);
  ExpectEqual(loglike1, loglike2, 0.01);
}

// Split + MergeKmeans on a 1-component GMM: the log-likelihood must be
// preserved.
TEST(DiagGmm, SplitMergeKmeans) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm1;
  MakeSingleGmm(t, &gmm1);

  DiagGmm gmm2;
  gmm2.CopyFromDiagGmm(gmm1);
  gmm2.Split(2, 0.001f);
  gmm2.MergeKmeans(1);
  float loglike1 = gmm1.LogLikelihood(t.feat);
  float loglike2 = gmm2.LogLikelihood(t.feat);
  ExpectEqual(loglike1, loglike2, 0.01);
}

// Duplicate Gaussians using the merge constructor that takes a vector: two
// weighted copies of the same GMM must keep the likelihood unchanged.
TEST(DiagGmm, MergeConstructor) {
  srand(1);
  TestData t;
  MakeTestData(&t);
  DiagGmm gmm1;
  MakeSingleGmm(t, &gmm1);

  std::vector<std::pair<float, const DiagGmm *>> vec;
  vec.push_back(std::make_pair(0.4f, static_cast<const DiagGmm *>(&gmm1)));
  vec.push_back(std::make_pair(0.6f, static_cast<const DiagGmm *>(&gmm1)));
  DiagGmm gmm2(vec);
  float loglike1 = gmm1.LogLikelihood(t.feat);
  float loglike2 = gmm2.LogLikelihood(t.feat);
  ExpectEqual(loglike1, loglike2, 0.01);
}

// kaldi's UnitTestDiagGmmGenerate: Generate + MleDiagGmmUpdate loop; the total
// objective change is bounded by an AIC-style estimate.  Relies on statistical
// laws; if it fails it does not *necessarily* mean something is wrong.
TEST(DiagGmm, GenerateAndUpdate) {
  srand(1);
  DiagGmm gmm;
  InitRandomGmm(&gmm);
  int32_t dim = gmm.Dim();
  int32_t npoints = 100 * gmm.NumGauss();
  FloatMatrix rand_points(npoints, dim);
  for (int32_t i = 0; i < npoints; ++i) {
    FloatVector point = RowVec(rand_points, i);
    gmm.Generate(&point);
    rand_points.row(i) = point.transpose();
  }

  int32_t niters = 15;
  float objf_change_tot = 0.0f;
  float objf_change = 0.0f;
  float count = 0.0f;
  for (int32_t j = 0; j < niters; ++j) {
    MleDiagGmmOptions opts;
    AccumDiagGmm stats(gmm, kGmmAll);  // all update flags
    for (int32_t i = 0; i < npoints; ++i) {
      FloatVector point = RowVec(rand_points, i);
      stats.AccumulateFromDiag(gmm, point, 1.0f);
    }
    MleDiagGmmUpdate(opts, stats, kGmmAll, &gmm, &objf_change, &count);
    objf_change_tot += objf_change;
  }
  ExpectEqual(count, npoints, 1e-6);
  int32_t num_params = gmm.NumGauss() * (gmm.Dim() * 2 + 1);
  float expected_objf_change = 0.5f * num_params;
  EXPECT_LT(objf_change_tot, 2.0f * expected_objf_change);
}

// Pre-existing smoke case: MergeKmeans on a small GMM with well-separated
// components must run without crashing and keep the model valid.
TEST(DiagGmm, MergeKmeansSmoke) {
  srand(1);
  int32_t nmix = 4;
  int32_t dim = 2;
  DiagGmm dgm(nmix, dim);

  FloatVector weights(nmix);
  for (int32_t m = 0; m < nmix; ++m) {
    weights(m) = RandUniform() + 1.0f;
  }
  weights = (weights / weights.sum()).eval();

  FloatMatrix means(nmix, dim);
  means << 2, 2, -10, -10, 1, 1, -100, -100;
  FloatMatrix vars(nmix, dim);
  for (int32_t m = 0; m < nmix; ++m) {
    for (int32_t d = 0; d < dim; ++d) {
      vars(m, d) = RandUniform() * 2.0f;
    }
  }
  FloatMatrix invvars = vars.cwiseInverse();

  dgm.SetWeights(weights);
  dgm.SetMeans(means);
  dgm.SetInvVars(invvars);
  dgm.ComputeGconsts();  // essential!!

  dgm.MergeKmeans(3);
  EXPECT_EQ(dgm.NumGauss(), 3);
  EXPECT_TRUE(dgm.valid_gconsts());
}

}  // namespace khg

// ---------------------------------------------------------------------------
// Ported from ../kaldi/src/gmm/diag-gmm.cc: DiagGmm::Perturb,
// DiagGmm::Interpolate, DiagGmm::RemoveComponent.
// ---------------------------------------------------------------------------

namespace khg {

TEST(DiagGmm, PerturbAndInterpolate) {
  srand(1);
  const int32_t num_comp = 3;
  const int32_t dim = 2;

  DiagGmm gmm(num_comp, dim);
  FloatVector weights(num_comp);
  weights << 0.2f, 0.3f, 0.5f;
  FloatMatrix means(num_comp, dim);
  means << 0, 0, 2, 2, -2, 1;
  FloatMatrix inv_vars(num_comp, dim);
  inv_vars.setConstant(1.0f);
  gmm.SetWeights(weights);
  gmm.SetInvVarsAndMeans(inv_vars, means);
  gmm.ComputeGconsts();

  FloatMatrix means_before = gmm.GetMeans();
  FloatMatrix vars_before = gmm.GetVars();
  FloatVector weights_before = gmm.weights();

  gmm.Perturb(0.1f);

  // Perturb changes the means but preserves the component count and the
  // weights/inv-vars.
  EXPECT_EQ(gmm.NumGauss(), num_comp);
  EXPECT_TRUE(gmm.valid_gconsts());
  EXPECT_FALSE(gmm.GetMeans().isApprox(means_before));
  EXPECT_TRUE(gmm.GetVars().isApprox(vars_before));
  EXPECT_TRUE(gmm.weights().isApprox(weights_before));

  // Interpolate(rho) with an identical source must leave the model and its
  // log-likelihood unchanged.
  DiagGmm source(gmm);
  gmm.Interpolate(0.3f, source, kGmmAll);
  EXPECT_TRUE(gmm.GetMeans().isApprox(source.GetMeans()));
  EXPECT_TRUE(gmm.GetVars().isApprox(source.GetVars()));
  EXPECT_TRUE(gmm.weights().isApprox(source.weights()));

  FloatVector pt(dim);
  pt << 0.5f, -1.25f;
  float loglike_before = source.LogLikelihood(pt);
  float loglike_after = gmm.LogLikelihood(pt);
  ExpectEqual(loglike_before, loglike_after, 1.0e-4);
}

TEST(DiagGmm, RemoveComponent) {
  srand(1);
  const int32_t num_comp = 3;
  const int32_t dim = 2;

  DiagGmm gmm(num_comp, dim);
  FloatVector weights(num_comp);
  weights << 0.2f, 0.3f, 0.5f;
  FloatMatrix means(num_comp, dim);
  means << 0, 0, 2, 2, -2, 1;
  FloatMatrix inv_vars(num_comp, dim);
  inv_vars << 1, 1, 2, 2, 0.5f, 0.5f;
  gmm.SetWeights(weights);
  gmm.SetInvVarsAndMeans(inv_vars, means);
  gmm.ComputeGconsts();

  FloatVector mean0 = gmm.GetComponentMean(0);
  FloatVector mean2 = gmm.GetComponentMean(2);
  FloatVector var0 = gmm.GetComponentVariance(0);
  FloatVector var2 = gmm.GetComponentVariance(2);

  gmm.RemoveComponent(1, true);

  // removing one of 3 components leaves the others' parameters unchanged
  EXPECT_EQ(gmm.NumGauss(), 2);
  EXPECT_TRUE(gmm.GetComponentMean(0).isApprox(mean0));
  EXPECT_TRUE(gmm.GetComponentMean(1).isApprox(mean2));
  EXPECT_TRUE(gmm.GetComponentVariance(0).isApprox(var0));
  EXPECT_TRUE(gmm.GetComponentVariance(1).isApprox(var2));

  // the weights of the survivors are renormalized to sum to one
  ExpectEqual(gmm.weights()(0), 0.2f / 0.7f);
  ExpectEqual(gmm.weights()(1), 0.5f / 0.7f);
  ExpectEqual(gmm.weights().sum(), 1.0, 1.0e-6);

  gmm.ComputeGconsts();
  EXPECT_TRUE(gmm.valid_gconsts());
}

}  // namespace khg
