// kaldi-hmm-gmm/csrc/mle-diag-gmm-test.cc
//
// Copyright 2009-2011  Georg Stemmer;  Jan Silovsky;  Saarland University;
//                      Microsoft Corporation;  Yanmin Qian
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/gmm/mle-diag-gmm-test.cc (gtest-ified).  It
// covers:
//   - natural<->normal conversion (DiagGmmNormal::CopyToDiagGmm)
//   - component-wise vs whole-model accumulation equivalence
//     (AccumulateForComponent / AccumulateFromPosteriors /
//      AccumulateFromDiag)
//   - flags-driven MleDiagGmmUpdate
//   - the estimation loop (accumulation + MleDiagGmmUpdate + Split) with a
//     non-decreasing likelihood
// kaldi's test_io() and the multi-threaded accumulation test are not
// applicable here: our AccumDiagGmm has no Read/Write and no
// AccumulateFromDiagMultiThreaded.

#include "kaldi-hmm-gmm/csrc/mle-diag-gmm.h"

#include <cmath>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm-normal.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"

namespace khg {

namespace {

// Relative comparison.  kaldi AssertEqual/ApproxEqual is
// |a-b| <= tol * (|a| + |b|) (../kaldi/src/base/kaldi-math.h);
// we use the stricter |a-b| <= 0.5 * tol * (|a| + |b|).
void ExpectEqual(double a, double b, double tol = 1.0e-6) {
  EXPECT_LE(std::fabs(a - b), 0.5 * tol * (std::fabs(a) + std::fabs(b)))
      << "a = " << a << ", b = " << b;
}

// Synthetic data: `n_mix` clusters of `dim`-dimensional Gaussians,
// `multiple` frames each (as in kaldi's UnitTestEstimateDiagGmm).
void GenData(int32_t n_mix, int32_t dim, int32_t multiple, FloatMatrix *means,
             FloatMatrix *vars, FloatMatrix *feats) {
  means->resize(n_mix, dim);
  vars->resize(n_mix, dim);
  feats->resize(n_mix * multiple, dim);
  for (int32_t m = 0; m < n_mix; ++m) {
    for (int32_t d = 0; d < dim; ++d) {
      (*means)(m, d) = Randn(0.0f, 1.0f) * 100.0f;
      (*vars)(m, d) = std::exp(Randn(0.0f, 1.0f)) * 1000.0f + 1.0f;
    }
  }
  int32_t counter = 0;
  for (int32_t m = 0; m < n_mix; ++m) {
    for (int32_t i = 0; i < multiple; ++i, ++counter) {
      for (int32_t d = 0; d < dim; ++d) {
        (*feats)(counter, d) =
            (*means)(m, d) + Randn(0.0f, 1.0f) * std::sqrt((*vars)(m, d));
      }
    }
  }
}

FloatVector RowVec(const FloatMatrix &m, int32_t r) {
  return m.row(r).transpose();
}

// kaldi's TestComponentAcc: accumulating with AccumulateFromDiag (all
// components at once) must equal accumulating component by component.
void TestComponentAcc(const DiagGmm &gmm, const FloatMatrix &feats) {
  MleDiagGmmOptions config;
  AccumDiagGmm est_atonce;  // updates all components
  AccumDiagGmm est_compwise;  // updates single components

  est_atonce.Resize(gmm.NumGauss(), gmm.Dim(), kGmmAll);
  est_atonce.SetZero(kGmmAll);
  est_compwise.Resize(gmm.NumGauss(), gmm.Dim(), kGmmAll);
  est_compwise.SetZero(kGmmAll);

  for (int32_t i = 0; i < feats.rows(); ++i) {
    FloatVector feat = RowVec(feats, i);
    est_atonce.AccumulateFromDiag(gmm, feat, 1.0f);
    FloatVector post;
    gmm.ComponentPosteriors(feat, &post);
    for (int32_t m = 0; m < gmm.NumGauss(); ++m) {
      est_compwise.AccumulateForComponent(feat, m, post(m));
    }
  }

  DiagGmm gmm_atonce;  // model with all components accumulated together
  DiagGmm gmm_compwise;  // model with each component accumulated separately
  gmm_atonce.Resize(gmm.NumGauss(), gmm.Dim());
  gmm_compwise.Resize(gmm.NumGauss(), gmm.Dim());

  MleDiagGmmUpdate(config, est_atonce, kGmmAll, &gmm_atonce, nullptr, nullptr);
  MleDiagGmmUpdate(config, est_compwise, kGmmAll, &gmm_compwise, nullptr,
                   nullptr);

  double loglike0 = 0.0;
  double loglike1 = 0.0;
  double loglike2 = 0.0;
  for (int32_t i = 0; i < feats.rows(); ++i) {
    FloatVector feat = RowVec(feats, i);
    loglike0 += static_cast<double>(gmm.LogLikelihood(feat));
    loglike1 += static_cast<double>(gmm_atonce.LogLikelihood(feat));
    loglike2 += static_cast<double>(gmm_compwise.LogLikelihood(feat));
  }

  // the two ways of accumulating must give the same model
  ExpectEqual(loglike1, loglike2);

  if (est_atonce.NumGauss() == gmm.NumGauss()) {
    // the update must not decrease the likelihood
    EXPECT_GE(loglike1 + (std::fabs(loglike1) + std::fabs(loglike0)) * 1.0e-6,
              loglike0);
    EXPECT_GE(loglike2 + (std::fabs(loglike2) + std::fabs(loglike0)) * 1.0e-6,
              loglike0);
  }
}

// kaldi's test_flags_driven_update: updating only the parameters selected by
// `flags` must leave the others untouched.
void TestFlagsDrivenUpdate(const DiagGmm &gmm, const FloatMatrix &feats,
                           GmmFlagsType flags) {
  MleDiagGmmOptions config;
  AccumDiagGmm est_gmm_allp;  // updates all params
  AccumDiagGmm est_gmm_somep;  // updates params indicated by flags

  est_gmm_allp.Resize(gmm.NumGauss(), gmm.Dim(), kGmmAll);
  est_gmm_allp.SetZero(kGmmAll);
  est_gmm_somep.Resize(gmm.NumGauss(), gmm.Dim(), flags);
  est_gmm_somep.SetZero(flags);

  for (int32_t i = 0; i < feats.rows(); ++i) {
    FloatVector feat = RowVec(feats, i);
    est_gmm_allp.AccumulateFromDiag(gmm, feat, 1.0f);
    est_gmm_somep.AccumulateFromDiag(gmm, feat, 1.0f);
  }

  DiagGmm gmm_all_update;  // model with all params updated
  DiagGmm gmm_some_update;  // model with some params updated
  gmm_all_update.CopyFromDiagGmm(gmm);
  gmm_some_update.CopyFromDiagGmm(gmm);

  MleDiagGmmUpdate(config, est_gmm_allp, kGmmAll, &gmm_all_update, nullptr,
                   nullptr);
  MleDiagGmmUpdate(config, est_gmm_somep, flags, &gmm_some_update, nullptr,
                   nullptr);

  if (est_gmm_allp.NumGauss() != gmm.NumGauss()) {
    // component removal during Update() makes the comparison meaningless
    return;
  }

  // back off the params that were not updated to the original values
  if (~flags & kGmmWeights) {
    gmm_all_update.SetWeights(gmm.weights());
  }
  if (~flags & kGmmMeans) {
    gmm_all_update.SetMeans(gmm.GetMeans());
  }
  if (~flags & kGmmVariances) {
    FloatMatrix vars = gmm.GetVars();
    gmm_all_update.SetInvVars((1.0f / vars.array()).matrix());
  }
  gmm_all_update.ComputeGconsts();

  if ((flags & kGmmVariances) && !(flags & kGmmMeans)) {
    return;  // variance-only updates give a different answer if the mean
             // is not updated (kaldi skips the check too)
  }

  double loglike1 = 0.0;
  double loglike2 = 0.0;
  for (int32_t i = 0; i < feats.rows(); ++i) {
    FloatVector feat = RowVec(feats, i);
    loglike1 += static_cast<double>(gmm_all_update.LogLikelihood(feat));
    loglike2 += static_cast<double>(gmm_some_update.LogLikelihood(feat));
  }
  ExpectEqual(loglike1, loglike2);
}

}  // namespace

TEST(MleDiagGmm, NaturalNormalRoundTrip) {
  srand(1);
  int32_t dim = 15;
  DiagGmm gmm;
  gmm.Resize(1, dim);
  FloatVector weights(1);
  weights(0) = 1.0f;
  FloatMatrix means(1, dim);
  FloatMatrix vars(1, dim);
  for (int32_t d = 0; d < dim; ++d) {
    means(0, d) = Randn(0.0f, 1.0f) * 100.0f;
    vars(0, d) = std::exp(Randn(0.0f, 1.0f)) * 10.0f + 1e-5f;
  }
  gmm.SetWeights(weights);
  gmm.SetInvVarsAndMeans((1.0f / vars.array()).matrix(), means);
  gmm.ComputeGconsts();

  DiagGmmNormal ngmm(gmm);
  DiagGmm rgmm;
  rgmm.Resize(1, dim);
  ngmm.CopyToDiagGmm(&rgmm);

  EXPECT_FLOAT_EQ(gmm.weights()(0), rgmm.weights()(0));
  for (int32_t d = 0; d < dim; ++d) {
    // kaldi compares 1/invvars (= the variance) with ngmm.vars_
    ExpectEqual(means(0, d), ngmm.means_(0, d));
    ExpectEqual(vars(0, d), ngmm.vars_(0, d));
    ExpectEqual(gmm.means_invvars()(0, d), rgmm.means_invvars()(0, d));
    ExpectEqual(gmm.inv_vars()(0, d), rgmm.inv_vars()(0, d));
  }
}

TEST(MleDiagGmm, EstimateDiagGmm) {
  srand(1);
  int32_t dim = 15;  // dimension of the gmm
  int32_t n_mix = 9;  // number of mixtures in the data
  int32_t max_iterations = 20;
  // maximum number of densities in the GMM (larger than the number of
  // mixtures in the data so that we can test the removal of unseen components)
  int32_t max_components = 10;

  FloatMatrix means;
  FloatMatrix vars;
  FloatMatrix feats;
  GenData(n_mix, dim, 200, &means, &vars, &feats);
  int32_t counter = feats.rows();

  // initial single-component GMM
  DiagGmm gmm;
  gmm.Resize(1, dim);
  FloatVector weights(1);
  weights(0) = 1.0f;
  FloatMatrix init_means(1, dim);
  FloatMatrix init_vars(1, dim);
  for (int32_t d = 0; d < dim; ++d) {
    init_means(0, d) = Randn(0.0f, 1.0f) * 100.0f;
    init_vars(0, d) = std::exp(Randn(0.0f, 1.0f)) * 10.0f + 1e-5f;
  }
  gmm.SetWeights(weights);
  gmm.SetInvVarsAndMeans((1.0f / init_vars.array()).matrix(), init_means);
  gmm.ComputeGconsts();

  MleDiagGmmOptions config;
  config.min_variance = 0.01;
  GmmFlagsType flags = kGmmAll;

  AccumDiagGmm est_gmm;
  double last_loglike = 0.0;
  int32_t last_loglike_nm = 0;

  for (int32_t iteration = 0; iteration < max_iterations; ++iteration) {
    est_gmm.Resize(gmm.NumGauss(), gmm.Dim(), flags);
    est_gmm.SetZero(flags);
    double loglike = 0.0;
    for (int32_t i = 0; i < counter; ++i) {
      FloatVector feat = RowVec(feats, i);
      loglike += static_cast<double>(est_gmm.AccumulateFromDiag(gmm, feat,
                                                                1.0f));
    }
    // likelihood should be increasing on the long term
    if (iteration % 5 == 0) {
      if ((iteration > 0) && (gmm.NumGauss() >= last_loglike_nm)) {
        EXPECT_GE(loglike + 1.0, last_loglike);
      }
      last_loglike = loglike;
      last_loglike_nm = gmm.NumGauss();
    }

    float obj = 0.0f;
    float count = 0.0f;
    MleDiagGmmUpdate(config, est_gmm, flags, &gmm, &obj, &count);
    EXPECT_GT(count, 0.0f);

    if ((iteration % 3 == 1) && (gmm.NumGauss() * 2 <= max_components)) {
      gmm.Split(gmm.NumGauss() * 2, 0.001);
    }

    if (iteration == 5) {  // run the sub-tests with a not-too-overfitted model
      TestFlagsDrivenUpdate(gmm, feats, kGmmAll);
      TestFlagsDrivenUpdate(gmm, feats, kGmmWeights);
      TestFlagsDrivenUpdate(gmm, feats, kGmmMeans);
      TestFlagsDrivenUpdate(gmm, feats, kGmmVariances);
      TestFlagsDrivenUpdate(gmm, feats, kGmmWeights | kGmmMeans);
      TestComponentAcc(gmm, feats);
    }
  }
  EXPECT_GE(gmm.NumGauss(), 2);
}

}  // namespace khg

// ---------------------------------------------------------------------------
// Ported from ../kaldi/src/gmm/mle-diag-gmm.cc: AccumDiagGmm::SmoothStats,
// AccumDiagGmm::Add, AccumDiagGmm::AddStatsForComponent, MapDiagGmmUpdate;
// and DiagGmmNormal::CopyToDiagGmm flag paths
// (../kaldi/src/gmm/diag-gmm-normal.cc).
// ---------------------------------------------------------------------------

namespace khg {

TEST(MleDiagGmm, SmoothStatsAndAdd) {
  srand(1);
  const int32_t num_comp = 2;
  const int32_t dim = 3;

  AccumDiagGmm acc;
  acc.Resize(num_comp, dim, kGmmAll);
  acc.SetZero(kGmmAll);

  // comp 0: frame (1,2,3) with weight 2 and frame (2,0,1) with weight 1
  // -> occ 3, x_stats (4,4,7), x2_stats (6,8,19)
  FloatVector frame0(dim);
  frame0 << 1.0f, 2.0f, 3.0f;
  FloatVector frame1(dim);
  frame1 << 2.0f, 0.0f, 1.0f;
  acc.AccumulateForComponent(frame0, 0, 2.0f);
  acc.AccumulateForComponent(frame1, 0, 1.0f);

  // comp 1: frames (-1,0,2) and (1,2,0) with weight 0.5 each
  // -> occ 1, x_stats (0,1,1), x2_stats (1,2,2)
  FloatVector frame2(dim);
  frame2 << -1.0f, 0.0f, 2.0f;
  FloatVector frame3(dim);
  frame3 << 1.0f, 2.0f, 0.0f;
  acc.AccumulateForComponent(frame2, 1, 0.5f);
  acc.AccumulateForComponent(frame3, 1, 0.5f);

  ExpectEqual(acc.occupancy()(0), 3.0);
  ExpectEqual(acc.occupancy()(1), 1.0);
  const double x_stats_0[3] = {4.0, 4.0, 7.0};
  const double x2_stats_0[3] = {6.0, 8.0, 19.0};
  const double x_stats_1[3] = {0.0, 1.0, 1.0};
  const double x2_stats_1[3] = {1.0, 2.0, 2.0};
  for (int32_t d = 0; d < dim; ++d) {
    ExpectEqual(acc.mean_accumulator()(0, d), x_stats_0[d]);
    ExpectEqual(acc.variance_accumulator()(0, d), x2_stats_0[d]);
    ExpectEqual(acc.mean_accumulator()(1, d), x_stats_1[d]);
    ExpectEqual(acc.variance_accumulator()(1, d), x2_stats_1[d]);
  }

  // AddStatsForComponent: injecting the same explicit stats (already
  // multiplied by the occupancy) into a fresh accumulator reproduces `acc`.
  AccumDiagGmm acc_inj;
  acc_inj.Resize(num_comp, dim, kGmmAll);
  acc_inj.SetZero(kGmmAll);
  for (int32_t g = 0; g < num_comp; ++g) {
    DoubleVector x_stats = acc.mean_accumulator().row(g);
    DoubleVector x2_stats = acc.variance_accumulator().row(g);
    acc_inj.AddStatsForComponent(g, acc.occupancy()(g), x_stats, x2_stats);
  }
  for (int32_t g = 0; g < num_comp; ++g) {
    ExpectEqual(acc_inj.occupancy()(g), acc.occupancy()(g));
    for (int32_t d = 0; d < dim; ++d) {
      ExpectEqual(acc_inj.mean_accumulator()(g, d),
                  acc.mean_accumulator()(g, d));
      ExpectEqual(acc_inj.variance_accumulator()(g, d),
                  acc.variance_accumulator()(g, d));
    }
  }

  // an MLE update from the injected stats equals one from the accumulated
  // data when the stats match
  MleDiagGmmOptions config;
  config.min_gaussian_occupancy = 0.0f;
  config.min_gaussian_weight = 0.0f;

  DiagGmm gmm_data(num_comp, dim);
  DiagGmm gmm_inj(num_comp, dim);
  {
    FloatVector weights(num_comp);
    weights << 0.4f, 0.6f;
    FloatMatrix means(num_comp, dim);
    means << 1, 2, 3, 4, 5, 6;
    FloatMatrix inv_vars(num_comp, dim);
    inv_vars << 1, 1, 1, 0.5f, 0.5f, 0.5f;
    gmm_data.SetWeights(weights);
    gmm_data.SetInvVarsAndMeans(inv_vars, means);
    gmm_data.ComputeGconsts();
  }
  gmm_inj.CopyFromDiagGmm(gmm_data);

  MleDiagGmmUpdate(config, acc, kGmmAll, &gmm_data, nullptr, nullptr);
  MleDiagGmmUpdate(config, acc_inj, kGmmAll, &gmm_inj, nullptr, nullptr);
  for (int32_t g = 0; g < num_comp; ++g) {
    ExpectEqual(gmm_data.weights()(g), gmm_inj.weights()(g));
    for (int32_t d = 0; d < dim; ++d) {
      ExpectEqual(gmm_data.GetMeans()(g, d), gmm_inj.GetMeans()(g, d));
      ExpectEqual(gmm_data.GetVars()(g, d), gmm_inj.GetVars()(g, d));
    }
  }

  // SmoothStats(tau) adds tau to the occupancies and scales the
  // accumulators by (occ + tau) / occ.
  float tau = 1.5f;
  acc.SmoothStats(tau);
  ExpectEqual(acc.occupancy()(0), 3.0 + 1.5);  // 4.5
  ExpectEqual(acc.occupancy()(1), 1.0 + 1.5);  // 2.5
  // comp 0 scaled by 4.5/3 = 1.5, comp 1 by 2.5/1 = 2.5
  const double scaled_x_0[3] = {6.0, 6.0, 10.5};
  const double scaled_x2_0[3] = {9.0, 12.0, 28.5};
  const double scaled_x_1[3] = {0.0, 2.5, 2.5};
  const double scaled_x2_1[3] = {2.5, 5.0, 5.0};
  for (int32_t d = 0; d < dim; ++d) {
    ExpectEqual(acc.mean_accumulator()(0, d), scaled_x_0[d]);
    ExpectEqual(acc.variance_accumulator()(0, d), scaled_x2_0[d]);
    ExpectEqual(acc.mean_accumulator()(1, d), scaled_x_1[d]);
    ExpectEqual(acc.variance_accumulator()(1, d), scaled_x2_1[d]);
  }

  // Add(scale, other) adds the scaled accumulators of `other`.
  AccumDiagGmm acc2;
  acc2.Resize(num_comp, dim, kGmmAll);
  acc2.SetZero(kGmmAll);
  FloatVector one(dim);
  one.setConstant(1.0f);
  acc2.AccumulateForComponent(one, 0, 1.0f);  // occ 1, x (1,1,1), x2 (1,1,1)
  acc2.AccumulateForComponent(one, 1, 2.0f);  // occ 2, x (2,2,2), x2 (2,2,2)
  acc.Add(0.5f, acc2);
  ExpectEqual(acc.occupancy()(0), 4.5 + 0.5);  // 5.0
  ExpectEqual(acc.occupancy()(1), 2.5 + 1.0);  // 3.5
  const double sum_x_0[3] = {6.5, 6.5, 11.0};
  const double sum_x2_0[3] = {9.5, 12.5, 29.0};
  const double sum_x_1[3] = {1.0, 3.5, 3.5};
  const double sum_x2_1[3] = {3.5, 6.0, 6.0};
  for (int32_t d = 0; d < dim; ++d) {
    ExpectEqual(acc.mean_accumulator()(0, d), sum_x_0[d]);
    ExpectEqual(acc.variance_accumulator()(0, d), sum_x2_0[d]);
    ExpectEqual(acc.mean_accumulator()(1, d), sum_x_1[d]);
    ExpectEqual(acc.variance_accumulator()(1, d), sum_x2_1[d]);
  }
}

TEST(MleDiagGmm, MapDiagGmmUpdateSmoke) {
  srand(1);
  const int32_t num_comp = 2;
  const int32_t dim = 3;

  // prior model
  DiagGmm prior(num_comp, dim);
  FloatVector weights(num_comp);
  weights << 0.4f, 0.6f;
  FloatMatrix means(num_comp, dim);
  means << 1, 2, 3, 4, 5, 6;
  FloatMatrix vars(num_comp, dim);
  vars << 1.0f, 1.5f, 2.0f, 2.0f, 1.0f, 1.5f;
  prior.SetWeights(weights);
  prior.SetInvVarsAndMeans(vars.cwiseInverse(), means);
  prior.ComputeGconsts();

  // 20 frames per component, accumulated explicitly
  AccumDiagGmm acc;
  acc.Resize(num_comp, dim, kGmmAll);
  acc.SetZero(kGmmAll);
  const float offsets[5][3] = {{-1.0f, 0.0f, 1.0f},
                               {0.0f, 1.0f, -1.0f},
                               {1.0f, -1.0f, 0.0f},
                               {0.5f, 0.5f, -0.5f},
                               {-0.5f, -1.0f, 1.0f}};
  for (int32_t g = 0; g < num_comp; ++g) {
    for (int32_t k = 0; k < 5; ++k) {
      FloatVector frame(dim);
      for (int32_t d = 0; d < dim; ++d) {
        frame(d) = means(g, d) + offsets[k][d];
      }
      acc.AccumulateForComponent(frame, g, 4.0f);  // occ 20 per component
    }
  }

  // MLE update (no component removal on this data)
  MleDiagGmmOptions mle_config;
  mle_config.min_gaussian_occupancy = 0.0f;
  mle_config.min_gaussian_weight = 0.0f;

  DiagGmm gmm_mle(num_comp, dim);
  gmm_mle.CopyFromDiagGmm(prior);
  float obj_mle = 0.0f;
  float count_mle = 0.0f;
  MleDiagGmmUpdate(mle_config, acc, kGmmAll, &gmm_mle, &obj_mle, &count_mle);
  ExpectEqual(count_mle, 40.0f, 1.0e-6);
  EXPECT_TRUE(std::isfinite(obj_mle));

  // Small taus: the MAP update reduces to the MLE update (kaldi
  // mle-diag-gmm.cc: weights (occ + w*tau)/(occ_sum + tau) -> occ/occ_sum
  // and the mean/variance blends -> pure data estimates as tau -> 0).
  MapDiagGmmOptions small_config;
  small_config.mean_tau = 1.0e-3f;
  small_config.variance_tau = 1.0e-3f;
  small_config.weight_tau = 1.0e-3f;

  DiagGmm gmm_small(num_comp, dim);
  gmm_small.CopyFromDiagGmm(prior);
  float obj_small = 0.0f;
  float count_small = 0.0f;
  MapDiagGmmUpdate(small_config, acc, kGmmAll, &gmm_small, &obj_small,
                   &count_small);
  ExpectEqual(count_small, 40.0f, 1.0e-6);
  EXPECT_TRUE(std::isfinite(obj_small));

  // Large taus: the MAP update pulls the model toward the prior.
  MapDiagGmmOptions big_config;
  big_config.mean_tau = 1.0e6f;
  big_config.variance_tau = 1.0e6f;
  big_config.weight_tau = 1.0e6f;

  DiagGmm gmm_big(num_comp, dim);
  gmm_big.CopyFromDiagGmm(prior);
  float obj_big = 0.0f;
  float count_big = 0.0f;
  MapDiagGmmUpdate(big_config, acc, kGmmAll, &gmm_big, &obj_big, &count_big);
  ExpectEqual(count_big, 40.0f, 1.0e-6);
  EXPECT_TRUE(std::isfinite(obj_big));

  // all three models stay finite, normalized and valid
  const DiagGmm *models[3] = {&gmm_mle, &gmm_small, &gmm_big};
  for (int32_t m = 0; m < 3; ++m) {
    const DiagGmm &g = *models[m];
    EXPECT_TRUE(g.valid_gconsts());
    ExpectEqual(g.weights().sum(), 1.0, 1.0e-4);
    for (int32_t i = 0; i < num_comp; ++i) {
      for (int32_t d = 0; d < dim; ++d) {
        EXPECT_TRUE(std::isfinite(g.GetMeans()(i, d)));
        EXPECT_TRUE(std::isfinite(g.GetVars()(i, d)));
        EXPECT_GT(g.GetVars()(i, d), 0.0f);
      }
    }
  }

  // small taus approximate the MLE update
  for (int32_t i = 0; i < num_comp; ++i) {
    ExpectEqual(gmm_small.weights()(i), gmm_mle.weights()(i), 0.01);
    for (int32_t d = 0; d < dim; ++d) {
      ExpectEqual(gmm_small.GetMeans()(i, d), gmm_mle.GetMeans()(i, d), 0.01);
      ExpectEqual(gmm_small.GetVars()(i, d), gmm_mle.GetVars()(i, d), 0.01);
    }
  }

  // large taus keep the model close to the prior
  for (int32_t i = 0; i < num_comp; ++i) {
    ExpectEqual(gmm_big.weights()(i), prior.weights()(i), 0.01);
    for (int32_t d = 0; d < dim; ++d) {
      ExpectEqual(gmm_big.GetMeans()(i, d), prior.GetMeans()(i, d), 0.01);
      ExpectEqual(gmm_big.GetVars()(i, d), prior.GetVars()(i, d), 0.01);
    }
  }
}

TEST(MleDiagGmm, CopyToDiagGmmFlagPaths) {
  srand(1);
  const int32_t num_comp = 2;
  const int32_t dim = 2;

  // target model with known old parameters:
  // weights (0.25, 0.75), means [[1,2],[3,4]], inv-vars [[0.5,2],[1,0.25]]
  // -> old means_invvars [[0.5,4],[3,1]]
  auto build_target = [num_comp, dim](DiagGmm *t) {
    t->Resize(num_comp, dim);
    FloatVector w(num_comp);
    w << 0.25f, 0.75f;
    FloatMatrix m(num_comp, dim);
    m << 1, 2, 3, 4;
    FloatMatrix v(num_comp, dim);
    v << 0.5f, 2.0f, 1.0f, 0.25f;
    t->SetWeights(w);
    t->SetInvVarsAndMeans(v, m);
  };

  // source model in normal form with different parameters:
  // weights (0.4, 0.6), means [[10,20],[30,40]], vars [[4,1],[9,16]]
  DiagGmmNormal ngmm;
  ngmm.weights_ = DoubleVector(num_comp);
  ngmm.weights_ << 0.4, 0.6;
  ngmm.means_ = DoubleMatrix(num_comp, dim);
  ngmm.means_ << 10, 20, 30, 40;
  ngmm.vars_ = DoubleMatrix(num_comp, dim);
  ngmm.vars_ << 4, 1, 9, 16;

  // kGmmVariances without kGmmMeans uses the old-mean compensation branch
  // (../kaldi/src/gmm/diag-gmm-normal.cc): the inv-vars become the new
  // variances while means_invvars keep the *old* means times the *new*
  // inv-vars.
  {
    DiagGmm t;
    build_target(&t);
    ngmm.CopyToDiagGmm(&t, kGmmVariances);
    ExpectEqual(t.weights()(0), 0.25);  // untouched
    ExpectEqual(t.weights()(1), 0.75);
    ExpectEqual(t.inv_vars()(0, 0), 0.25);       // 1/4
    ExpectEqual(t.inv_vars()(0, 1), 1.0);        // 1/1
    ExpectEqual(t.inv_vars()(1, 0), 1.0 / 9.0);
    ExpectEqual(t.inv_vars()(1, 1), 1.0 / 16.0);
    ExpectEqual(t.means_invvars()(0, 0), 1.0 * 0.25);
    ExpectEqual(t.means_invvars()(0, 1), 2.0 * 1.0);
    ExpectEqual(t.means_invvars()(1, 0), 3.0 / 9.0);
    ExpectEqual(t.means_invvars()(1, 1), 4.0 / 16.0);
  }

  // kGmmMeans without kGmmVariances: the new means with the old inv-vars
  {
    DiagGmm t;
    build_target(&t);
    ngmm.CopyToDiagGmm(&t, kGmmMeans);
    ExpectEqual(t.weights()(0), 0.25);  // untouched
    ExpectEqual(t.weights()(1), 0.75);
    ExpectEqual(t.inv_vars()(0, 0), 0.5);  // untouched
    ExpectEqual(t.inv_vars()(0, 1), 2.0);
    ExpectEqual(t.inv_vars()(1, 0), 1.0);
    ExpectEqual(t.inv_vars()(1, 1), 0.25);
    ExpectEqual(t.means_invvars()(0, 0), 10.0 * 0.5);
    ExpectEqual(t.means_invvars()(0, 1), 20.0 * 2.0);
    ExpectEqual(t.means_invvars()(1, 0), 30.0 * 1.0);
    ExpectEqual(t.means_invvars()(1, 1), 40.0 * 0.25);
  }

  // kGmmAll: everything comes from the normal-form model
  {
    DiagGmm t;
    build_target(&t);
    ngmm.CopyToDiagGmm(&t, kGmmAll);
    ExpectEqual(t.weights()(0), 0.4);
    ExpectEqual(t.weights()(1), 0.6);
    ExpectEqual(t.inv_vars()(0, 0), 0.25);
    ExpectEqual(t.inv_vars()(0, 1), 1.0);
    ExpectEqual(t.inv_vars()(1, 0), 1.0 / 9.0);
    ExpectEqual(t.inv_vars()(1, 1), 1.0 / 16.0);
    ExpectEqual(t.means_invvars()(0, 0), 10.0 * 0.25);
    ExpectEqual(t.means_invvars()(0, 1), 20.0 * 1.0);
    ExpectEqual(t.means_invvars()(1, 0), 30.0 / 9.0);
    ExpectEqual(t.means_invvars()(1, 1), 40.0 / 16.0);
  }

  // kGmmWeights only: only the weights are transferred
  {
    DiagGmm t;
    build_target(&t);
    ngmm.CopyToDiagGmm(&t, kGmmWeights);
    ExpectEqual(t.weights()(0), 0.4);
    ExpectEqual(t.weights()(1), 0.6);
    ExpectEqual(t.inv_vars()(0, 0), 0.5);      // untouched
    ExpectEqual(t.means_invvars()(0, 0), 0.5);  // untouched
    ExpectEqual(t.means_invvars()(1, 1), 1.0);
  }
}

}  // namespace khg
