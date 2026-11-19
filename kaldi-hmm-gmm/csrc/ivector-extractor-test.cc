// kaldi-hmm-gmm/csrc/ivector-extractor-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Unit tests for the i-vector runtime subset (ivector-extractor.{h,cc} and
// online-ivector-extractor.{h,cc}), gtest-ified in the style of
// diag-gmm-test.cc.  Integration parity with kaldi's ivector-extract-online
// lives in tests/parity/parity_ivector.py.

#include "kaldi-hmm-gmm/csrc/ivector-extractor.h"

#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/online-ivector-extractor.h"
#include "kaldi_native_io/csrc/io-funcs.h"
#include "kaldi_native_io/csrc/kaldi-matrix.h"
#include "kaldi_native_io/csrc/kaldi-vector.h"

namespace khg {

namespace {

// kaldi ApproxEqual-style relative compare.
void ExpectEqual(double a, double b, double tol = 1.0e-6) {
  EXPECT_LE(std::fabs(a - b), 0.5 * tol * (std::fabs(a) + std::fabs(b)))
      << "a = " << a << ", b = " << b;
}

// Deterministic pseudo-random values in a realistic range.
double Frand(int32_t *seed) {
  *seed = *seed * 1103515245 + 12345;
  return static_cast<double>((*seed >> 8) % 2000 - 1000) / 100.0;
}

// A tiny extractor (D=3, S=2, I=4, no weight projection), written in kaldi's
// final.ie binary format and read back through the real reader.
IvectorExtractor MakeExtractor() {
  const int32_t D = 3, S = 2, I = 4;
  std::ostringstream os(std::ios::binary);
  kaldiio::WriteToken(os, true, "<IvectorExtractor>");
  kaldiio::WriteToken(os, true, "<w>");
  kaldiio::Matrix<double> w(0, 0);
  w.Write(os, true);
  kaldiio::WriteToken(os, true, "<w_vec>");
  kaldiio::Vector<double> w_vec(I);
  for (int32_t i = 0; i < I; i++) w_vec(i) = 1.0 / I;
  w_vec.Write(os, true);
  kaldiio::WriteToken(os, true, "<M>");
  kaldiio::WriteBasicType(os, true, I);
  int32_t seed = 7;
  for (int32_t i = 0; i < I; i++) {
    kaldiio::Matrix<double> m(D, S);
    for (int32_t d = 0; d < D; d++) {
      for (int32_t s = 0; s < S; s++) m(d, s) = Frand(&seed) / 10.0;
    }
    m.Write(os, true);
  }
  kaldiio::WriteToken(os, true, "<SigmaInv>");
  for (int32_t i = 0; i < I; i++) {
    // packed lower-triangular symmetric ("DP"), diagonal-dominant
    kaldiio::WriteToken(os, true, "DP");
    kaldiio::WriteBasicType(os, true, D);
    for (int32_t r = 0; r < D; r++) {
      for (int32_t c = 0; c <= r; c++) {
        double v = (r == c ? 1.0 + std::fabs(Frand(&seed)) / 5.0 : 0.05);
        os.write(reinterpret_cast<const char *>(&v), sizeof(double));
      }
    }
  }
  kaldiio::WriteToken(os, true, "<IvectorOffset>");
  kaldiio::WriteBasicType(os, true, 1.5);
  kaldiio::WriteToken(os, true, "</IvectorExtractor>");

  std::istringstream is(os.str(), std::ios::binary);
  IvectorExtractor extractor;
  extractor.Read(is, true);
  return extractor;
}

// Random feats + normalized sparse posteriors for one utterance.
void MakeData(int32_t T, int32_t D, int32_t I, int32_t *seed,
              FloatMatrix *feats, std::vector<GaussPost> *post) {
  *feats = FloatMatrix(T, D);
  post->resize(T);
  for (int32_t t = 0; t < T; t++) {
    double tot = 0.0;
    for (int32_t d = 0; d < D; d++) (*feats)(t, d) = Frand(seed);
    for (int32_t i = 0; i < I; i++) {
      double w = std::fabs(Frand(seed)) + 0.1;
      (*post)[t].emplace_back(i, w);
      tot += w;
    }
    for (auto &p : (*post)[t]) p.second /= tot;
  }
}

}  // namespace

TEST(IvectorExtractor, RoundTripDims) {
  IvectorExtractor e = MakeExtractor();
  EXPECT_EQ(e.FeatDim(), 3);
  EXPECT_EQ(e.IvectorDim(), 2);
  EXPECT_EQ(e.NumGauss(), 4);
  EXPECT_NEAR(e.PriorOffset(), 1.5, 1e-12);
  EXPECT_FALSE(e.IvectorDependentWeights());
}

TEST(IvectorExtractor, VectorToPosteriorEntryNormalizes) {
  FloatVector log_likes(4);
  log_likes << 1.0f, 2.0f, 0.5f, -1.0f;
  GaussPost post = VectorToPosteriorEntry(log_likes, 0, 0.0);
  ASSERT_EQ(post.size(), 4u);
  double tot = 0.0;
  for (const auto &p : post) tot += p.second;
  ExpectEqual(tot, 1.0, 1e-6);
  // sorted by weight, decreasing
  for (size_t i = 1; i < post.size(); i++) {
    EXPECT_LE(post[i].second, post[i - 1].second);
  }
}

TEST(IvectorExtractor, VectorToPosteriorEntryTopN) {
  FloatVector log_likes(4);
  log_likes << 1.0f, 2.0f, 0.5f, -1.0f;
  GaussPost post = VectorToPosteriorEntry(log_likes, 2, 0.0);
  ASSERT_EQ(post.size(), 2u);
  EXPECT_EQ(post[0].first, 1);  // largest posterior
  EXPECT_EQ(post[1].first, 0);
  double tot = 0.0;
  for (const auto &p : post) tot += p.second;
  ExpectEqual(tot, 1.0, 1e-6);
}

TEST(IvectorExtractor, VectorToPosteriorEntryMinPost) {
  FloatVector log_likes(3);
  log_likes << 0.0f, -10.0f, -11.0f;
  GaussPost post = VectorToPosteriorEntry(log_likes, 0, 0.05);
  // the two tiny posteriors are below 5% of the total and get pruned
  ASSERT_EQ(post.size(), 1u);
  EXPECT_EQ(post[0].first, 0);
  ExpectEqual(post[0].second, 1.0, 1e-6);
}

TEST(IvectorExtractor, LinearCgdMatchesExactSolve) {
  // randomized, kaldi-test style
  int32_t seed = 17;
  for (int32_t iter = 0; iter < 100; iter++) {
    int32_t n = 2 + iter % 7;
    DoubleMatrix a(n, n);
    for (int32_t i = 0; i < n; i++) {
      for (int32_t j = 0; j < n; j++) a(i, j) = Frand(&seed);
    }
    DoubleMatrix A = a.transpose() * a +
                     DoubleMatrix::Identity(n, n);  // SPD
    DoubleVector b(n);
    for (int32_t i = 0; i < n; i++) b(i) = Frand(&seed);
    DoubleVector x = DoubleVector::Zero(n);
    LinearCgdOptions opts;  // max_iters = -1: solve to convergence
    LinearCgd(opts, A, b, &x);
    DoubleVector exact = A.ldlt().solve(b);
    for (int32_t i = 0; i < n; i++) {
      EXPECT_NEAR(x(i), exact(i), 1e-6) << "iter=" << iter << " i=" << i;
    }
  }
}

TEST(IvectorExtractor, UtteranceStatsMatchOnlineStats) {
  // The batch stats + GetIvectorDistribution and the incremental
  // OnlineIvectorEstimationStats must give the same posterior mean.
  IvectorExtractor e = MakeExtractor();
  int32_t seed = 99;
  FloatMatrix feats;
  std::vector<GaussPost> post;
  MakeData(40, e.FeatDim(), e.NumGauss(), &seed, &feats, &post);

  IvectorExtractorUtteranceStats utt(e.NumGauss(), e.FeatDim());
  utt.AccStats(feats, post);
  DoubleVector mean(e.IvectorDim());
  e.GetIvectorDistribution(utt, &mean, nullptr);

  OnlineIvectorEstimationStats online(e.IvectorDim(), e.PriorOffset(), 0.0);
  online.AccStats(e, feats, post);
  DoubleVector iv(e.IvectorDim());
  online.GetIvector(-1, &iv);  // exact

  for (int32_t i = 0; i < e.IvectorDim(); i++) {
    EXPECT_NEAR(iv(i), mean(i), 1e-5) << "i=" << i;
  }
}

TEST(IvectorExtractor, OnlineIvectorWarmStartConverges) {
  IvectorExtractor e = MakeExtractor();
  int32_t seed = 5;
  FloatMatrix feats;
  std::vector<GaussPost> post;
  MakeData(100, e.FeatDim(), e.NumGauss(), &seed, &feats, &post);

  OnlineIvectorEstimationStats online(e.IvectorDim(), e.PriorOffset(), 0.0);
  online.AccStats(e, feats, post);
  DoubleVector warm = DoubleVector::Zero(e.IvectorDim());
  online.GetIvector(15, &warm);          // warm-started CG
  DoubleVector exact = DoubleVector::Zero(e.IvectorDim());
  online.GetIvector(-1, &exact);         // exact
  for (int32_t i = 0; i < e.IvectorDim(); i++) {
    EXPECT_NEAR(warm(i), exact(i), 1e-3) << "i=" << i;
  }
}

TEST(IvectorExtractor, MaxCountStrengthensPrior) {
  IvectorExtractor e = MakeExtractor();
  int32_t seed = 23;
  FloatMatrix feats;
  std::vector<GaussPost> post;
  MakeData(200, e.FeatDim(), e.NumGauss(), &seed, &feats, &post);

  OnlineIvectorEstimationStats unlimited(e.IvectorDim(), e.PriorOffset(), 0.0);
  unlimited.AccStats(e, feats, post);
  DoubleVector iv0 = DoubleVector::Zero(e.IvectorDim());
  unlimited.GetIvector(-1, &iv0);

  OnlineIvectorEstimationStats limited(e.IvectorDim(), e.PriorOffset(), 5.0);
  limited.AccStats(e, feats, post);
  DoubleVector iv1 = DoubleVector::Zero(e.IvectorDim());
  limited.GetIvector(-1, &iv1);

  // with max_count=5 the estimate is pulled back toward the prior mean
  // [prior_offset, 0, ...]
  DoubleVector prior = DoubleVector::Zero(e.IvectorDim());
  prior(0) = e.PriorOffset();
  EXPECT_LT((iv1 - prior).norm(), (iv0 - prior).norm());
}

TEST(IvectorExtractor, ObjfChangeNonNegative) {
  IvectorExtractor e = MakeExtractor();
  int32_t seed = 3;
  FloatMatrix feats;
  std::vector<GaussPost> post;
  MakeData(50, e.FeatDim(), e.NumGauss(), &seed, &feats, &post);
  OnlineIvectorEstimationStats online(e.IvectorDim(), e.PriorOffset(), 0.0);
  online.AccStats(e, feats, post);
  DoubleVector iv(e.IvectorDim());
  online.GetIvector(-1, &iv);
  EXPECT_GE(online.ObjfChange(iv), 0.0);
}

TEST(OnlineIvectorExtractor, CadenceAndShapes) {
  IvectorExtractor e = MakeExtractor();
  int32_t seed = 11;
  const int32_t D = e.FeatDim();
  // lda selects the center frame of the splice window (L=R=1)
  FloatMatrix lda = FloatMatrix::Zero(D, D * 3);
  for (int32_t d = 0; d < D; d++) lda(d, D + d) = 1.0f;
  DoubleMatrix glob = DoubleMatrix::Zero(2, D + 1);
  glob(0, D) = 100.0;

  const int32_t I = e.NumGauss();
  FloatVector weights(I);
  FloatMatrix inv_vars(I, D), means_invvars(I, D);
  for (int32_t i = 0; i < I; i++) {
    weights(i) = 1.0f / I;
    for (int32_t d = 0; d < D; d++) {
      inv_vars(i, d) = 1.0f;
      means_invvars(i, d) = Frand(&seed);
    }
  }
  DiagGmm ubm(weights, inv_vars, means_invvars);

  OnlineIvectorExtractorConfig cfg;
  cfg.left_context = 1;
  cfg.right_context = 1;
  cfg.lda_mat = lda;
  cfg.global_cmvn_stats = glob;
  cfg.ivector_period = 10;
  OnlineIvectorExtractor engine(cfg, e, ubm);

  FloatMatrix feats(25, D);
  for (int32_t t = 0; t < 25; t++) {
    for (int32_t d = 0; d < D; d++) feats(t, d) = Frand(&seed);
  }
  engine.AcceptFeatures(feats);
  engine.Flush();

  EXPECT_EQ(engine.NumFrames(), 25);
  // rows at frames 0, 10, 20
  EXPECT_EQ(engine.NumIvectorsReady(), 3);
  DoubleMatrix ivs = engine.GetIvectors();
  EXPECT_EQ(ivs.rows(), 3);
  EXPECT_EQ(ivs.cols(), e.IvectorDim());
  for (int32_t t = 0; t < ivs.rows(); t++) {
    for (int32_t i = 0; i < ivs.cols(); i++) {
      ASSERT_TRUE(std::isfinite(ivs(t, i)));
    }
  }
  EXPECT_EQ(engine.LatestIvector().size(), e.IvectorDim());
  EXPECT_GT(engine.Count(), 0.0);
}

}  // namespace khg
