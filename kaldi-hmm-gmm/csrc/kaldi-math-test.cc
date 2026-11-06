// kaldi-hmm-gmm/csrc/kaldi-math-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Regression tests for the deterministic random-number stream
// (Randn/RandnVector/RandnMatrix).  Baseline: ../kaldi/src/base/kaldi-math.cc
// (RandGauss2/RandUniform) — Box-Muller over the global rand() stream, i.e.
// reproducible given the same srand() sequence.  A previous implementation
// used std::random_device (true randomness), which made training runs
// irreproducible and diverge from kaldi; these tests pin the kaldi semantics.

#include "kaldi-hmm-gmm/csrc/eigen.h"

#include <cmath>
#include <cstdlib>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"

namespace khg {

TEST(Randn, DeterministicGivenSeed) {
  srand(42);
  float a = Randn(0.0f, 1.0f);
  float v = RandnVector(3, 0.0f, 1.0f).norm();
  srand(42);
  float b = Randn(0.0f, 1.0f);
  float v2 = RandnVector(3, 0.0f, 1.0f).norm();
  EXPECT_FLOAT_EQ(a, b);
  EXPECT_FLOAT_EQ(v, v2);
}

TEST(Randn, MatchesKaldiRandGauss2Formula) {
  // kaldi: u1,u2 = (Rand()+1)/(RAND_MAX+2); z = sqrt(-2 ln u1) * cos(2 pi u2)
  srand(7);
  float got = Randn(0.0f, 1.0f);
  srand(7);
  float u1 = (Rand() + 1.0f) / (RAND_MAX + 2.0f);
  float u2 = (Rand() + 1.0f) / (RAND_MAX + 2.0f);
  float expect = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
  EXPECT_FLOAT_EQ(got, expect);

  // shifted/scaled form
  srand(7);
  float got2 = Randn(5.0f, 2.0f);
  EXPECT_FLOAT_EQ(got2, 5.0f + 2.0f * expect);
}

TEST(Randn, RoughlyStandardNormal) {
  srand(1);
  const int n = 20000;
  double sum = 0.0, sumsq = 0.0;
  for (int i = 0; i < n; ++i) {
    float x = Randn(0.0f, 1.0f);
    sum += x;
    sumsq += x * x;
  }
  double mean = sum / n;
  double var = sumsq / n - mean * mean;
  EXPECT_NEAR(mean, 0.0, 0.05);
  EXPECT_NEAR(var, 1.0, 0.05);
}

TEST(DiagGmmSplit, DeterministicGivenSeed) {
  // Split/Perturb consume Randn; they must be reproducible too (this was the
  // user-visible symptom: training was not reproducible run-to-run).
  auto build = [](DiagGmm *g) {
    FloatVector w(2);
    w << 0.5f, 0.5f;
    FloatMatrix m(2, 3);
    m << 0, 0, 0, 1, 1, 1;
    FloatMatrix v(2, 3);
    v.setConstant(1.0f);
    g->SetWeights(w);
    g->SetMeans(m);
    g->SetInvVars(v);
    g->ComputeGconsts();
  };

  DiagGmm g1(2, 3), g2(2, 3);
  build(&g1);
  build(&g2);
  srand(99);
  g1.Split(3, 0.01f);
  srand(99);
  g2.Split(3, 0.01f);
  ASSERT_EQ(g1.NumGauss(), g2.NumGauss());
  EXPECT_TRUE(g1.weights().isApprox(g2.weights()));
  EXPECT_TRUE(g1.means_invvars().isApprox(g2.means_invvars()));

  DiagGmm p1(2, 3), p2(2, 3);
  build(&p1);
  build(&p2);
  srand(5);
  p1.Perturb(0.1f);
  srand(5);
  p2.Perturb(0.1f);
  EXPECT_TRUE(p1.means_invvars().isApprox(p2.means_invvars()));
}

}  // namespace khg

// ---------------------------------------------------------------------------
// Ported from ../kaldi/src/base/kaldi-math.{h,cc}: RandUniform range,
// LogAdd, ApproxEqual.
// ---------------------------------------------------------------------------

namespace khg {

TEST(KaldiMath, RandUniformRange) {
  srand(1);
  const int32_t n = 20000;
  double sum = 0.0;
  for (int32_t i = 0; i < n; ++i) {
    float x = RandUniform();
    // RandUniform is in the open interval (0, 1)
    EXPECT_GT(x, 0.0f);
    EXPECT_LT(x, 1.0f);
    sum += x;
  }
  // large-sample mean of a uniform variable on (0, 1) is 0.5
  EXPECT_NEAR(sum / n, 0.5, 0.01);
}

TEST(KaldiMath, LogAdd) {
  srand(1);
  // close magnitudes: LogAdd must match log(exp(a) + exp(b))
  {
    float a = 0.5f;
    float b = -0.5f;
    double expected = std::log(std::exp(static_cast<double>(a)) +
                               std::exp(static_cast<double>(b)));
    EXPECT_NEAR(static_cast<double>(LogAdd(a, b)), expected, 1.0e-5);
    EXPECT_NEAR(static_cast<double>(LogAdd(b, a)), expected, 1.0e-5);
    // exactly symmetric in float
    EXPECT_FLOAT_EQ(LogAdd(a, b), LogAdd(b, a));
  }
  {
    float a = -2.0f;
    float b = -2.3f;
    double expected = std::log(std::exp(static_cast<double>(a)) +
                               std::exp(static_cast<double>(b)));
    EXPECT_NEAR(static_cast<double>(LogAdd(a, b)), expected, 1.0e-5);
  }
  // equal arguments give a + log(2)
  {
    float a = 1.0f;
    EXPECT_NEAR(static_cast<double>(LogAdd(a, a)),
                1.0 + std::log(2.0), 1.0e-5);
  }
  // far apart (beyond kMinLogDiff): kaldi's LogAdd returns the larger one
  {
    float a = 0.0f;
    float b = -100.0f;
    EXPECT_FLOAT_EQ(LogAdd(a, b), a);
    EXPECT_FLOAT_EQ(LogAdd(b, a), a);
    float c = 5.0f;
    float d = -500.0f;
    EXPECT_FLOAT_EQ(LogAdd(c, d), c);
    EXPECT_FLOAT_EQ(LogAdd(d, c), c);
  }
}

TEST(KaldiMath, ApproxEqual) {
  srand(1);
  float inf = INFINITY;
  float nan = NAN;
  float tol = 0.001f;

  // a == b is always equal, and it also handles the infinities
  EXPECT_TRUE(ApproxEqual(0.0f, 0.0f));
  EXPECT_TRUE(ApproxEqual(3.25f, 3.25f));
  EXPECT_TRUE(ApproxEqual(inf, inf));
  EXPECT_TRUE(ApproxEqual(-inf, -inf));
  // NaN is never equal, not even to itself
  EXPECT_FALSE(ApproxEqual(nan, nan));
  EXPECT_FALSE(ApproxEqual(nan, 1.0f));
  // infinite vs finite
  EXPECT_FALSE(ApproxEqual(inf, 1.0f));
  EXPECT_FALSE(ApproxEqual(inf, -inf));

  // tolerance boundary: equal iff |a-b| <= tol * (|a|+|b|)
  EXPECT_TRUE(ApproxEqual(1.0f, 1.0019f, tol));   // 0.0019 <= 0.001 * 2.0019
  EXPECT_FALSE(ApproxEqual(1.0f, 1.0021f, tol));  // 0.0021 >  0.001 * 2.0021
  EXPECT_TRUE(ApproxEqual(-2.0f, -2.0039f, tol));  // 0.0039 <= 0.001 * 4.0039
  EXPECT_FALSE(ApproxEqual(0.5f, 0.5011f, tol));  // 0.0011 >  0.001 * 1.0011

  // symmetry over a spread of pairs, including specials
  const float vals[] = {0.0f, 1.0f, -1.0f, 1.0e-3f,
                        -7.5f, 100.0f, inf, -inf, nan};
  const int32_t num_vals = sizeof(vals) / sizeof(vals[0]);
  for (int32_t i = 0; i < num_vals; ++i) {
    for (int32_t j = 0; j < num_vals; ++j) {
      EXPECT_EQ(ApproxEqual(vals[i], vals[j]), ApproxEqual(vals[j], vals[i]));
    }
  }
}

}  // namespace khg
