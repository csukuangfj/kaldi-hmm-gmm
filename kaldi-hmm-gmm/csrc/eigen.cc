// kaldi-hmm-gmm/csrc/eigen.cc

// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/eigen.h"

#include <cmath>

#include "kaldi-hmm-gmm/csrc/kaldi-math.h"

namespace khg {

// kaldi VectorBase::LogSumExp semantics (../kaldi/src/matrix/kaldi-vector.cc):
// terms below max + kMinLogDiffFloat are dropped (or below max - prune when
// prune > 0), the rest is accumulated in double:
// log(sum(exp(x))) == log(sum(exp(x - max(x)))) + max(x)
float LogSumExp(const FloatVector &v, float prune /*= -1.0f*/) {
  float max_v = v.maxCoeff();

  float cutoff = max_v + kMinLogDiffFloat;
  if (prune > 0.0f && max_v - prune > cutoff) {
    cutoff = max_v - prune;
  }

  double sum_relto_max =
      (v.array() >= cutoff)
          .select((v.array() - max_v).exp(), 0.0f)
          .cast<double>()
          .sum();

  return max_v + std::log(sum_relto_max);
}

FloatVector Softmax(const FloatVector &v, float *log_sum_exp /*= nullptr*/) {
  // kaldi VectorBase::ApplySoftMax semantics: subtract max, exp, normalize
  // by the sum; log_sum_exp = max + log(sum).
  float max_v = v.maxCoeff();
  FloatVector ans = (v.array() - max_v).exp();

  float ans_sum = ans.sum();
  if (log_sum_exp) {
    *log_sum_exp = std::log(ans_sum) + max_v;
  }
  return ans / ans_sum;
}

// The functions below follow kaldi's RandGauss2/RandUniform semantics
// (../kaldi/src/base/kaldi-math.cc): Box-Muller over the global rand()
// stream, so results are deterministic/reproducible given the same srand()
// sequence.  (The previous implementation used std::random_device, i.e. true
// randomness, which made runs irreproducible and diverge from kaldi.)

static void RandGauss2Local(float *a, float *b) {
  float u1 = RandUniform();
  float u2 = RandUniform();
  u1 = sqrtf(-2.0f * logf(u1));
  u2 = 2.0f * kPi * u2;
  *a = u1 * cosf(u2);
  *b = u1 * sinf(u2);
}

float Randn(float mean /*= 0*/, float stddev /*= 1*/) {
  float a, b;
  RandGauss2Local(&a, &b);
  return mean + stddev * a;
}

FloatVector RandnVector(int32_t n, float mean /*= 0*/, float stddev /*= 1*/) {
  FloatVector ans(n);
  for (int32_t i = 0; i != n; ++i) ans[i] = Randn(mean, stddev);
  return ans;
}

FloatMatrix RandnMatrix(int32_t rows, int32_t cols, float mean /*= 0*/,
                        float stddev /*= 1*/) {
  // Fill row-major, matching kaldi's (i, d) nested loops (see the rand_mat
  // fills in ../kaldi/src/gmm/diag-gmm.cc); Eigen's linear index is
  // column-major, which would assign draws to different elements.
  FloatMatrix ans(rows, cols);
  for (int32_t i = 0; i != rows; ++i)
    for (int32_t d = 0; d != cols; ++d) ans(i, d) = Randn(mean, stddev);
  return ans;
}

}  // namespace khg
