// kaldi-hmm-gmm/kaldi-hmm-gmm/csrc/ivector-extractor.h
//
// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// See LICENSE for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_HMM_GMM_CSRC_IVECTOR_EXTRACTOR_H_
#define KALDI_HMM_GMM_CSRC_IVECTOR_EXTRACTOR_H_

#include <cstdint>
#include <istream>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/eigen.h"

// Runtime subset of kaldi's ivector/ivector-extractor.{h,cc}: what is needed
// to load a trained extractor (final.ie) and estimate iVectors from features
// plus Gaussian-level posteriors.  Training-related classes (extractor
// estimation, auxf diagnostics, TransformIvectors) are intentionally not
// ported.  Numerics follow kaldi exactly.

namespace khg {

// Sparse Gaussian posterior for one frame: (gaussian-index, weight) pairs.
using GaussPost = std::vector<std::pair<int32_t, float> >;

/// Convert log-likes to pruned, normalized Gaussian posteriors (kaldi's
/// VectorToPosteriorEntry in hmm/posterior.cc).  num_gselect <= 0 means
/// "keep all Gaussians"; trailing posteriors below min_post * (sum of kept)
/// are dropped before renormalizing.
GaussPost VectorToPosteriorEntry(const FloatVector &log_likes,
                                 int32_t num_gselect, double min_post);

/// Sufficient stats for estimating an iVector of one utterance
/// (kaldi's IvectorExtractorUtteranceStats, 1st-order stats only).
class IvectorExtractorUtteranceStats {
 public:
  IvectorExtractorUtteranceStats(int32_t num_gauss, int32_t feat_dim)
      : gamma_(num_gauss), X_(num_gauss, feat_dim) {
    gamma_.setZero();
    X_.setZero();
  }

  /// Accumulate stats for a sequence of frames (rows) with per-frame
  /// Gaussian posteriors.
  void AccStats(const FloatMatrix &feats, const std::vector<GaussPost> &post);

  void Scale(double scale);

  double NumFrames() const { return gamma_.sum(); }

 protected:
  friend class IvectorExtractor;
  DoubleVector gamma_;  // zeroth-order stats [I]
  DoubleMatrix X_;      // first-order stats [I][D]
};

/// A trained i-vector extractor (final.ie).
class IvectorExtractor {
 public:
  IvectorExtractor() : prior_offset_(0.0) {}

  /// Read from kaldi's binary/text format (final.ie).
  void Read(std::istream &is, bool binary);

  /// Distribution over iVectors (Gaussian approximation).  "var" may be
  /// nullptr; if present it is set to the inverse of the quadratic term.
  void GetIvectorDistribution(const IvectorExtractorUtteranceStats &utt_stats,
                              DoubleVector *mean,
                              DoubleMatrix *var) const;

  double PriorOffset() const { return prior_offset_; }
  int32_t FeatDim() const { return M_.empty() ? 0 : M_[0].rows(); }
  int32_t IvectorDim() const { return M_.empty() ? 0 : M_[0].cols(); }
  int32_t NumGauss() const { return static_cast<int32_t>(M_.size()); }
  bool IvectorDependentWeights() const { return w_.rows() != 0; }

  // Packed-data accessors used by the stats accumulators.
  const std::vector<DoubleMatrix> &SigmaInvM() const { return Sigma_inv_M_; }
  const std::vector<DoubleMatrix> &U() const { return U_; }

 protected:
  void ComputeDerivedVars();
  void ComputeDerivedVars(int32_t i);
  void GetIvectorDistMean(const IvectorExtractorUtteranceStats &utt_stats,
                          DoubleVector *linear, DoubleMatrix *quadratic) const;
  void GetIvectorDistPrior(const IvectorExtractorUtteranceStats &utt_stats,
                           DoubleVector *linear, DoubleMatrix *quadratic) const;
  void GetIvectorDistWeight(const IvectorExtractorUtteranceStats &utt_stats,
                            const DoubleVector &mean, DoubleVector *linear,
                            DoubleMatrix *quadratic) const;

  // var <-- quadratic_term^{-1}, eigenvalues floored to 1.0 first.
  static void InvertWithFlooring(const DoubleMatrix &quadratic_term,
                                 DoubleMatrix *var);

  DoubleMatrix w_;     // [I][S] weight projection, or empty
  DoubleVector w_vec_; // [I] UBM weights if no w_
  std::vector<DoubleMatrix> M_;          // [I] of [D][S]
  std::vector<DoubleMatrix> Sigma_inv_;  // [I] of [D][D] (symmetric)
  double prior_offset_;

  // Derived variables:
  std::vector<DoubleMatrix> U_;          // [I] of [S][S] = M^T Sigma^-1 M
  std::vector<DoubleMatrix> Sigma_inv_M_;  // [I] of [D][S]
};

/// Incremental i-vector estimation as frames arrive (kaldi's
/// OnlineIvectorEstimationStats).  Handles the max_count prior scaling.
class OnlineIvectorEstimationStats {
 public:
  OnlineIvectorEstimationStats(int32_t ivector_dim, double prior_offset,
                               double max_count);

  /// Accumulate stats for one frame (sparse posteriors).
  void AccStats(const IvectorExtractor &extractor, const FloatVector &feature,
                const GaussPost &gauss_post);

  /// Accumulate stats for a sequence of frames.
  void AccStats(const IvectorExtractor &extractor, const FloatMatrix &features,
                const std::vector<GaussPost> &gauss_post);

  /// Current iVector estimate.  If num_cg_iters > 0, warm-started conjugate
  /// gradient with that many iterations (pass a recently estimated iVector in
  /// *ivector for a better starting point); if negative, solve exactly.
  void GetIvector(int32_t num_cg_iters, DoubleVector *ivector) const;

  int32_t IvectorDim() const { return linear_term_.size(); }
  double Count() const { return num_frames_; }
  double PriorOffset() const { return prior_offset_; }
  double ObjfChange(const DoubleVector &ivector) const;

  void Scale(double scale);

  // State access (for save/restore across utterances).
  const DoubleVector &LinearTerm() const { return linear_term_; }
  const DoubleMatrix &QuadraticTerm() const { return quadratic_term_; }
  double NumFrames() const { return num_frames_; }

 protected:
  double Objf(const DoubleVector &ivector) const;
  double DefaultObjf() const;

  double prior_offset_;
  double max_count_;
  double num_frames_;
  DoubleMatrix quadratic_term_;  // [S][S] (symmetric)
  DoubleVector linear_term_;     // [S]
};

/// Options for LinearCgd (kaldi's matrix/optimization.h).
struct LinearCgdOptions {
  int32_t max_iters = -1;
  double max_error = 0.0;
  double recompute_residual_factor = 0.01;
};

/// Conjugate gradient for A x = b (A symmetric positive definite), a port of
/// kaldi's LinearCgd (matrix/optimization.cc).  "x" is warm-started in place.
/// Returns the number of iterations performed.
int32_t LinearCgd(const LinearCgdOptions &opts, const DoubleMatrix &A,
                  const DoubleVector &b, DoubleVector *x);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_IVECTOR_EXTRACTOR_H_
