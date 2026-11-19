// kaldi-hmm-gmm/kaldi-hmm-gmm/csrc/ivector-extractor.cc
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

#include "kaldi-hmm-gmm/csrc/ivector-extractor.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

#include "Eigen/Cholesky"
#include "Eigen/Eigenvalues"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi_native_io/csrc/io-funcs.h"
#include "kaldi_native_io/csrc/kaldi-matrix.h"
#include "kaldi_native_io/csrc/kaldi-vector.h"

namespace khg {

namespace {

DoubleMatrix ToEigen(const kaldiio::Matrix<double> &m) {
  DoubleMatrix out(m.NumRows(), m.NumCols());
  for (int32_t r = 0; r < m.NumRows(); ++r)
    for (int32_t c = 0; c < m.NumCols(); ++c) out(r, c) = m(r, c);
  return out;
}

DoubleVector ToEigen(const kaldiio::Vector<double> &v) {
  DoubleVector out(v.Dim());
  for (int32_t i = 0; i < v.Dim(); ++i) out(i) = v(i);
  return out;
}

// kaldi's PackedMatrix<double> ("DP"): int32 rows, then rows*(rows+1)/2
// doubles, lower-triangular row-major ((r*(r+1))/2 + c, c <= r).  Unpacks
// into a full symmetric matrix.  (kaldi_native_io has no packed support.)
DoubleMatrix ReadPackedMatrix(std::istream &is, bool binary) {
  if (!binary) {
    KHG_ERR << "Text-mode packed matrices are not supported";
  }
  kaldiio::ExpectToken(is, binary, "DP");
  int32_t rows;
  kaldiio::ReadBasicType(is, binary, &rows);
  std::vector<double> packed(static_cast<size_t>(rows) * (rows + 1) / 2);
  is.read(reinterpret_cast<char *>(packed.data()),
          sizeof(double) * packed.size());
  KHG_ASSERT(is.good());  // failed to read packed matrix data
  DoubleMatrix out = DoubleMatrix::Zero(rows, rows);
  for (int32_t r = 0; r < rows; r++) {
    for (int32_t c = 0; c <= r; c++) {
      double v = packed[(r * (r + 1)) / 2 + c];
      out(r, c) = v;
      out(c, r) = v;
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------
// IvectorExtractorUtteranceStats
// ---------------------------------------------------------------------

void IvectorExtractorUtteranceStats::AccStats(
    const FloatMatrix &feats, const std::vector<GaussPost> &post) {
  int32_t num_frames = feats.rows();
  int32_t num_gauss = X_.rows(), feat_dim = feats.cols();
  KHG_ASSERT(X_.cols() == feat_dim);
  KHG_ASSERT(static_cast<int32_t>(post.size()) == num_frames);
  for (int32_t t = 0; t < num_frames; t++) {
    auto frame = feats.row(t).cast<double>();
    const GaussPost &this_post(post[t]);
    for (const auto &pair : this_post) {
      int32_t i = pair.first;
      double weight = pair.second;
      KHG_ASSERT(i >= 0 && i < num_gauss);
      gamma_(i) += weight;
      X_.row(i) += weight * frame;
    }
  }
}

void IvectorExtractorUtteranceStats::Scale(double scale) {
  gamma_ *= scale;
  X_ *= scale;
}

// ---------------------------------------------------------------------
// IvectorExtractor
// ---------------------------------------------------------------------

void IvectorExtractor::ComputeDerivedVars() {
  U_.resize(NumGauss());
  Sigma_inv_M_.resize(NumGauss());
  for (int32_t i = 0; i < NumGauss(); i++) ComputeDerivedVars(i);
}

void IvectorExtractor::ComputeDerivedVars(int32_t i) {
  // U_i = M_i^T Sigma_i^{-1} M_i
  U_[i].noalias() = M_[i].transpose() * Sigma_inv_[i] * M_[i];
  // (Sigma^{-1} M)_i
  Sigma_inv_M_[i].noalias() = Sigma_inv_[i] * M_[i];
}

// static
void IvectorExtractor::InvertWithFlooring(const DoubleMatrix &quadratic_term,
                                          DoubleMatrix *var) {
  Eigen::SelfAdjointEigenSolver<DoubleMatrix> eig(quadratic_term);
  DoubleVector s = eig.eigenvalues();
  for (int32_t i = 0; i < s.size(); i++) {
    if (s(i) < 1.0) s(i) = 1.0;
    s(i) = 1.0 / s(i);
  }
  *var = eig.eigenvectors() * s.asDiagonal() * eig.eigenvectors().transpose();
}

void IvectorExtractor::GetIvectorDistribution(
    const IvectorExtractorUtteranceStats &utt_stats, DoubleVector *mean,
    DoubleMatrix *var) const {
  int32_t S = IvectorDim();
  if (!IvectorDependentWeights()) {
    DoubleVector linear = DoubleVector::Zero(S);
    DoubleMatrix quadratic = DoubleMatrix::Zero(S, S);
    GetIvectorDistMean(utt_stats, &linear, &quadratic);
    GetIvectorDistPrior(utt_stats, &linear, &quadratic);
    if (var != nullptr) {
      *var = quadratic.inverse();
      mean->noalias() = (*var) * linear;
    } else {
      *mean = quadratic.ldlt().solve(linear);
    }
  } else {
    DoubleVector linear = DoubleVector::Zero(S);
    DoubleMatrix quadratic = DoubleMatrix::Zero(S, S);
    GetIvectorDistMean(utt_stats, &linear, &quadratic);
    GetIvectorDistPrior(utt_stats, &linear, &quadratic);

    DoubleVector cur_mean(S);
    DoubleMatrix quadratic_inv(S, S);
    InvertWithFlooring(quadratic, &quadratic_inv);
    cur_mean = quadratic_inv * linear;

    // Successively better expansion points for the quadratic approximation
    // of the weights (kaldi: 4 iters, abort if the change is < 0.1).
    const int32_t num_iters = 4;
    const double change_threshold = 0.1;
    for (int32_t iter = 0; iter < num_iters; iter++) {
      DoubleVector this_linear = linear;
      DoubleMatrix this_quadratic = quadratic;
      GetIvectorDistWeight(utt_stats, cur_mean, &this_linear, &this_quadratic);
      InvertWithFlooring(this_quadratic, &quadratic_inv);
      DoubleVector mean_diff = cur_mean;
      cur_mean = quadratic_inv * this_linear;
      mean_diff -= cur_mean;
      double change = mean_diff.norm();
      if (change < change_threshold) break;
    }
    *mean = cur_mean;
    if (var != nullptr) *var = quadratic_inv;
  }
}

void IvectorExtractor::GetIvectorDistMean(
    const IvectorExtractorUtteranceStats &utt_stats, DoubleVector *linear,
    DoubleMatrix *quadratic) const {
  int32_t I = NumGauss();
  for (int32_t i = 0; i < I; i++) {
    double gamma = utt_stats.gamma_(i);
    if (gamma != 0.0) {
      // linear += Sigma_inv_M_i^T x_i   (x_i = gamma_i * mean_i)
      *linear += Sigma_inv_M_[i].transpose() * utt_stats.X_.row(i).transpose();
    }
    if (gamma != 0.0) *quadratic += gamma * U_[i];
  }
}

void IvectorExtractor::GetIvectorDistPrior(
    const IvectorExtractorUtteranceStats & /*utt_stats*/, DoubleVector *linear,
    DoubleMatrix *quadratic) const {
  (*linear)(0) += prior_offset_;  // the zero'th dim has an offset mean
  quadratic->diagonal().array() += 1.0;  // unit prior variance
}

void IvectorExtractor::GetIvectorDistWeight(
    const IvectorExtractorUtteranceStats &utt_stats, const DoubleVector &mean,
    DoubleVector *linear, DoubleMatrix *quadratic) const {
  if (!IvectorDependentWeights()) return;

  DoubleVector logw_unnorm = w_ * mean;  // [I]
  DoubleVector w = logw_unnorm;
  // softmax
  double max_w = w.maxCoeff();
  w = (w.array() - max_w).exp();
  w /= w.sum();

  // See eq.58 in the SGMM paper.
  double gamma = utt_stats.gamma_.sum();
  DoubleVector linear_coeff(NumGauss()), quadratic_coeff(NumGauss());
  for (int32_t i = 0; i < NumGauss(); i++) {
    double gamma_i = utt_stats.gamma_(i);
    double max_term = std::max(gamma_i, gamma * w(i));
    linear_coeff(i) = gamma_i - gamma * w(i) + max_term * logw_unnorm(i);
    quadratic_coeff(i) = max_term;
  }
  *linear += w_.transpose() * linear_coeff;
  for (int32_t i = 0; i < NumGauss(); i++) {
    if (quadratic_coeff(i) != 0.0)
      *quadratic += quadratic_coeff(i) * w_.row(i).transpose() *
                    w_.row(i);
  }
}

void IvectorExtractor::Read(std::istream &is, bool binary) {
  kaldiio::ExpectToken(is, binary, "<IvectorExtractor>");
  kaldiio::ExpectToken(is, binary, "<w>");
  {
    kaldiio::Matrix<double> w;
    w.Read(is, binary);
    w_ = ToEigen(w);
  }
  kaldiio::ExpectToken(is, binary, "<w_vec>");
  {
    kaldiio::Vector<double> w_vec;
    w_vec.Read(is, binary);
    w_vec_ = ToEigen(w_vec);
  }
  kaldiio::ExpectToken(is, binary, "<M>");
  int32_t size;
  kaldiio::ReadBasicType(is, binary, &size);
  KHG_ASSERT(size > 0);
  M_.resize(size);
  for (int32_t i = 0; i < size; i++) {
    kaldiio::Matrix<double> m;
    m.Read(is, binary);
    M_[i] = ToEigen(m);
  }
  kaldiio::ExpectToken(is, binary, "<SigmaInv>");
  Sigma_inv_.resize(size);
  for (int32_t i = 0; i < size; i++) {
    Sigma_inv_[i] = ReadPackedMatrix(is, binary);
  }
  kaldiio::ExpectToken(is, binary, "<IvectorOffset>");
  kaldiio::ReadBasicType(is, binary, &prior_offset_);
  kaldiio::ExpectToken(is, binary, "</IvectorExtractor>");
  ComputeDerivedVars();
}

// ---------------------------------------------------------------------
// OnlineIvectorEstimationStats
// ---------------------------------------------------------------------

OnlineIvectorEstimationStats::OnlineIvectorEstimationStats(
    int32_t ivector_dim, double prior_offset, double max_count)
    : prior_offset_(prior_offset),
      max_count_(max_count),
      num_frames_(0.0),
      quadratic_term_(DoubleMatrix::Zero(ivector_dim, ivector_dim)),
      linear_term_(DoubleVector::Zero(ivector_dim)) {
  if (ivector_dim != 0) {
    linear_term_(0) += prior_offset;
    quadratic_term_.diagonal().array() += 1.0;
  }
}

void OnlineIvectorEstimationStats::AccStats(const IvectorExtractor &extractor,
                                            const FloatVector &feature,
                                            const GaussPost &gauss_post) {
  KHG_ASSERT(extractor.IvectorDim() == static_cast<int32_t>(linear_term_.size()));
  KHG_ASSERT(!extractor.IvectorDependentWeights());

  DoubleVector feature_dbl = feature.cast<double>();
  double tot_weight = 0.0;

  for (const auto &pair : gauss_post) {
    int32_t g = pair.first;
    double weight = pair.second;
    // negative weights are allowed (kaldi needs them for silence subtraction)
    if (weight == 0.0) continue;
    linear_term_.noalias() +=
        weight * (extractor.SigmaInvM()[g].transpose() * feature_dbl);
    quadratic_term_ += weight * extractor.U()[g];
    tot_weight += weight;
  }
  if (max_count_ > 0.0) {
    double old_num_frames = num_frames_,
           new_num_frames = num_frames_ + tot_weight;
    double old_prior_scale = std::max(old_num_frames, max_count_) / max_count_,
           new_prior_scale = std::max(new_num_frames, max_count_) / max_count_;
    double prior_scale_change = new_prior_scale - old_prior_scale;
    if (prior_scale_change != 0.0) {
      linear_term_(0) += prior_offset_ * prior_scale_change;
      quadratic_term_.diagonal().array() += prior_scale_change;
    }
  }
  num_frames_ += tot_weight;
}

void OnlineIvectorEstimationStats::AccStats(
    const IvectorExtractor &extractor, const FloatMatrix &features,
    const std::vector<GaussPost> &gauss_post) {
  KHG_ASSERT(static_cast<int32_t>(gauss_post.size()) == features.rows());
  for (int32_t t = 0; t < features.rows(); t++) {
    FloatVector frame = features.row(t).transpose();
    AccStats(extractor, frame, gauss_post[t]);
  }
}

void OnlineIvectorEstimationStats::Scale(double scale) {
  KHG_ASSERT(scale >= 0.0 && scale <= 1.0);
  double old_num_frames = num_frames_;
  num_frames_ *= scale;
  quadratic_term_ *= scale;
  linear_term_ *= scale;

  if (max_count_ == 0.0) {
    linear_term_(0) += prior_offset_ * (1.0 - scale);
    quadratic_term_.diagonal().array() += (1.0 - scale);
  } else {
    double new_num_frames = num_frames_;
    double old_prior_scale =
               scale * std::max(old_num_frames, max_count_) / max_count_,
           new_prior_scale = std::max(new_num_frames, max_count_) / max_count_;
    linear_term_(0) += prior_offset_ * (new_prior_scale - old_prior_scale);
    quadratic_term_.diagonal().array() +=
        (new_prior_scale - old_prior_scale);
  }
}

void OnlineIvectorEstimationStats::GetIvector(int32_t num_cg_iters,
                                              DoubleVector *ivector) const {
  KHG_ASSERT(static_cast<int32_t>(ivector->size()) == IvectorDim());
  if (num_frames_ > 0.0) {
    if ((*ivector)(0) == 0.0) (*ivector)(0) = prior_offset_;  // initial guess
    LinearCgdOptions opts;
    opts.max_iters = num_cg_iters;
    LinearCgd(opts, quadratic_term_, linear_term_, ivector);
  } else {
    ivector->setZero();
    (*ivector)(0) = prior_offset_;
  }
}

double OnlineIvectorEstimationStats::ObjfChange(const DoubleVector &ivector) const {
  return Objf(ivector) - DefaultObjf();
}

double OnlineIvectorEstimationStats::Objf(const DoubleVector &ivector) const {
  if (num_frames_ == 0.0) return 0.0;
  return (1.0 / num_frames_) *
         (-0.5 * ivector.dot(quadratic_term_ * ivector) +
          ivector.dot(linear_term_));
}

double OnlineIvectorEstimationStats::DefaultObjf() const {
  if (num_frames_ == 0.0) return 0.0;
  double x = prior_offset_;
  return (1.0 / num_frames_) *
         (-0.5 * quadratic_term_(0, 0) * x * x + x * linear_term_(0));
}

// ---------------------------------------------------------------------
// LinearCgd (kaldi's matrix/optimization.cc)
// ---------------------------------------------------------------------

int32_t LinearCgd(const LinearCgdOptions &opts, const DoubleMatrix &A,
                  const DoubleVector &b, DoubleVector *x) {
  int32_t M = A.cols();

  DoubleVector r(M), p(M), Ap(M), x_orig(M);
  p = b - A * (*x);  // p_0 = b - A x_0
  r = -p;            // r_0 = -p_0
  x_orig = *x;

  double r_cur_norm_sq = r.squaredNorm(),
         r_initial_norm_sq = r_cur_norm_sq,
         r_recompute_norm_sq = r_cur_norm_sq;

  KHG_ASSERT(opts.recompute_residual_factor <= 1.0);
  double max_error_sq =
             std::max(opts.max_error * opts.max_error, 1.0e-300),
         residual_factor =
             opts.recompute_residual_factor * opts.recompute_residual_factor,
         inv_residual_factor = 1.0 / residual_factor;

  int32_t k = 0;
  for (; k < M + 5 && k != opts.max_iters; k++) {
    Ap = A * p;

    // alpha_k = (-p^T r) / (p^T A p)  (kaldi's robust variant)
    double alpha = -p.dot(r) / p.dot(Ap);

    *x += alpha * p;
    r += alpha * Ap;
    double r_next_norm_sq = r.squaredNorm();

    if (r_next_norm_sq < residual_factor * r_recompute_norm_sq ||
        r_next_norm_sq > inv_residual_factor * r_recompute_norm_sq) {
      // recompute the residual from scratch to keep it accurate
      r = A * (*x) - b;
      r_next_norm_sq = r.squaredNorm();
      r_recompute_norm_sq = r_next_norm_sq;
    }
    if (r_next_norm_sq <= max_error_sq) break;

    double beta_next = r_next_norm_sq / r_cur_norm_sq;
    p = -r + beta_next * p;
    r_cur_norm_sq = r_next_norm_sq;
  }

  if (r_cur_norm_sq > r_initial_norm_sq &&
      r_cur_norm_sq > r_initial_norm_sq + 1.0e-10 * b.squaredNorm()) {
    // Residual got worse: fall back to an exact solve (kaldi calls
    // SolveQuadraticProblem here).
    KHG_WARN << "Linear CGD residual got worse after " << k
             << " iterations; doing an exact solve.";
    *x = A.ldlt().solve(b);
  }
  return k;
}

GaussPost VectorToPosteriorEntry(const FloatVector &log_likes,
                                 int32_t num_gselect, double min_post) {
  int32_t num_gauss = log_likes.size();
  KHG_ASSERT(num_gauss > 0);
  KHG_ASSERT(min_post >= 0 && min_post < 1.0);
  if (num_gselect <= 0) num_gselect = num_gauss;
  if (num_gselect > num_gauss) num_gselect = num_gauss;

  GaussPost temp_post;
  double max_like = log_likes.maxCoeff();
  if (min_post != 0.0) {
    double like_cutoff = max_like + std::log(min_post);
    for (int32_t g = 0; g < num_gauss; g++) {
      double like = log_likes(g);
      if (like > like_cutoff) {
        double post = std::exp(like - max_like);
        temp_post.emplace_back(g, post);
      }
    }
  }
  if (temp_post.empty()) {
    // min_post was 0.0, or nothing reached the cutoff: use everything.
    temp_post.resize(num_gauss);
    for (int32_t g = 0; g < num_gauss; g++)
      temp_post[g] = std::make_pair(g, std::exp(log_likes(g) - max_like));
  }

  std::sort(temp_post.begin(), temp_post.end(),
            [](const std::pair<int32_t, float> &a,
               const std::pair<int32_t, float> &b) {
              return a.second > b.second;
            });
  if (static_cast<int32_t>(temp_post.size()) > num_gselect)
    temp_post.resize(num_gselect);

  double tot_post = 0.0;
  for (const auto &pair : temp_post) tot_post += pair.second;
  double cutoff = min_post * tot_post;
  while (temp_post.size() > 1 && temp_post.back().second < cutoff) {
    tot_post -= temp_post.back().second;
    temp_post.pop_back();
  }
  // renormalize to sum to one after pruning
  double inv_tot = 1.0 / tot_post;
  for (auto &pair : temp_post) pair.second *= inv_tot;
  return temp_post;
}


}  // namespace khg
