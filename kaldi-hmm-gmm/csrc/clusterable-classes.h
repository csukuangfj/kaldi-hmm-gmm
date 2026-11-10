// kaldi-hmm-gmm/csrc/clusterable-classes.h
//
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey
//                2023  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/tree/clusterable-classes.h
//
#ifndef KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#define KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#include <string>

#include "kaldi-hmm-gmm/csrc/clusterable-itf.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"

/**
 * \file clusterable-classes.h
 * \brief Concrete \c Clusterable classes holding scalar and Gaussian
 *        statistics.
 *
 * This file is copied/modified from kaldi/src/tree/clusterable-classes.h.
 *
 * It provides two concrete implementations of the \c Clusterable interface
 * (clusterable-itf.h):
 * - \c ScalarClusterable, which clusters scalars with an \f$x^2\f$ loss.
 * - \c GaussClusterable, which wraps per-dimension Gaussian sufficient
 *   statistics (sum and sum-of-squares) in a form accessible to generic
 *   clustering algorithms.
 *
 * The Gaussian statistics are stored as \c DoubleVector (see eigen.h), which
 * corresponds to kaldi's \c Matrix<double> layout of two rows (sum, then
 * sum-squared) here split into two separate vectors.
 */

namespace khg {

/// \addtogroup clustering_group
/// @{

/**
 * \brief ScalarClusterable clusters scalars with x^2 loss.
 *
 * It accumulates the count \f$n\f$, the sum \f$x\f$ and the sum of squares
 * \f$x_2\f$ of a set of scalars.  The objective function (to be maximized) is
 * the negated weighted sum of squared distances to the mean,
 * \f$\mathrm{Objf} = -(x_2 - x^2 / n)\f$ (zero when \f$n = 0\f$).
 */
class ScalarClusterable : public Clusterable {
 public:
  ~ScalarClusterable() override = default;
  ScalarClusterable() : x_(0), x2_(0), count_(0) {}
  /**
   * \brief Construct from a single scalar observation.
   *
   * \param x  The observed scalar (count becomes 1).
   */
  explicit ScalarClusterable(float x) : x_(x), x2_(x * x), count_(1) {}
  std::string Type() const override { return "scalar"; }
  float Objf() const override;
  void SetZero() override { count_ = x_ = x2_ = 0.0; }
  void Add(const Clusterable &other_in) override;
  void Sub(const Clusterable &other_in) override;
  Clusterable *Copy() const override;
  float Normalizer() const override { return static_cast<float>(count_); }

  /// \brief Returns a string describing the stats; for debugging.
  std::string Info() const;  // For debugging.

  /// \brief Returns the mean of the accumulated scalars (0 if empty).
  float Mean() const { return (count_ != 0 ? x_ / count_ : 0.0); }

 private:
  float x_;
  float x2_;
  float count_;
};

/**
 * \brief GaussClusterable wraps Gaussian statistics in a form accessible
 *        to generic clustering algorithms.
 *
 * It accumulates, per dimension \f$d\f$, the count \f$n\f$, the sum
 * \f$x_d\f$ and the sum of squares \f$x^{(2)}_d\f$.  Writing
 * \f$\mu_d = x_d / n\f$ and
 * \f$\sigma_d^2 = x^{(2)}_d / n - \mu_d^2\f$ with the floored variance
 * \f$\tilde{\sigma}_d^2 = \max(\sigma_d^2, \sigma_{\mathrm{floor}})\f$, the
 * objective function (to be maximized) is the (variance-floored) Gaussian
 * log-likelihood of the accumulated stats,
 * \f[
 *   \mathrm{Objf} = n \left[
 *     \sum_{d} \left(
 *       -\frac{1}{2} \frac{\sigma_d^2}{\tilde{\sigma}_d^2}
 *       -\frac{1}{2} \log \tilde{\sigma}_d^2
 *     \right)
 *     - \frac{D}{2} \log(2\pi)
 *   \right],
 * \f]
 * where \f$D\f$ is the dimension; it is zero when \f$n \le 0\f$.
 */
class GaussClusterable : public Clusterable {
 public:
  ~GaussClusterable() override = default;
  GaussClusterable() : count_(0.0), var_floor_(0.0) {}
  /**
   * \brief Construct empty statistics of a given dimension.
   *
   * \param dim       The dimension of the feature vectors.
   * \param var_floor  The variance floor applied when computing Objf; should
   *                  be common for all objects that are combined.
   */
  GaussClusterable(int32_t dim, float var_floor)
      : count_(0.0),
        x_stats_(DoubleVector::Zero(dim)),
        x2_stats_(DoubleVector::Zero(dim)),
        var_floor_(var_floor) {}

  /**
   * \brief Construct from pre-computed statistics.
   *
   * \param x_stats   1-D tensor of shape (dim,): the per-dimension sums.
   * \param x2_stats  1-D tensor of shape (dim,): the per-dimension sums of
   *                  squares.
   * \param var_floor  The variance floor applied when computing Objf.
   * \param count     The accumulated frame count.
   */
  GaussClusterable(const DoubleVector &x_stats,   // 1-D tensor of shape (dim,)
                   const DoubleVector &x2_stats,  // 1-D tensor of shape (dim,)
                   float var_floor, float count)
      : count_(count),
        x_stats_(x_stats),
        x2_stats_(x2_stats),
        var_floor_(var_floor) {}

  std::string Type() const override { return "gauss"; }

  /**
   * \brief Accumulates one observation vector into the statistics.
   *
   * \param vec     1-D tensor of shape (dim,): the observation.
   * \param weight  Weight of this observation (default 1.0).
   */
  void AddStats(const DoubleVector &vec,  // 1-D tensor of shape (dim,)
                float weight = 1.0);

  float Objf() const override;
  void SetZero() override;
  void Add(const Clusterable &other_in) override;
  void Sub(const Clusterable &other_in) override;
  float Normalizer() const override { return count_; }
  Clusterable *Copy() const override;
  void Scale(float f) override;

  /// \brief Returns the accumulated frame count.
  float count() const { return count_; }

  // Return a 1-D tensor
  /// \brief Returns the per-dimension sums (1-D tensor of shape (dim,)).
  const DoubleVector &x_stats() const { return x_stats_; }

  // Return a 1-D tensor
  /// \brief Returns the per-dimension sums of squares (1-D tensor of shape
  ///        (dim,)).
  const DoubleVector &x2_stats() const { return x2_stats_; }

 private:
  double count_;
  DoubleVector x_stats_;
  DoubleVector x2_stats_;
  double var_floor_;  // should be common for all objects created.
};

/// @} end of "addtogroup clustering_group"

inline void GaussClusterable::SetZero() {
  count_ = 0;
  x_stats_.setZero();
  x2_stats_.setZero();
}

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
