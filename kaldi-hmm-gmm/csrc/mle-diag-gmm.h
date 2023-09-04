// kaldi-hmm-gmm/csrc/mle-diag-gmm.h
//
// Copyright 2009-2012  Saarland University;  Georg Stemmer;
//                      Microsoft Corporation;  Jan Silovsky; Yanmin Qian
//                      Johns Hopkins University (author: Daniel Povey)
//                      Cisco Systems (author: Neha Agrawal)
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_MLE_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_MLE_DIAG_GMM_H_

#include <string>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/model-common.h"

namespace khg {

/** \struct MleDiagGmmOptions
 *  Configuration variables like variance floor, minimum occupancy, etc.
 *  needed in the estimation process.
 */
struct MleDiagGmmOptions {
  /// Variance floor for each dimension [empty if not supplied].
  /// It is in double since the variance is computed in double precision.
  DoubleVector variance_floor_vector;
  /// Minimum weight below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true);
  float min_gaussian_weight;
  /// Minimum count below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true).
  float min_gaussian_occupancy;
  /// Minimum allowed variance in any dimension (if no variance floor)
  /// It is in double since the variance is computed in double precision.
  double min_variance;
  bool remove_low_count_gaussians;
  MleDiagGmmOptions() {
    // don't set var floor vector by default.
    min_gaussian_weight = 1.0e-05;
    min_gaussian_occupancy = 10.0;
    min_variance = 0.001;
    remove_low_count_gaussians = true;
  }
  std::string ToString() const;
};

/** \struct MapDiagGmmOptions
 *  Configuration variables for Maximum A Posteriori (MAP) update.
 */
struct MapDiagGmmOptions {
  /// Tau value for the means.
  float mean_tau;

  /// Tau value for the variances.  (Note:
  /// whether or not the variances are updated at all will
  /// be controlled by flags.)
  float variance_tau;

  /// Tau value for the weights-- this tau value is applied
  /// per state, not per Gaussian.
  float weight_tau;

  MapDiagGmmOptions() : mean_tau(10.0), variance_tau(50.0), weight_tau(10.0) {}

  std::string ToString() const;
};

class AccumDiagGmm {
 public:
  AccumDiagGmm() : dim_(0), num_comp_(0), flags_(0) {}

  explicit AccumDiagGmm(const DiagGmm &gmm, GmmFlagsType flags) {
    Resize(gmm, flags);
  }

  /// Calls ResizeAccumulators with arguments based on gmm
  void Resize(const DiagGmm &gmm, GmmFlagsType flags) {
    Resize(gmm.NumGauss(), gmm.Dim(), flags);
  }

  /// Allocates memory for accumulators
  /// @param num_gauss  Number of gaussians, i.e., number of components/mixtures
  /// @param dim  Dimension of each gaussian
  /// @param flags
  void Resize(int32_t num_gauss, int32_t dim, GmmFlagsType flags);

  /// Returns the number of mixture components
  int32_t NumGauss() const { return num_comp_; }

  /// Returns the dimensionality of the feature vectors
  int32_t Dim() const { return dim_; }

  void SetZero(GmmFlagsType flags);

  void Scale(float f, GmmFlagsType flags);

  /// Accumulate for a single component, given the posterior
  /// @param data A 1-D float tensor of shape (dim_,). Used only if we
  ///             are going to update the mean and the variance
  /// @param comp_index  Only mean_accumulator_[comp_index] and
  ///                    variance_accumulator_[comp_index] are updated
  ///                    if flags_ is kGmmAll
  /// @param weight   mean_accumulator_[comp_index] += data * weight
  ///                 variance_accumulator_[comp_index] + data.square() * weight
  void AccumulateForComponent(const FloatVector &data, int32_t comp_index,
                              float weight);

  /// Accumulate for all components, given the posteriors.
  ///
  /// @param data 1-D float tensor of shape (dim,)
  /// @param gauss_posteriors  1-D float tensor of shape (num_comp,)
  ///
  /// occupancy_ += posteriors
  /// mean_accumulator_ += gauss_posteriors.unsqueeze(1) * data
  /// variance_accumulator_ += gauss_posteriors.unsqueeze(1) * data.square()
  void AccumulateFromPosteriors(const FloatVector &data,
                                const FloatVector &gauss_posteriors);

  /// Accumulate for all components given a diagonal-covariance GMM.
  /// Computes posteriors and returns log-likelihood
  ///
  /// @param gmm
  /// @param data 1-D float tensor of shape (dim,)
  float AccumulateFromDiag(const DiagGmm &gmm, const FloatVector &data,
                           float weight);

  /// Increment the stats for this component by the specified amount
  /// (not all parts may be taken, depending on flags).
  /// Note: x_stats and x2_stats are assumed to already be multiplied by "occ"
  ///
  /// @param comp_id
  /// @param occ
  /// @param x_stats 1-D double tensor of shape (dim,)
  /// @param x2_stats 1-D double tensor of shape (dim,)
  void AddStatsForComponent(int32_t comp_id, double occ,
                            const DoubleVector &x_stats,
                            const DoubleVector &x2_stats);

  /// Increment with stats from this other accumulator (times scale)
  void Add(float scale, const AccumDiagGmm &acc);

  /// Smooths the accumulated counts by adding 'tau' extra frames. An example
  /// use for this is I-smoothing for MMIE.   Calls SmoothWithAccum.
  void SmoothStats(float tau);

  /// Smooths the accumulated counts using some other accumulator. Performs a
  /// weighted sum of the current accumulator with the given one. An example use
  /// for this is I-smoothing for MMI and MPE. Both accumulators must have the
  /// same dimension and number of components.
  void SmoothWithAccum(float tau, const AccumDiagGmm &src_acc);

  /// Smooths the accumulated counts using the parameters of a given model.
  /// An example use of this is MAP-adaptation. The model must have the
  /// same dimension and number of components as the current accumulator.
  void SmoothWithModel(float tau, const DiagGmm &src_gmm);

  GmmFlagsType Flags() const { return flags_; }

  const DoubleVector &occupancy() const { return occupancy_; }

  DoubleVector &occupancy() { return occupancy_; }

  const DoubleMatrix &mean_accumulator() const { return mean_accumulator_; }

  DoubleMatrix &mean_accumulator() { return mean_accumulator_; }

  const DoubleMatrix &variance_accumulator() const {
    return variance_accumulator_;
  }

  DoubleMatrix &variance_accumulator() { return variance_accumulator_; }

 private:
  int32_t dim_;
  int32_t num_comp_;
  /// Flags corresponding to the accumulators that are stored.
  GmmFlagsType flags_;

  DoubleVector occupancy_;             // 1-D double tensor, (num_comp_,)
  DoubleMatrix mean_accumulator_;      // (num_comp_, dim)
  DoubleMatrix variance_accumulator_;  // (num_comp_, dim)
};

/// for computing the maximum-likelihood estimates of the parameters of
/// a Gaussian mixture model.
/// Update using the DiagGmm: exponential form.  Sets, does not increment,
/// objf_change_out, floored_elements_out and floored_gauss_out.
void MleDiagGmmUpdate(const MleDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc, GmmFlagsType flags,
                      DiagGmm *gmm, float *obj_change_out, float *count_out,
                      int32_t *floored_elements_out = nullptr,
                      int32_t *floored_gauss_out = nullptr,
                      int32_t *removed_gauss_out = nullptr);

/// Maximum A Posteriori estimation of the model.
void MapDiagGmmUpdate(const MapDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc, GmmFlagsType flags,
                      DiagGmm *gmm, float *obj_change_out, float *count_out);

/// Calc using the DiagGMM exponential form
float MlObjective(const DiagGmm &gmm, const AccumDiagGmm &diaggmm_acc);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_MLE_DIAG_GMM_H_
