// kaldi-hmm-gmm/csrc/mle-diag-gmm.h
//
// Copyright 2009-2012  Saarland University;  Georg Stemmer;
//                      Microsoft Corporation;  Jan Silovsky; Yanmin Qian
//                      Johns Hopkins University (author: Daniel Povey)
//                      Cisco Systems (author: Neha Agrawal)
//                2023  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/gmm/mle-diag-gmm.h
//
#ifndef KALDI_HMM_GMM_CSRC_MLE_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_MLE_DIAG_GMM_H_

/**
 * \file mle-diag-gmm.h
 * \brief Maximum-likelihood and MAP estimation for diagonal-covariance GMMs.
 *
 * This file is copied/modified from ../kaldi/src/gmm/mle-diag-gmm.h.
 *
 * It provides the estimation side of \c DiagGmm training:
 *
 * - Option structs: \c MleDiagGmmOptions (variance floor, minimum
 *   occupancy/weight, removal of low-count Gaussians) and
 *   \c MapDiagGmmOptions (the MAP taus for means, variances and weights).
 * - \c AccumDiagGmm: accumulators for the per-component zero-, first- and
 *   second-order statistics of a single \c DiagGmm.
 * - The update functions MleDiagGmmUpdate() (ML re-estimation) and
 *   MapDiagGmmUpdate() (MAP adaptation), plus MlObjective() which evaluates
 *   the data log-likelihood that the ML update maximizes.
 *
 * The statistics are accumulated in double precision.  For a component
 * \f$j\f$ and frames \f$t\f$ with posteriors \f$\gamma_{t,j}\f$:
 * \f[ o_j = \sum_t \gamma_{t,j}, \qquad
 *     s^{(1)}_j = \sum_t \gamma_{t,j} \, x_t, \qquad
 *     s^{(2)}_j = \sum_t \gamma_{t,j} \, x_t^{2} \f]
 * (elementwise square; stored in AccumDiagGmm::occupancy(),
 * mean_accumulator() and variance_accumulator() respectively).
 *
 * The ML update replaces the parameters by
 * \f[ \hat\mu_j = \frac{s^{(1)}_j}{o_j}, \qquad
 *     \hat\sigma^{2}_j = \frac{s^{(2)}_j}{o_j} - \hat\mu_j^{2}, \qquad
 *     \hat w_j = \frac{o_j}{\sum_k o_k}, \f]
 * with variance flooring and gating/removal of low-count components; the MAP
 * update interpolates the same estimates with the old parameters using the
 * taus from \c MapDiagGmmOptions.  Only the parameter groups selected by the
 * \p flags argument are updated (and transferred back to the model).
 *
 * \section mle_diag_gmm_update_flags GMM update flags
 *
 * The \p flags arguments select the parameter groups to update.  They are
 * built from \c GmmUpdateFlags (defined in model-common.h, bit values shown
 * below) using a bitwise OR; the corresponding kaldi command-line option is
 * a subset of the string \c "mvwt" (e.g. \c --update-flags=mvw):
 *
 * - \c kGmmMeans = \c 0x001 (kaldi \c 'm'): update the means.
 * - \c kGmmVariances = \c 0x002 (kaldi \c 'v'): update the variances
 *   (requires the means to be accumulated too).
 * - \c kGmmWeights = \c 0x004 (kaldi \c 'w'): update the mixture weights.
 * - \c kGmmTransitions = \c 0x008 (kaldi \c 't'): transition parameters;
 *   not really part of the GMM and not used by the functions in this file.
 * - \c kGmmAll = \c 0x00F (kaldi \c 'a'): all of the above.
 *
 * The concrete typedef of the flag word is \c GmmFlagsType; see also
 * StringToGmmFlags() and AugmentGmmFlags() in model-common.h.
 *
 * Typical usage (mirrors kaldi's \c gmm-acc-stats-ali + \c gmm-est): create
 * an \c AccumDiagGmm with the same flags as the update, fill it with
 * AccumulateFromDiag() (or AccumulateFromPosteriors() /
 * AccumulateForComponent()), then call MleDiagGmmUpdate() and finally
 * DiagGmm::ComputeGconsts().
 */

#include <string>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/model-common.h"

namespace khg {

/** \struct MleDiagGmmOptions
 *  Configuration variables like variance floor, minimum occupancy, etc.
 *  needed in the estimation process.
 *
 *  The defaults match kaldi's (they are registered by kaldi's
 *  \c gmm-est under the same flag names).
 */
struct MleDiagGmmOptions {
  /// Variance floor for each dimension [empty if not supplied].
  /// It is in double since the variance is computed in double precision.
  /// When non-empty it takes precedence over \c min_variance.  There is no
  /// kaldi command-line flag for it; it is set programmatically (kaldi only
  /// registers the four options below).
  DoubleVector variance_floor_vector;
  /// Minimum weight below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true);
  /// kaldi flag \c --min-gaussian-weight; default 1.0e-05.
  float min_gaussian_weight;
  /// Minimum count below which a Gaussian is not updated (and is
  /// removed, if remove_low_count_gaussians == true).
  /// kaldi flag \c --min-gaussian-occupancy; default 10.0.
  float min_gaussian_occupancy;
  /// Minimum allowed variance in any dimension (if no variance floor)
  /// It is in double since the variance is computed in double precision.
  /// kaldi flag \c --min-variance; default 0.001.
  double min_variance;
  /// If true, remove Gaussians whose occupancy or weight falls below the
  /// floors above (the last remaining Gaussian is never removed).
  /// kaldi flag \c --remove-low-count-gaussians; default true.
  bool remove_low_count_gaussians;
  MleDiagGmmOptions() {
    // don't set var floor vector by default.
    min_gaussian_weight = 1.0e-05;
    min_gaussian_occupancy = 10.0;
    min_variance = 0.001;
    remove_low_count_gaussians = true;
  }
  /// \return A human-readable summary of the options.
  std::string ToString() const;
};

/** \struct MapDiagGmmOptions
 *  Configuration variables for Maximum A Posteriori (MAP) update.
 *
 *  The taus act like pseudo-counts of the old parameters: larger tau means
 *  stronger regularization towards the old model.
 */
struct MapDiagGmmOptions {
  /// Tau value for the means.
  /// kaldi flag \c --mean-tau; default 10.0.
  float mean_tau;

  /// Tau value for the variances.  (Note:
  /// whether or not the variances are updated at all will
  /// be controlled by flags.)
  /// kaldi flag \c --variance-tau; default 50.0.
  float variance_tau;

  /// Tau value for the weights-- this tau value is applied
  /// per state, not per Gaussian.
  /// kaldi flag \c --weight-tau; default 10.0.
  float weight_tau;

  MapDiagGmmOptions() : mean_tau(10.0), variance_tau(50.0), weight_tau(10.0) {}

  /// \return A human-readable summary of the options.
  std::string ToString() const;
};

/**
 * \brief Statistics accumulator for one \c DiagGmm.
 *
 * Holds the zero-order (occupancy), first-order (mean) and second-order
 * (variance) accumulators for each mixture component, in double precision.
 * The accumulators that are actually stored are controlled by the flags
 * given at construction: occupancy is stored for \c kGmmWeights, the mean
 * accumulator for \c kGmmMeans, and the variance accumulator for
 * \c kGmmVariances (which implies \c kGmmMeans; see AugmentGmmFlags()).
 */
class AccumDiagGmm {
 public:
  /// Empty constructor.
  AccumDiagGmm() : dim_(0), num_comp_(0), flags_(0) {}

  /// Constructor that sizes the accumulators from a model.
  /// \param gmm  The model the statistics will be accumulated for.
  /// \param flags  Which accumulators to allocate.
  explicit AccumDiagGmm(const DiagGmm &gmm, GmmFlagsType flags) {
    Resize(gmm, flags);
  }

  /// Calls ResizeAccumulators with arguments based on gmm
  /// \param gmm  The model the statistics will be accumulated for.
  /// \param flags  Which accumulators to allocate.
  void Resize(const DiagGmm &gmm, GmmFlagsType flags) {
    Resize(gmm.NumGauss(), gmm.Dim(), flags);
  }

  /// Allocates memory for accumulators
  /// @param num_gauss  Number of gaussians, i.e., number of components/mixtures
  /// @param dim  Dimension of each gaussian
  /// @param flags  Which accumulators to allocate (they are zeroed).
  void Resize(int32_t num_gauss, int32_t dim, GmmFlagsType flags);

  /// Returns the number of mixture components
  /// \return The number of components the accumulators are sized for.
  int32_t NumGauss() const { return num_comp_; }

  /// Returns the dimensionality of the feature vectors
  /// \return The feature dimension.
  int32_t Dim() const { return dim_; }

  /// Zeros the accumulators selected by \p flags.
  /// \param flags  Which accumulators to zero; must be a subset of the
  ///              flags the accumulator was created with.
  void SetZero(GmmFlagsType flags);

  /// Multiplies the accumulators selected by \p flags by a scalar.
  /// \param f  The scale factor.
  /// \param flags  Which accumulators to scale; must be a subset of the
  ///              flags the accumulator was created with.
  void Scale(float f, GmmFlagsType flags);

  /// Accumulate for a single component, given the posterior
  /// @param data A 1-D float tensor of shape (dim_,). Used only if we
  ///             are going to update the mean and the variance
  /// @param comp_index  Only mean_accumulator_[comp_index] and
  ///                    variance_accumulator_[comp_index] are updated
  ///                    if flags_ is kGmmAll
  /// @param weight   mean_accumulator_[comp_index] += data * weight
  ///                 variance_accumulator_[comp_index] + data.square() * weight
  ///                 occupancy_[comp_index] += weight
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
  /// @param gmm  The model that generates the posteriors (must have the same
  ///             number of components and dimension as this accumulator).
  /// @param data 1-D float tensor of shape (dim,)
  /// @param weight  Scale applied to the posteriors before accumulating.
  /// @return The log-likelihood \f$\log p(\mathrm{data})\f$ under \p gmm.
  float AccumulateFromDiag(const DiagGmm &gmm, const FloatVector &data,
                           float weight);

  /// Increment the stats for this component by the specified amount
  /// (not all parts may be taken, depending on flags).
  /// Note: x_stats and x2_stats are assumed to already be multiplied by "occ"
  ///
  /// @param comp_id  Index of the mixture component.
  /// @param occ  Zero-order statistic to add.
  /// @param x_stats 1-D double tensor of shape (dim,)
  /// @param x2_stats 1-D double tensor of shape (dim,)
  void AddStatsForComponent(int32_t comp_id, double occ,
                            const DoubleVector &x_stats,
                            const DoubleVector &x2_stats);

  /// Increment with stats from this other accumulator (times scale)
  /// \param scale  Scale applied to \p acc before adding.
  /// \param acc  The accumulator to add; must have the same dimension,
  ///            number of components and flags.
  void Add(float scale, const AccumDiagGmm &acc);

  /// Smooths the accumulated counts by adding 'tau' extra frames. An example
  /// use for this is I-smoothing for MMIE.   Calls SmoothWithAccum.
  /// \param tau  Number of extra (self-smoothing) frames per component.
  void SmoothStats(float tau);

  /// Smooths the accumulated counts using some other accumulator. Performs a
  /// weighted sum of the current accumulator with the given one. An example use
  /// for this is I-smoothing for MMI and MPE. Both accumulators must have the
  /// same dimension and number of components.
  /// \param tau  Strength of the smoothing.
  /// \param src_acc  The accumulator providing the smoothing stats.
  void SmoothWithAccum(float tau, const AccumDiagGmm &src_acc);

  /// Smooths the accumulated counts using the parameters of a given model.
  /// An example use of this is MAP-adaptation. The model must have the
  /// same dimension and number of components as the current accumulator.
  /// \param tau  Strength of the smoothing.
  /// \param src_gmm  The model providing the smoothing stats.
  void SmoothWithModel(float tau, const DiagGmm &src_gmm);

  /// \return The flags the accumulator was created with (which accumulators
  ///         are stored).
  GmmFlagsType Flags() const { return flags_; }

  /// Const accessor for the zero-order statistics.
  /// \return 1-D double tensor of shape (num_comp_,).
  const DoubleVector &occupancy() const { return occupancy_; }

  /// Mutable accessor for the zero-order statistics.
  /// \return 1-D double tensor of shape (num_comp_,).
  DoubleVector &occupancy() { return occupancy_; }

  /// Const accessor for the first-order statistics.
  /// \return 2-D double tensor of shape (num_comp_, dim).
  const DoubleMatrix &mean_accumulator() const { return mean_accumulator_; }

  /// Mutable accessor for the first-order statistics.
  /// \return 2-D double tensor of shape (num_comp_, dim).
  DoubleMatrix &mean_accumulator() { return mean_accumulator_; }

  /// Const accessor for the second-order statistics.
  /// \return 2-D double tensor of shape (num_comp_, dim).
  const DoubleMatrix &variance_accumulator() const {
    return variance_accumulator_;
  }

  /// Mutable accessor for the second-order statistics.
  /// \return 2-D double tensor of shape (num_comp_, dim).
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

/**
 * \brief ML re-estimation of a \c DiagGmm from accumulated statistics.
 *
 * For each component \f$j\f$ with occupancy \f$o_j\f$, first-order statistic
 * \f$s^{(1)}_j\f$ and second-order statistic \f$s^{(2)}_j\f$ the estimates are
 * \f[ \hat\mu_j = \frac{s^{(1)}_j}{o_j}, \qquad
 *     \hat\sigma^{2}_j = \frac{s^{(2)}_j}{o_j} - \hat\mu_j^{2}, \qquad
 *     \hat w_j = \frac{o_j}{\sum_k o_k}, \f]
 * where the variance is floored at \c config.min_variance (or at
 * \c config.variance_floor_vector if set).  A component is updated only if
 * its occupancy exceeds \c config.min_gaussian_occupancy and its weight
 * exceeds \c config.min_gaussian_weight; otherwise its weight is floored, or
 * the component is removed entirely if
 * \c config.remove_low_count_gaussians is true (never removing the last
 * component).  If only the variances are being updated, the estimate is
 * computed around the old mean so that the mean does not move.
 *
 * Update using the DiagGmm: exponential form.  Sets, does not increment,
 * objf_change_out, floored_elements_out and floored_gauss_out.
 *
 * \param config  Estimation options (floors, minimum counts).
 * \param diag_gmm_acc  The accumulated statistics; must have the same number
 *                      of components and dimension as \p gmm.
 * \param flags  Which parameter groups to update; must be a subset of
 *              diag_gmm_acc.Flags() (see the "GMM update flags" section in
 *              the file docs).
 * \param gmm  The model to update in place.
 * \param obj_change_out  If non-NULL, on return the change in the ML
 *                       objective (see MlObjective()) due to the update.
 * \param count_out  If non-NULL, on return the total occupancy
 *                  \f$\sum_j o_j\f$.
 * \param floored_elements_out  If non-NULL, on return the number of variance
 *                             elements that were floored.
 * \param floored_gauss_out  If non-NULL, on return the number of Gaussians
 *                          that had at least one variance element floored.
 * \param removed_gauss_out  If non-NULL, on return the number of Gaussians
 *                          removed for low counts.
 */
void MleDiagGmmUpdate(const MleDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc, GmmFlagsType flags,
                      DiagGmm *gmm, float *obj_change_out, float *count_out,
                      int32_t *floored_elements_out = nullptr,
                      int32_t *floored_gauss_out = nullptr,
                      int32_t *removed_gauss_out = nullptr);

/**
 * \brief Maximum A Posteriori estimation of the model.
 *
 * Interpolates the ML estimates above with the old parameters using the taus
 * from \p config:
 * \f[ \hat w_j = \frac{o_j + \tau_w w_j}{\sum_k o_k + \tau_w}, \qquad
 *     \hat\mu_j = \frac{s^{(1)}_j + \tau_\mu \mu_j}{o_j + \tau_\mu}, \f]
 * and the variance is computed around the updated mean and mixed with the old
 * variance as
 * \f[ \hat\sigma^{2}_j = \frac{o_j \tilde\sigma^{2}_j
 *        + \tau_\sigma \sigma^{2}_j}{o_j + \tau_\sigma}, \f]
 * where \f$\tilde\sigma^{2}_j\f$ is the data variance around
 * \f$\hat\mu_j\f$.  The weight tau is applied per pdf (state), not per
 * Gaussian.  As in the ML update, only the groups selected by \p flags are
 * transferred back to the model.
 *
 * \param config  MAP taus for means, variances and weights.
 * \param diag_gmm_acc  The accumulated statistics; must have the same number
 *                      of components and dimension as \p gmm.
 * \param flags  Which parameter groups to update (see the "GMM update flags"
 *              section in the file docs).
 * \param gmm  The model to update in place.
 * \param obj_change_out  If non-NULL, on return the change in the ML
 *                       objective due to the update.
 * \param count_out  If non-NULL, on return the total occupancy.
 */
void MapDiagGmmUpdate(const MapDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc, GmmFlagsType flags,
                      DiagGmm *gmm, float *obj_change_out, float *count_out);

/// Calc using the DiagGMM exponential form
/// Evaluates the log-likelihood of the accumulated data under the model,
/// \f[ \mathrm{obj} = \sum_j o_j g_j
///     + \sum_{j,d} s^{(1)}_{jd} \frac{\mu_{jd}}{\sigma^{2}_{jd}}
///     - \frac{1}{2} \sum_{j,d} s^{(2)}_{jd} \frac{1}{\sigma^{2}_{jd}}, \f]
/// i.e. the objective that MleDiagGmmUpdate() maximizes.  Only the parts
/// corresponding to the accumulators present in \p diaggmm_acc are included.
/// \param gmm  The model to evaluate.
/// \param diaggmm_acc  The accumulated statistics.
/// \return The objective value (nats).
float MlObjective(const DiagGmm &gmm, const AccumDiagGmm &diaggmm_acc);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_MLE_DIAG_GMM_H_
