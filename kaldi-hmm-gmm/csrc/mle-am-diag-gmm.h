// kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h
//
// Copyright 2009-2012  Saarland University (author: Arnab Ghoshal);
//                      Yanmin Qian; Johns Hopkins University (author: Daniel
//                      Povey) Cisco Systems (author: Neha Agrawal)
//                2023  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/gmm/mle-am-diag-gmm.h
//
#ifndef KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_

/**
 * \file mle-am-diag-gmm.h
 * \brief ML/MAP estimation for whole acoustic models of diagonal GMMs.
 *
 * This file is copied/modified from ../kaldi/src/gmm/mle-am-diag-gmm.h.
 *
 * It provides the model-level counterparts of the per-GMM estimation code in
 * mle-diag-gmm.h:
 *
 * - \c AccumAmDiagGmm: one \c AccumDiagGmm per pdf of an \c AmDiagGmm, plus
 *   running totals of the frame count and log-likelihood for diagnostics.
 * - MleAmDiagGmmUpdate() and MapAmDiagGmmUpdate(): apply
 *   MleDiagGmmUpdate()/MapDiagGmmUpdate() to every pdf and report the summed
 *   objective change and count.
 *
 * The update functions take the same \c MleDiagGmmOptions /
 * \c MapDiagGmmOptions and the same GMM update flags as their per-GMM
 * counterparts (see the "GMM update flags" section in mle-diag-gmm.h); there
 * is no separate options struct at the acoustic-model level.  The reported
 * objective change is the sum of the per-pdf changes of the ML objective
 * (MlObjective()), so it can be printed as the usual
 * <tt>GMM update: objf impr (...)</tt> line of kaldi's \c gmm-est.
 *
 * Typical usage (mirrors kaldi's \c gmm-acc-stats-ali + \c gmm-est): create
 * an \c AccumAmDiagGmm, Init() it from the model, fill it with
 * AccumulateForGmm() once per frame of the training data, then call
 * MleAmDiagGmmUpdate() and AmDiagGmm::ComputeGconsts().
 */

#include <vector>

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/mle-diag-gmm.h"

namespace khg {

/**
 * \brief Statistics accumulator for a whole \c AmDiagGmm.
 *
 * Wraps one \c AccumDiagGmm per pdf (indexed by pdf-id) and keeps running
 * totals of the accumulated frame count and log-likelihood for diagnostics.
 * The accumulator list is sized by Init() and the object owns its
 * accumulators (copy construction and assignment are disabled).
 */
class AccumAmDiagGmm {
 public:
  /// Empty constructor.
  AccumAmDiagGmm() : total_frames_(0.0), total_log_like_(0.0) {}

  /// Destructor; frees the owned per-pdf accumulators.
  ~AccumAmDiagGmm();

  /// Copy construction is not allowed.
  AccumAmDiagGmm(const AccumAmDiagGmm &) = delete;

  /// Assignment is not allowed.
  AccumAmDiagGmm &operator=(const AccumAmDiagGmm &) = delete;

  /// Initializes accumulators for each GMM based on the number of components
  /// and dimension.
  /// \param model  The model the statistics will be accumulated for.
  /// \param flags  Which accumulators to allocate for every pdf.
  void Init(const AmDiagGmm &model, GmmFlagsType flags);

  /// Initialization using different dimension than model.
  /// \param model  The model the statistics will be accumulated for.
  /// \param dim  Feature dimension to allocate (may differ from the model's).
  /// \param flags  Which accumulators to allocate for every pdf.
  void Init(const AmDiagGmm &model, int32_t dim, GmmFlagsType flags);

  /// Zeros the accumulators selected by \p flags.
  /// \param flags  Which accumulators to zero; must be a subset of the
  ///              flags the accumulator was created with.
  void SetZero(GmmFlagsType flags);

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// This does not work with multiple feature transforms.
  ///
  /// @param model  The model being accumulated (used for the posteriors).
  /// @param data 1-D float tensor of shape (dim,)
  /// @param gmm_index  Zero-based pdf-id of the pdf to accumulate for.
  /// @param weight  Scale applied to the statistics of this frame.
  /// @return The log-likelihood of \p data under pdf \p gmm_index.
  float AccumulateForGmm(const AmDiagGmm &model, const FloatVector &data,
                         int32_t gmm_index, float weight);

  /// Accumulate stats for a single GMM in the model; uses data1 for
  /// getting posteriors and data2 for stats. Returns log likelihood.
  ///
  /// @param model  The model being accumulated.
  /// @param data1 1-d float tensor of shape (dim,); used for the posteriors.
  /// @param data2 1-d float tensor of shape (dim,); used for the statistics.
  /// @param gmm_index  Zero-based pdf-id of the pdf to accumulate for.
  /// @param weight  Scale applied to the statistics of this frame.
  /// @return The log-likelihood of \p data1 under pdf \p gmm_index.
  float AccumulateForGmmTwofeats(const AmDiagGmm &model,
                                 const FloatVector &data1,
                                 const FloatVector &data2, int32_t gmm_index,
                                 float weight);

  /// Accumulates stats for a single GMM in the model using pre-computed
  /// Gaussian posteriors.
  ///
  /// @param model  The model being accumulated.
  /// @param data 1-d float tensor (dim,)
  /// @param gmm_index  Zero-based pdf-id of the pdf to accumulate for.
  /// @param posteriors 1-d float tensor (num_comp,)
  void AccumulateFromPosteriors(const AmDiagGmm &model, const FloatVector &data,
                                int32_t gmm_index,
                                const FloatVector &posteriors);

  /// Accumulate stats for a single Gaussian component in the model.
  ///
  /// @param am  The model being accumulated.
  /// @param data 1-d float tensor of shape (dim,)
  /// @param gmm_index  Zero-based pdf-id of the pdf to accumulate for.
  /// @param gauss_index  Index of the mixture component within the pdf.
  /// @param weight  Scale applied to the statistics of this frame.
  void AccumulateForGaussian(const AmDiagGmm &am, const FloatVector &data,
                             int32_t gmm_index, int32_t gauss_index,
                             float weight);

  /// \return The number of per-pdf accumulators (equal to the model's
  ///         NumPdfs()).
  int32_t NumAccs() const { return gmm_accumulators_.size(); }

  /// returns the total count got by summing the count
  /// of the actual stats, may differ from TotCount() if e.g. you did
  /// I-smoothing.
  float TotStatsCount() const;

  /// \return The accumulated frame count.
  /// Be careful since total_frames_ is not updated in AccumulateForGaussian
  float TotCount() const { return total_frames_; }

  /// \return The accumulated log-likelihood of the data seen so far.
  float TotLogLike() const { return total_log_like_; }

  /// Const accessor for a per-pdf accumulator.
  /// \param index  Zero-based pdf-id.
  /// \return Const reference to the \c AccumDiagGmm of the given pdf.
  const AccumDiagGmm &GetAcc(int32_t index) const;

  /// Mutable accessor for a per-pdf accumulator.
  /// \param index  Zero-based pdf-id.
  /// \return Reference to the \c AccumDiagGmm of the given pdf.
  AccumDiagGmm &GetAcc(int32_t index);

  /// Adds the statistics of another accumulator, scaled by \p scale.
  /// \param scale  Scale applied to \p other before adding.
  /// \param other  The accumulator to add; must have the same number of pdfs,
  ///              dimension and flags.
  void Add(float scale, const AccumAmDiagGmm &other);

  /// Scales all statistics by \p scale.
  /// \param scale  The scale factor.
  void Scale(float scale);

  /// \return The feature dimension of the accumulators (0 if there are none).
  int32_t Dim() const {
    return (gmm_accumulators_.empty() || !gmm_accumulators_[0]
                ? 0
                : gmm_accumulators_[0]->Dim());
  }

 private:
  /// MLE accumulators and update methods for the GMMs
  std::vector<AccumDiagGmm *> gmm_accumulators_;

  /// Total counts & likelihood (for diagnostics)
  double total_frames_, total_log_like_;
};

/**
 * \brief ML re-estimation of a whole \c AmDiagGmm.
 *
 * Applies MleDiagGmmUpdate() to every pdf and sums the per-pdf objective
 * changes and counts.  If the dimension of the accumulator does not match the
 * model, the model is resized and reset to zero-mean, unit-variance (with a
 * warning).  Logs the total number of floored variance elements and removed
 * Gaussians, like kaldi's \c gmm-est.
 *
 * \param config  Estimation options (floors, minimum counts).
 * \param amdiaggmm_acc  The accumulated statistics; must have one accumulator
 *                      per pdf of \p am_gmm.
 * \param flags  Which parameter groups to update (see mle-diag-gmm.h).
 * \param am_gmm  The model to update in place.
 * \param obj_change_out  If non-NULL, on return the total change of the ML
 *                       objective over all pdfs.
 * \param count_out  If non-NULL, on return the total occupancy over all
 *                  pdfs.
 */
void MleAmDiagGmmUpdate(const MleDiagGmmOptions &config,
                        const AccumAmDiagGmm &amdiaggmm_acc, GmmFlagsType flags,
                        AmDiagGmm *am_gmm, float *obj_change_out,
                        float *count_out);

/**
 * \brief Maximum A Posteriori update of a whole \c AmDiagGmm.
 *
 * Applies MapDiagGmmUpdate() to every pdf and sums the per-pdf objective
 * changes and counts.  The dimension and the number of pdfs of the
 * accumulator and the model must match.
 *
 * \param config  MAP taus for means, variances and weights.
 * \param amdiag_gmm_acc  The accumulated statistics; must have one
 *                       accumulator per pdf of \p am_gmm.
 * \param flags  Which parameter groups to update (see mle-diag-gmm.h).
 * \param am_gmm  The model to update in place.
 * \param obj_change_out  If non-NULL, on return the total change of the ML
 *                       objective over all pdfs.
 * \param count_out  If non-NULL, on return the total occupancy over all
 *                  pdfs.
 */
void MapAmDiagGmmUpdate(const MapDiagGmmOptions &config,
                        const AccumAmDiagGmm &amdiag_gmm_acc,
                        GmmFlagsType flags, AmDiagGmm *am_gmm,
                        float *obj_change_out, float *count_out);

}  // namespace khg
#endif  // KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_
