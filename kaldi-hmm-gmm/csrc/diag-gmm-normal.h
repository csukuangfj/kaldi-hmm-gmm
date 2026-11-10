// kaldi-hmm-gmm/csrc/diag-gmm-normal.h
//
// Copyright 2009-2011  Saarland University  Korbinian Riedhammer  Yanmin Qian
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_DIAG_GMM_NORMAL_H_
#define KALDI_HMM_GMM_CSRC_DIAG_GMM_NORMAL_H_

/**
 * \file diag-gmm-normal.h
 * \brief Normal (mean/variance) representation of diagonal GMMs
 *  (\c DiagGmmNormal).
 *
 * This file is copied/modified from kaldi/src/gmm/diag-gmm-normal.h.
 *
 * It provides \c khg::DiagGmmNormal, a lightweight companion of \c DiagGmm
 * (diag-gmm.h) that stores the parameters of a diagonal-covariance GMM in
 * plain "normal" form: linear mixture weights, means and diagonal variances,
 * all in double precision.  This is the form the MLE/MAP update formulas of
 * mle-diag-gmm.h are written in, and the form used for smoothing and for
 * serializing parameters outside the kaldi on-disk format.
 *
 * The API is deliberately minimal:
 *
 * - CopyFromDiagGmm() converts a \c DiagGmm from its exponential form
 *   (\f$\mu \odot \sigma^{-2}\f$, \f$\sigma^{-2}\f$) to plain means and
 *   variances; the constructor from a \c DiagGmm does the same.
 * - CopyToDiagGmm() converts back, optionally copying only the parameter
 *   groups selected by \c GmmUpdateFlags (weights, means, variances).
 * - NumGauss()/Dim() report the size; the parameters are exposed as the
 *   public members weights_, means_ and vars_ so that updates can modify
 *   them in place.
 *
 * Typical usage: construct from the \c DiagGmm being updated, modify the
 * public members with the estimation formulas, then CopyToDiagGmm() with the
 * desired flags and call DiagGmm::ComputeGconsts().
 */

#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/model-common.h"

namespace khg {

class DiagGmm;

/** \class DiagGmmNormal
 *  Definition for Gaussian Mixture Model with diagonal covariances in normal
 *  mode: where the parameters are stored as means and variances (instead of
 *  the exponential form that the DiagGmm class is stored as). This class will
 *  be used in the update (since the update formulas are for the standard
 *  class. The DiagGmmNormal class will not be used anywhere else, and should
 *  not have any extra methods that are not needed.
 */
class DiagGmmNormal {
 public:
  /// Empty constructor.
  DiagGmmNormal() = default;

  /// Constructor that converts from the exponential form.
  /// \param gmm  The model to convert (must have the same number of
  ///            components and dimension as the target of CopyToDiagGmm()).
  explicit DiagGmmNormal(const DiagGmm &gmm) { CopyFromDiagGmm(gmm); }

  /// Copy construction is not allowed.
  DiagGmmNormal(const DiagGmmNormal &) = delete;

  /// Assignment is not allowed.
  DiagGmmNormal &operator=(const DiagGmmNormal &) = delete;

  /// Copies from given DiagGmm
  /// Converts from exponential form: weights are copied as-is, variances are
  /// obtained as \f$\sigma^2 = 1 / \sigma^{-2}\f$ and means as
  /// \f$\mu = (\mu \odot \sigma^{-2}) \, \sigma^{2}\f$.
  /// \param diaggmm  The model to convert from.
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

  /// Copies to DiagGmm the requested parameters
  /// Only the parameter groups selected by \p flags are transferred (weights
  /// for \c kGmmWeights, variances for \c kGmmVariances, means for
  /// \c kGmmMeans).  If the variances are copied without the means, the
  /// means-times-inverse-variances of the target are recomputed from its old
  /// means and the new variances.  The target's gconsts are invalidated.
  /// \param diaggmm  The model to convert to; must have the same number of
  ///                components and dimension as this object.
  /// \param flags  Which parameter groups to copy (default: all).
  void CopyToDiagGmm(DiagGmm *diaggmm, GmmFlagsType flags = kGmmAll) const;

  /// \return The number of mixture components.
  int32_t NumGauss() const { return weights_.size(); }

  /// \return The dimensionality of the feature vectors.
  int32_t Dim() const { return means_.cols(); }

  /// The mixture weights, 1-D double tensor of shape (num_mix,), not log.
  DoubleVector weights_;  // not log

  /// The means, 2-D double tensor of shape (num_mix, dim).
  DoubleMatrix means_;

  /// The diagonal variances, 2-D double tensor of shape (num_mix, dim).
  DoubleMatrix vars_;  // diagonal variance
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DIAG_GMM_NORMAL_H_
