// kaldi-hmm-gmm/csrc/diag-gmm-normal.h
//
// Copyright 2009-2011  Saarland University  Korbinian Riedhammer  Yanmin Qian
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_DIAG_GMM_NORMAL_H_
#define KALDI_HMM_GMM_CSRC_DIAG_GMM_NORMAL_H_

#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/model-common.h"

namespace khg {

class DiagGmm;

/** \class DiagGmmNormal
 *  Definition for Gaussian Mixture Model with diagonal covariances in normal
 *  mode: where the parameters are stored as means and variances (instead of
 *  the exponential form that the DiagGmm class is stored as). This class will
 *  be used in the update (since the update formulas are for the standard
 *  parameterization) and then copied to the exponential form of the DiagGmm
 *  class. The DiagGmmNormal class will not be used anywhere else, and should
 *  not have any extra methods that are not needed.
 */
class DiagGmmNormal {
 public:
  /// Empty constructor.
  DiagGmmNormal() = default;

  explicit DiagGmmNormal(const DiagGmm &gmm) { CopyFromDiagGmm(gmm); }

  DiagGmmNormal(const DiagGmmNormal &) = delete;
  DiagGmmNormal &operator=(const DiagGmmNormal &) = delete;

  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

  /// Copies to DiagGmm the requested parameters
  void CopyToDiagGmm(DiagGmm *diaggmm, GmmFlagsType flags = kGmmAll) const;

  int32_t NumGauss() const { return weights_.size(); }
  int32_t Dim() const { return means_.cols(); }

  DoubleVector weights_;  // not log
  DoubleMatrix means_;
  DoubleMatrix vars_;  // diagonal variance
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DIAG_GMM_NORMAL_H_
