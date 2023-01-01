// kaldi-hmm-gmm/csrc/diag-gmm.h
//
// Copyright 2009-2011  Microsoft Corporation;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Georg Stemmer;  Jan Silovsky
//           2012       Arnab Ghoshal
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_DIAG_GMM_H_
// this if is copied and modified from
// kaldi/src/gmm/diag-gmm.h

#include "torch/script.h"

namespace khg {

/// Definition for Gaussian Mixture Model with diagonal covariances
class DiagGmm {
 public:
  /// Empty constructor.
  DiagGmm() : valid_gconsts_(false) {}

  explicit DiagGmm(const DiagGmm &gmm) : valid_gconsts_(false) {
    CopyFromDiagGmm(gmm);
  }

  /// Resizes arrays to this dim. Does not initialize data.
  void Resize(int32_t nMix, int32_t dim);

  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

 private:
  /// Equals log(weight) - 0.5 * (log det(var) + mean*mean*inv(var))
  torch::Tensor gconsts_;  // 1-d tensor
  bool valid_gconsts_;     ///< Recompute gconsts_ if false

  /// 1-D, (nmix,) weights (not log).
  torch::Tensor weights_;
  /// 2-D, (nmix, dim), Inverted (diagonal) variances
  torch::Tensor inv_vars_;

  /// 2-D, (nmix, dim), Means times inverted variance
  torch::Tensor means_invvars_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DIAG_GMM_H_
