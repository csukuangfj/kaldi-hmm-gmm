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

  /// Returns the number of mixture components in the GMM
  int32_t NumGauss() const { return weights_.size(0); }
  /// Returns the dimensionality of the Gaussian mean vectors
  int32_t Dim() const { return means_invvars_.size(1); }

  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

  DiagGmm(int32_t nMix, int32_t dim) : valid_gconsts_(false) {
    Resize(nMix, dim);
  }

  /// Constructor that allows us to merge GMMs with weights.  Weights must sum
  /// to one, or this GMM will not be properly normalized (we don't check this).
  /// Weights must be positive (we check this).
  explicit DiagGmm(const std::vector<std::pair<float, const DiagGmm *>> &gmms);

  /// Sets the gconsts.  Returns the number that are "invalid" e.g. because of
  /// zero weights or variances.
  int32_t ComputeGconsts();

  /// Returns the log-likelihood of a data point (vector) given the GMM
  // @param data  A 1-D tensor
  float LogLikelihood(const torch::Tensor &data) const;

  /// Outputs the per-component log-likelihoods
  /// @param data  1-D tensor.
  /// @param loglikes  1-D tensor.
  void LogLikelihoods(const torch::Tensor &data, torch::Tensor *loglikes) const;

  /// This version of the LogLikelihoods function operates on
  /// a sequence of frames simultaneously; the row index of both "data" and
  /// "loglikes" is the frame index.
  /// @param data 2-D matrix of (num-frames, dim)
  /// @param loglikes On ouput, it contains a  2-D matrix of (num_frames, nmix)
  void LogLikelihoodsMatrix(const torch::Tensor &data,
                            torch::Tensor *loglikes) const;

  /// Outputs the per-component log-likelihoods of a subset of mixture
  /// components.  Note: at output, loglikes->Dim() will equal indices.size().
  /// loglikes[i] will correspond to the log-likelihood of the Gaussian
  /// indexed indices[i], including the mixture weight.
  ///
  /// @param data 1-D tensor of shape (dim,)
  /// @param indices
  /// @param loglikes 1-D tensor of shape (indices.size(),)
  void LogLikelihoodsPreselect(const torch::Tensor &data,
                               const std::vector<int32_t> &indices,
                               torch::Tensor *loglikes) const;

 private:
  /// Equals log(weight) - 0.5 * (log det(var) + mean*mean*inv(var))
  torch::Tensor gconsts_;  // 1-d tensor, (nimx,)
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
