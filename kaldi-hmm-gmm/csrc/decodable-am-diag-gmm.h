// kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Lukas Burget
//                2013  Johns Hopkins Universith (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/gmm/decodable-am-diag-gmm.h
#ifndef KALDI_HMM_GMM_CSRC_DECODABLE_AM_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_DECODABLE_AM_DIAG_GMM_H_

#include <vector>

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/decodable-itf.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"

namespace khg {

/// DecodableAmDiagGmmUnmapped is a decodable object that
/// takes indices that correspond to pdf-id's plus one.
/// This may be used in future in a decoder that doesn't need
/// to output alignments, if we create FSTs that have the pdf-ids
/// plus one as the input labels (we couldn't use the pdf-ids
/// themselves because they start from zero, and the graph might
/// have epsilon transitions).

class DecodableAmDiagGmmUnmapped : public DecodableInterface {
 public:
  /// If you set log_sum_exp_prune to a value greater than 0 it will prune
  /// in the LogSumExp operation (larger = more exact); I suggest 5.
  /// This is advisable if it's spending a long time doing exp
  /// operations.
  DecodableAmDiagGmmUnmapped(const AmDiagGmm &am,
                             const FloatMatrix &feats,  // 2-D float matrix
                             float log_sum_exp_prune = -1.0)
      : acoustic_model_(am),
        feature_matrix_(feats),
        log_sum_exp_prune_(log_sum_exp_prune) {
    ResetLogLikeCache();
  }

  DecodableAmDiagGmmUnmapped(const DecodableAmDiagGmmUnmapped &) = delete;
  DecodableAmDiagGmmUnmapped &operator=(const DecodableAmDiagGmmUnmapped &) =
      delete;

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  float LogLikelihood(int32_t frame, int32_t state_index) override {
    return LogLikelihoodZeroBased(frame, state_index - 1);
  }

  int32_t NumFramesReady() const override { return feature_matrix_.rows(); }

  // Indices are one-based!  This is for compatibility with OpenFst.
  int32_t NumIndices() const override { return acoustic_model_.NumPdfs(); }

  bool IsLastFrame(int32_t frame) const override {
    KHG_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 protected:
  void ResetLogLikeCache();
  virtual float LogLikelihoodZeroBased(int32_t frame, int32_t state_index);

  const AmDiagGmm &acoustic_model_;
  const FloatMatrix feature_matrix_;  // (num_frames, feature_dim)
  float log_sum_exp_prune_;           // never used

  /// Defines a cache record for a state
  struct LikelihoodCacheRecord {
    float log_like;    ///< Cache value
    int32_t hit_time;  ///< Frame for which this value is relevant
  };
  std::vector<LikelihoodCacheRecord> log_like_cache_;
};

class DecodableAmDiagGmmScaled : public DecodableAmDiagGmmUnmapped {
 public:
  DecodableAmDiagGmmScaled(const AmDiagGmm &am, const TransitionModel &tm,
                           const FloatMatrix &feats,  // 2-D float matrix
                           float scale, float log_sum_exp_prune = -1.0)
      : DecodableAmDiagGmmUnmapped(am, feats, log_sum_exp_prune),
        trans_model_(tm),
        scale_(scale) {}

  DecodableAmDiagGmmScaled(const DecodableAmDiagGmmScaled &) = delete;
  DecodableAmDiagGmmScaled &operator=(const DecodableAmDiagGmmScaled &) =
      delete;

  // Note, frames are numbered from zero but transition-ids from one.
  float LogLikelihood(int32_t frame, int32_t tid) override {
    return scale_ *
           LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid));
  }
  // Indices are one-based!  This is for compatibility with OpenFst.
  int32_t NumIndices() const override {
    return trans_model_.NumTransitionIds();
  }

  const TransitionModel *TransModel() { return &trans_model_; }

 private:  // want to access it public to have pdf id information
  const TransitionModel &trans_model_;  // for transition-id to pdf mapping
  float scale_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DECODABLE_AM_DIAG_GMM_H_
