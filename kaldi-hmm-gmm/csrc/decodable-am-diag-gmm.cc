// kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.cc
//
// Copyright 2009-2011  Saarland University;  Lukas Burget
//                2013  Johns Hopkins Universith (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/gmm/decodable-am-diag-gmm.cc

#include "kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/kaldi-math.h"

namespace khg {

void DecodableAmDiagGmmUnmapped::ResetLogLikeCache() {
  if (static_cast<int32_t>(log_like_cache_.size()) !=
      acoustic_model_.NumPdfs()) {
    log_like_cache_.resize(acoustic_model_.NumPdfs());
  }

  auto it = log_like_cache_.begin(), end = log_like_cache_.end();

  for (; it != end; ++it) {
    it->hit_time = -1;
  }
}

float DecodableAmDiagGmmUnmapped::LogLikelihoodZeroBased(int32_t frame,
                                                         int32_t state) {
  KHG_ASSERT(static_cast<size_t>(frame) <
             static_cast<size_t>(NumFramesReady()));

  KHG_ASSERT(static_cast<size_t>(state) < static_cast<size_t>(NumIndices()) &&
             "Likely graph/model mismatch, e.g. using wrong HCLG.fst");

  if (log_like_cache_[state].hit_time == frame) {
    return log_like_cache_[state].log_like;  // return cached value, if found
  }

  const DiagGmm &pdf = acoustic_model_.GetPdf(state);

  // check if everything is in order
  if (pdf.Dim() != feature_matrix_.cols()) {
    KHG_ERR << "Dim mismatch: data dim = " << feature_matrix_.cols()
            << " vs. model dim = " << pdf.Dim();
  }

  if (!pdf.valid_gconsts()) {
    KHG_ERR << "State " << state
            << ": Must call ComputeGconsts() "
               "before computing likelihood.";
  }

  auto data = feature_matrix_.row(frame).transpose();
  FloatVector loglikes = pdf.gconsts() + pdf.means_invvars() * data -
                         0.5 * pdf.inv_vars() * data.array().square().matrix();

  // float log_sum = loglikes.LogSumExp(log_sum_exp_prune_);
  // Note: log_sum_exp_prune_ is never used
  float log_sum = LogSumExp(loglikes);

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum)) {
    KHG_ERR << "Invalid answer (overflow or invalid variances/features?)";
  }

  log_like_cache_[state].log_like = log_sum;
  log_like_cache_[state].hit_time = frame;

  return log_sum;
}

}  // namespace khg
