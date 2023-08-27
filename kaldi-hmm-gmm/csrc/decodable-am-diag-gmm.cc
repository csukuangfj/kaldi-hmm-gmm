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

  if (frame != previous_frame_) {  // cache the squared stats.
    data_squared_ = Row(feature_matrix_, frame).square();
    previous_frame_ = frame;
  }

  const DiagGmm &pdf = acoustic_model_.GetPdf(state);
  auto data = Row(feature_matrix_, frame);

  // check if everything is in order
  if (pdf.Dim() != data.size(0)) {
    KHG_ERR << "Dim mismatch: data dim = " << static_cast<int32_t>(data.size(0))
            << " vs. model dim = " << pdf.Dim();
  }

  if (!pdf.valid_gconsts()) {
    KHG_ERR << "State " << state
            << ": Must call ComputeGconsts() "
               "before computing likelihood.";
  }

  auto loglikes = pdf.gconsts().clone();  // need to recreate for each pdf

  // loglikes +=  means * inv(vars) * data.
  loglikes = loglikes.unsqueeze(1);  // (nmix, 1)
  loglikes.addmm_(pdf.means_invvars(), data.unsqueeze(1));

  // loglikes += -0.5 * inv(vars) * data_sq.
  loglikes.addmm_(pdf.inv_vars(), data_squared_.unsqueeze(1), /*beta*/ 1.0,
                  /*alpha*/ -0.5);

  // float log_sum = loglikes.LogSumExp(log_sum_exp_prune_);
  // Note: log_sum_exp_prune_ is never used
  float log_sum = loglikes.logsumexp(/*dim*/ 0).item().toFloat();

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum)) {
    KHG_ERR << "Invalid answer (overflow or invalid variances/features?)";
  }

  log_like_cache_[state].log_like = log_sum;
  log_like_cache_[state].hit_time = frame;

  return log_sum;
}

}  // namespace khg
