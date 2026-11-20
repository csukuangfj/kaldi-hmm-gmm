// kaldi-hmm-gmm/kaldi-hmm-gmm/csrc/online-ivector-extractor.cc
//
// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// See LICENSE for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kaldi-hmm-gmm/csrc/online-ivector-extractor.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

OnlineIvectorExtractor::OnlineIvectorExtractor(
    const OnlineIvectorExtractorConfig &config,
    const IvectorExtractor &extractor, const DiagGmm &ubm)
    : config_(config), extractor_(extractor), ubm_(ubm),
      stats_(extractor.IvectorDim(), extractor.PriorOffset(),
             config.max_count) {
  // Follow kaldi's online-ivector-feature.cc checks exactly: the raw
  // feature dim comes from the global CMVN stats, and lda_mat's columns are
  // raw_dim * num_splice (optionally +1 for an affine offset).  The raw dim
  // is NOT assumed equal to the extractor's dim (e.g. the German vosk
  // model is 30 -> 40).
  KHG_ASSERT(config_.left_context >= 0 && config_.right_context >= 0);
  KHG_ASSERT(config_.global_cmvn_stats.rows() == 2);
  raw_feat_dim_ = config_.global_cmvn_stats.cols() - 1;
  int32_t num_splice = 1 + config_.left_context + config_.right_context;
  int32_t spliced_input_dim = raw_feat_dim_ * num_splice;
  KHG_ASSERT(config_.lda_mat.cols() == spliced_input_dim ||
             config_.lda_mat.cols() == spliced_input_dim + 1);
  lda_affine_ = (config_.lda_mat.cols() == spliced_input_dim + 1);
  spliced_dim_ = spliced_input_dim;
  feat_dim_ = extractor_.FeatDim();   // post-lda dim
  KHG_ASSERT(config_.lda_mat.rows() == feat_dim_);
  KHG_ASSERT(config_.posterior_scale > 0.0 && config_.posterior_scale <= 1.0);
  KHG_ASSERT(config_.ivector_period > 0);
  lda_dim_ = config_.lda_mat.rows();

  Reset();
}

void OnlineIvectorExtractor::Reset() {
  knf::OnlineCmvnState state(config_.global_cmvn_stats);
  posterior_cmvn_.reset(new knf::OnlineCmvn(config_.cmvn_opts, state));
  if (config_.online_cmvn_iextractor) {
    stats_cmvn_.reset(new knf::OnlineCmvn(config_.cmvn_opts, state));
  } else {
    stats_cmvn_.reset();
  }
  raw_frames_.clear();
  cmvn_frames_.clear();
  num_frames_fed_ = 0;
  num_frames_processed_ = 0;
  ivectors_.clear();
  cur_ivector_ = DoubleVector::Zero(extractor_.IvectorDim());
  stats_ = OnlineIvectorEstimationStats(
      extractor_.IvectorDim(), extractor_.PriorOffset(), config_.max_count);
}

void OnlineIvectorExtractor::AcceptFeatures(const FloatMatrix &raw_feats) {
  KHG_ASSERT(raw_feats.cols() == raw_feat_dim_);
  for (int32_t t = 0; t < raw_feats.rows(); t++) {
    FloatVector frame = raw_feats.row(t).transpose();
    raw_frames_.push_back(frame);
    cmvn_frames_.push_back(posterior_cmvn_->AcceptFeature(frame));
    num_frames_fed_++;
    // Process frames whose right context is fully available (kaldi's
    // OnlineSpliceFrames::NumFramesReady() == n - right_context).
    while (num_frames_processed_ + config_.right_context < num_frames_fed_) {
      ProcessFrame(num_frames_processed_++, /*last_chunk=*/false);
    }
  }
}

void OnlineIvectorExtractor::Flush() {
  while (num_frames_processed_ < num_frames_fed_) {
    ProcessFrame(num_frames_processed_++, /*last_chunk=*/true);
  }
}

FloatMatrix OnlineIvectorExtractor::SpliceChain(
    const std::vector<FloatVector> &frames, int32_t t, bool /*last_chunk*/) const {
  int32_t T = static_cast<int32_t>(frames.size());
  FloatMatrix out(1, spliced_dim_);
  for (int32_t n = 0; n < 1 + config_.left_context + config_.right_context;
       n++) {
    int32_t t2 = t - config_.left_context + n;
    if (t2 < 0) t2 = 0;
    if (t2 >= T) t2 = T - 1;  // kaldi clamps at the utterance level
    out.block(0, n * raw_feat_dim_, 1, raw_feat_dim_) = frames[t2].transpose();
  }
  return out;
}

void OnlineIvectorExtractor::ProcessFrame(int32_t t, bool last_chunk) {
  // Posteriors are computed on the running-CMVN chain (like kaldi's
  // lda_normalized_), stats on the raw chain unless configured otherwise.
  FloatMatrix spliced_post = SpliceChain(cmvn_frames_, t, last_chunk);
  FloatMatrix spliced_stats = SpliceChain(
      config_.online_cmvn_iextractor ? cmvn_frames_ : raw_frames_, t,
      last_chunk);

  auto apply_lda = [&](const FloatMatrix &spliced) -> FloatVector {
    if (lda_affine_) {
      FloatMatrix x(1, spliced_dim_ + 1);
      x.block(0, 0, 1, spliced_dim_) = spliced;
      x(0, spliced_dim_) = 1.0f;
      return (config_.lda_mat * x.transpose()).col(0);
    }
    return (config_.lda_mat * spliced.transpose()).col(0);
  };

  FloatVector post_feat = apply_lda(spliced_post);
  FloatVector stats_feat = apply_lda(spliced_stats);

  FloatVector log_likes;
  ubm_.LogLikelihoods(post_feat, &log_likes);
  GaussPost post = VectorToPosteriorEntry(
      log_likes, config_.num_gselect, config_.min_post);
  if (config_.posterior_scale != 1.0) {
    for (auto &pair : post) pair.second *= config_.posterior_scale;
  }

  stats_.AccStats(extractor_, stats_feat, post);

  if (t % config_.ivector_period == 0) {
    stats_.GetIvector(config_.num_cg_iters, &cur_ivector_);
    ivectors_.push_back(cur_ivector_);
  }
}

DoubleVector OnlineIvectorExtractor::GetIvector(int32_t index) const {
  DoubleVector out = ivectors_[index];
  out(0) -= extractor_.PriorOffset();
  return out;
}

DoubleVector OnlineIvectorExtractor::LatestIvector() const {
  DoubleVector out = cur_ivector_;
  out(0) -= extractor_.PriorOffset();
  return out;
}

DoubleMatrix OnlineIvectorExtractor::GetIvectors() const {
  DoubleMatrix out(static_cast<int32_t>(ivectors_.size()),
                   extractor_.IvectorDim());
  for (size_t i = 0; i < ivectors_.size(); i++) out.row(i) = GetIvector(i);
  return out;
}

}  // namespace khg
