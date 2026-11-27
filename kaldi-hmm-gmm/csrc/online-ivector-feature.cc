// kaldi-hmm-gmm/csrc/online-ivector-feature.cc
//
// Copyright 2013-2014  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from
// kaldi/src/online2/online-ivector-feature.cc

#include "kaldi-hmm-gmm/csrc/online-ivector-feature.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

OnlineIvectorFeature::OnlineIvectorFeature(
    const OnlineIvectorExtractorConfig &config,
    const IvectorExtractor &extractor, const DiagGmm &ubm)
    : config_(config),
      extractor_(extractor),
      ubm_(ubm),
      ivector_stats_(extractor.IvectorDim(), extractor.PriorOffset(),
                     config.max_count) {
  // Same checks as kaldi's OnlineIvectorExtractionInfo::Check() plus the
  // splice/LDA dims (the raw dim comes from the global CMVN stats).
  KHG_ASSERT(config_.left_context >= 0 && config_.right_context >= 0);
  KHG_ASSERT(config_.global_cmvn_stats.rows() == 2);
  KHG_ASSERT(config_.ivector_period > 0);
  KHG_ASSERT(config_.num_cg_iters > 0);
  KHG_ASSERT(config_.posterior_scale > 0.0 && config_.posterior_scale <= 1.0);
  KHG_ASSERT(config_.min_post >= 0.0 && config_.min_post < 0.5);
  raw_feat_dim_ = config_.global_cmvn_stats.cols() - 1;
  int32_t num_splice = 1 + config_.left_context + config_.right_context;
  spliced_dim_ = raw_feat_dim_ * num_splice;
  KHG_ASSERT(config_.lda_mat.cols() == spliced_dim_ ||
             config_.lda_mat.cols() == spliced_dim_ + 1);
  lda_affine_ = (config_.lda_mat.cols() == spliced_dim_ + 1);
  KHG_ASSERT(config_.lda_mat.rows() == extractor_.FeatDim());
  KHG_ASSERT(ubm_.Dim() == extractor_.FeatDim());

  knf::OnlineCmvnState naive_cmvn_state(config_.global_cmvn_stats);
  cmvn_.reset(new knf::OnlineCmvn(config_.cmvn_opts, naive_cmvn_state));

  // Set the iVector to its default value, [ prior_offset, 0, 0, ... ].
  current_ivector_ = DoubleVector::Zero(extractor_.IvectorDim());
  current_ivector_(0) = extractor_.PriorOffset();
}

void OnlineIvectorFeature::AcceptFeatures(const FloatMatrix &raw_feats) {
  KHG_ASSERT(!input_finished_);
  KHG_ASSERT(raw_feats.rows() == 0 || raw_feats.cols() == raw_feat_dim_);
  for (int32_t t = 0; t < raw_feats.rows(); t++) {
    FloatVector frame = raw_feats.row(t).transpose();
    raw_frames_.push_back(frame);
    // kaldi's OnlineCmvn::GetFrame(t) uses only frames <= t, so the
    // normalized frame can be computed as soon as the frame arrives.
    cmvn_frames_.push_back(cmvn_->AcceptFeature(frame));
  }
  lda_cache_.resize(raw_frames_.size());
  lda_cached_.resize(raw_frames_.size(), false);
  lda_normalized_cache_.resize(raw_frames_.size());
  lda_normalized_cached_.resize(raw_frames_.size(), false);
}

void OnlineIvectorFeature::InputFinished() { input_finished_ = true; }

int32_t OnlineIvectorFeature::Dim() const { return extractor_.IvectorDim(); }

int32_t OnlineIvectorFeature::NumFramesReady() const {
  int32_t num_frames = static_cast<int32_t>(raw_frames_.size());
  if (num_frames > 0 && input_finished_) return num_frames;
  return std::max<int32_t>(0, num_frames - config_.right_context);
}

bool OnlineIvectorFeature::IsLastFrame(int32_t frame) const {
  return input_finished_ &&
         frame == static_cast<int32_t>(raw_frames_.size()) - 1;
}

FloatVector OnlineIvectorFeature::SpliceAndTransform(
    const std::vector<FloatVector> &frames, int32_t frame) const {
  // kaldi's OnlineSpliceFrames::GetFrame: context frames are clamped to
  // [0, T - 1] with T = the number of frames available right now.
  int32_t T = static_cast<int32_t>(frames.size());
  FloatVector spliced(spliced_dim_);
  int32_t n = 0;
  for (int32_t t2 = frame - config_.left_context;
       t2 <= frame + config_.right_context; t2++, n++) {
    int32_t t2_limited = std::min(std::max(t2, 0), T - 1);
    spliced.segment(n * raw_feat_dim_, raw_feat_dim_) = frames[t2_limited];
  }
  // kaldi's OnlineTransform: linear, or affine if the matrix has an extra
  // column.
  if (lda_affine_) {
    return config_.lda_mat.leftCols(spliced_dim_) * spliced +
           config_.lda_mat.col(spliced_dim_);
  }
  return config_.lda_mat * spliced;
}

const FloatVector &OnlineIvectorFeature::LdaFrame(int32_t frame) {
  if (!lda_cached_[frame]) {
    lda_cache_[frame] = SpliceAndTransform(raw_frames_, frame);
    lda_cached_[frame] = true;
  }
  return lda_cache_[frame];
}

const FloatVector &OnlineIvectorFeature::LdaNormalizedFrame(int32_t frame) {
  if (!lda_normalized_cached_[frame]) {
    lda_normalized_cache_[frame] = SpliceAndTransform(cmvn_frames_, frame);
    lda_normalized_cached_[frame] = true;
  }
  return lda_normalized_cache_[frame];
}

void OnlineIvectorFeature::UpdateFrameWeights(
    const std::vector<std::pair<int32_t, float>> &delta_weights) {
  // add the elements to delta_weights_, which is a priority queue.  The top
  // element of the priority queue is the lowest numbered frame (we ensured
  // this by making the comparison object std::greater instead of std::less).
  for (size_t i = 0; i < delta_weights.size(); i++) {
    delta_weights_.push(delta_weights[i]);
    int32_t frame = delta_weights[i].first;
    KHG_ASSERT(frame >= 0);
    if (frame > most_recent_frame_with_weight_)
      most_recent_frame_with_weight_ = frame;
  }
  delta_weights_provided_ = true;
}

float OnlineIvectorFeature::GetMinPost(float weight) const {
  float min_post = static_cast<float>(config_.min_post);
  float abs_weight = std::fabs(weight);
  // If we return 0.99, it will have the same effect as just picking the
  // most probable Gaussian on that frame.
  if (abs_weight == 0.0f) return 0.99f;  // I don't anticipate reaching here.
  min_post /= abs_weight;
  if (min_post > 0.99f) min_post = 0.99f;
  return min_post;
}

void OnlineIvectorFeature::UpdateStatsForFrames(
    const std::vector<std::pair<int32_t, float>> &frame_weights_in) {
  // Remove duplicates of frames (kaldi's MergePairVectorSumming): sort on
  // the frame, sum the weights of each frame, drop zero sums.
  std::vector<std::pair<int32_t, float>> frame_weights(frame_weights_in);
  std::sort(frame_weights.begin(), frame_weights.end(),
            [](const std::pair<int32_t, float> &a,
               const std::pair<int32_t, float> &b) {
              return a.first < b.first;
            });
  std::vector<std::pair<int32_t, float>> merged;
  for (size_t i = 0; i < frame_weights.size();) {
    std::pair<int32_t, float> cur = frame_weights[i++];
    while (i < frame_weights.size() &&
           frame_weights[i].first == cur.first) {
      cur.second += frame_weights[i++].second;
    }
    if (cur.second != 0.0f) merged.push_back(cur);
  }
  if (merged.empty()) return;

  int32_t num_frames = static_cast<int32_t>(merged.size());
  std::vector<GaussPost> posteriors(num_frames);
  FloatMatrix feats(num_frames, extractor_.FeatDim());
  FloatVector log_likes;
  for (int32_t i = 0; i < num_frames; i++) {
    int32_t t = merged[i].first;
    float weight = merged[i].second;
    ubm_.LogLikelihoods(LdaNormalizedFrame(t), &log_likes);
    // "posteriors" stores, for each frame, the pruned posteriors for the
    // Gaussians in the UBM.
    GaussPost &posterior = posteriors[i];
    posterior = VectorToPosteriorEntry(log_likes, config_.num_gselect,
                                       GetMinPost(weight));
    for (size_t j = 0; j < posterior.size(); j++) {
      posterior[j].second *=
          static_cast<float>(config_.posterior_scale) * weight;
    }
    if (!config_.online_cmvn_iextractor) {
      // default: get features without OnlineCmvn
      feats.row(i) = LdaFrame(t).transpose();
    } else {
      // get features with OnlineCmvn
      feats.row(i) = LdaNormalizedFrame(t).transpose();
    }
  }
  ivector_stats_.AccStats(extractor_, feats, posteriors);
}

void OnlineIvectorFeature::UpdateStatsUntilFrame(int32_t frame) {
  KHG_ASSERT(frame >= 0 && frame < NumFramesReady() &&
             !delta_weights_provided_);
  updated_with_no_delta_weights_ = true;

  int32_t ivector_period = config_.ivector_period;
  int32_t num_cg_iters = config_.num_cg_iters;

  std::vector<std::pair<int32_t, float>> frame_weights;

  for (; num_frames_stats_ <= frame; num_frames_stats_++) {
    int32_t t = num_frames_stats_;
    float frame_weight = 1.0f;
    frame_weights.push_back(std::pair<int32_t, float>(t, frame_weight));
    if ((!config_.use_most_recent_ivector && t % ivector_period == 0) ||
        (config_.use_most_recent_ivector && t == frame)) {
      // The call below to UpdateStatsForFrames() is equivalent to doing, for
      // all valid indexes i:
      //  UpdateStatsForFrame(cur_start_frame + i, frame_weights[i])
      UpdateStatsForFrames(frame_weights);
      frame_weights.clear();
      ivector_stats_.GetIvector(num_cg_iters, &current_ivector_);
      if (!config_.use_most_recent_ivector) {  // need to cache iVectors.
        int32_t ivec_index = t / ivector_period;
        KHG_ASSERT(ivec_index ==
                   static_cast<int32_t>(ivectors_history_.size()));
        ivectors_history_.push_back(current_ivector_.cast<float>());
      }
    }
  }
  if (!frame_weights.empty()) UpdateStatsForFrames(frame_weights);
}

void OnlineIvectorFeature::UpdateStatsUntilFrameWeighted(int32_t frame) {
  KHG_ASSERT(frame >= 0 && frame < NumFramesReady() &&
             delta_weights_provided_ && !updated_with_no_delta_weights_ &&
             frame <= most_recent_frame_with_weight_);

  int32_t ivector_period = config_.ivector_period;
  int32_t num_cg_iters = config_.num_cg_iters;

  std::vector<std::pair<int32_t, float>> frame_weights;
  frame_weights.reserve(delta_weights_.size());

  for (; num_frames_stats_ <= frame; num_frames_stats_++) {
    int32_t t = num_frames_stats_;
    // Instead of just updating frame t, we update all frames that need
    // updating with index <= t, in case old frames were reclassified as
    // silence/nonsilence.
    while (!delta_weights_.empty() && delta_weights_.top().first <= t) {
      frame_weights.push_back(delta_weights_.top());
      delta_weights_.pop();
    }
    if ((!config_.use_most_recent_ivector && t % ivector_period == 0) ||
        (config_.use_most_recent_ivector && t == frame)) {
      UpdateStatsForFrames(frame_weights);
      frame_weights.clear();
      ivector_stats_.GetIvector(num_cg_iters, &current_ivector_);
      if (!config_.use_most_recent_ivector) {  // need to cache iVectors.
        int32_t ivec_index = t / ivector_period;
        KHG_ASSERT(ivec_index ==
                   static_cast<int32_t>(ivectors_history_.size()));
        ivectors_history_.push_back(current_ivector_.cast<float>());
      }
    }
  }
  if (!frame_weights.empty()) UpdateStatsForFrames(frame_weights);
}

void OnlineIvectorFeature::GetFrame(int32_t frame, FloatVector *feat) {
  int32_t frame_to_update_until =
      (config_.greedy_ivector_extractor ? NumFramesReady() - 1 : frame);
  if (!delta_weights_provided_)  // No silence weighting.
    UpdateStatsUntilFrame(frame_to_update_until);
  else
    UpdateStatsUntilFrameWeighted(frame_to_update_until);

  if (config_.use_most_recent_ivector) {
    // use the most recent iVector we have, even if 'frame' is significantly
    // in the past.
    *feat = current_ivector_.cast<float>();
  } else {
    int32_t i = frame / config_.ivector_period;  // rounds down.
    // if the following fails, UpdateStatsUntilFrame would have a bug.
    KHG_ASSERT(static_cast<size_t>(i) < ivectors_history_.size());
    *feat = ivectors_history_[i];
  }
  // Subtract the prior-mean from the first dimension of the output feature
  // so it's approximately zero-mean.
  // (kaldi: float -= double, evaluated in double)
  (*feat)(0) = static_cast<float>((*feat)(0) - extractor_.PriorOffset());
}

float OnlineIvectorFeature::NumFrames() const {
  return static_cast<float>(ivector_stats_.NumFrames() /
                            config_.posterior_scale);
}

float OnlineIvectorFeature::ObjfImprPerFrame() const {
  return static_cast<float>(ivector_stats_.ObjfChange(current_ivector_));
}

// ---------------------------------------------------------------------------
// OnlineSilenceWeighting
// ---------------------------------------------------------------------------

namespace {

// kaldi's SplitStringToIntegers(str, ":,", false, &out): empty fields are
// an error.
std::vector<int32_t> ParseSilencePhones(const std::string &str) {
  std::vector<int32_t> ans;
  if (str.empty()) return ans;
  size_t start = 0;
  while (true) {
    size_t end = str.find_first_of(":,", start);
    std::string field = str.substr(start, end - start);
    if (field.empty()) {
      KHG_ERR << "Invalid silence-phones string '" << str << "'";
    }
    char *p = nullptr;
    long v = std::strtol(field.c_str(), &p, 10);  // NOLINT
    if (p == field.c_str() || *p != '\0') {
      KHG_ERR << "Invalid silence-phones string '" << str << "'";
    }
    ans.push_back(static_cast<int32_t>(v));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return ans;
}

}  // namespace

OnlineSilenceWeighting::OnlineSilenceWeighting(
    const TransitionModel &trans_model,
    const OnlineSilenceWeightingConfig &config,
    int32_t frame_subsampling_factor)
    : trans_model_(trans_model),
      config_(config),
      frame_subsampling_factor_(frame_subsampling_factor) {
  KHG_ASSERT(frame_subsampling_factor_ >= 1);
  for (int32_t p : ParseSilencePhones(config.silence_phones_str))
    silence_phones_.insert(p);
}

void OnlineSilenceWeighting::ComputeCurrentTraceback(
    const std::vector<int32_t> &transition_ids) {
  int32_t num_frames_decoded = static_cast<int32_t>(transition_ids.size());
  int32_t num_frames_prev = static_cast<int32_t>(frame_info_.size());
  // note, num_frames_prev is not the number of frames previously decoded,
  // it's the generally-larger number of frames that we were requested to
  // provide weights for.
  if (num_frames_prev < num_frames_decoded)
    frame_info_.resize(num_frames_decoded);
  if (num_frames_prev > num_frames_decoded &&
      frame_info_[num_frames_decoded].transition_id != -1) {
    KHG_ERR << "Number of frames decoded decreased";  // Likely bug
  }
  if (num_frames_decoded == 0) return;
  // kaldi walks the traceback backwards and stops at the first token it had
  // seen before; from there on the traceback is unchanged.  Overwriting all
  // frames gives the same transition-ids.
  for (int32_t frame = num_frames_decoded - 1; frame >= 0; frame--) {
    int32_t transition_id = transition_ids[frame];
    KHG_ASSERT(transition_id > 0);
    frame_info_[frame].transition_id = transition_id;
  }
}

void OnlineSilenceWeighting::GetDeltaWeights(
    int32_t num_frames_ready, int32_t first_decoder_frame,
    std::vector<std::pair<int32_t, float>> *delta_weights) {
  // num_frames_ready is at the feature frame-rate, most of the code
  // in this function is at the decoder frame-rate.
  // round up, so we are sure to get weights for at least the frame
  // 'num_frames_ready - 1', and maybe one or two frames afterward.
  KHG_ASSERT(num_frames_ready > first_decoder_frame || num_frames_ready == 0);
  int32_t fs = frame_subsampling_factor_;
  int32_t num_decoder_frames_ready =
      (num_frames_ready - first_decoder_frame + fs - 1) / fs;

  const float max_state_duration = config_.max_state_duration;
  const float silence_weight = config_.silence_weight;

  delta_weights->clear();

  int32_t prev_num_frames_processed = static_cast<int32_t>(frame_info_.size());
  if (static_cast<int32_t>(frame_info_.size()) < num_decoder_frames_ready)
    frame_info_.resize(num_decoder_frames_ready);

  // Don't go further backward into the past then 100 frames before the most
  // recent frame previously than 100 frames when modifying the traceback.
  int32_t begin_frame = std::max<int32_t>(0, prev_num_frames_processed - 100);
  int32_t frames_out = static_cast<int32_t>(frame_info_.size()) - begin_frame;
  // frames_out is the number of frames we will output.
  KHG_ASSERT(frames_out >= 0);
  std::vector<float> frame_weight(frames_out, 1.0f);
  // we will set frame_weight to the value silence_weight for silence frames
  // and for transition-ids that repeat with duration > max_state_duration.
  // Frames newer than the most recent traceback will get a weight equal to
  // the weight for the most recent frame in the traceback; or the silence
  // weight, if there is no traceback at all available yet.

  // First treat some special cases.
  if (frames_out == 0)  // Nothing to output.
    return;
  if (frame_info_[begin_frame].transition_id == -1) {
    // We do not have any traceback at all within the frames we are to
    // output...  find the most recent weight that we output and apply the
    // same weight to all the new output; or output the silence weight, if
    // nothing was output.
    float weight = (begin_frame == 0 ? silence_weight
                                     : frame_info_[begin_frame - 1]
                                           .current_weight);
    for (int32_t offset = 0; offset < frames_out; offset++)
      frame_weight[offset] = weight;
  } else {
    int32_t current_run_start_offset = 0;
    for (int32_t offset = 0; offset < frames_out; offset++) {
      int32_t frame = begin_frame + offset;
      int32_t transition_id = frame_info_[frame].transition_id;
      if (transition_id == -1) {
        // this frame does not yet have a decoder traceback, so just
        // duplicate the silence/non-silence status of the most recent
        // frame we have a traceback for (probably a reasonable guess).
        frame_weight[offset] = frame_weight[offset - 1];
      } else {
        int32_t phone = trans_model_.TransitionIdToPhone(transition_id);
        bool is_silence = (silence_phones_.count(phone) != 0);
        if (is_silence) frame_weight[offset] = silence_weight;
        // now deal with max-duration issues.
        if (max_state_duration > 0 &&
            (offset + 1 == frames_out ||
             transition_id != frame_info_[frame + 1].transition_id)) {
          // If this is the last frame of a run...
          int32_t run_length = offset - current_run_start_offset + 1;
          if (run_length >= max_state_duration) {
            // treat runs of the same transition-id longer than the max, as
            // silence, even if they were not silence.
            for (int32_t offset2 = current_run_start_offset;
                 offset2 <= offset; offset2++)
              frame_weight[offset2] = silence_weight;
          }
          if (offset + 1 < frames_out) current_run_start_offset = offset + 1;
        }
      }
    }
  }
  // Now commit the stats...
  for (int32_t offset = 0; offset < frames_out; offset++) {
    int32_t frame = begin_frame + offset;
    float old_weight = frame_info_[frame].current_weight;
    float new_weight = frame_weight[offset];
    float weight_diff = new_weight - old_weight;
    frame_info_[frame].current_weight = new_weight;
    // Even if the delta-weight is zero for the last frame, we provide it,
    // because the identity of the most recent frame with a weight is used in
    // some debugging/checking code.
    if (weight_diff != 0.0f || offset + 1 == frames_out) {
      for (int32_t i = 0; i < frame_subsampling_factor_; i++) {
        int32_t input_frame =
            first_decoder_frame + (frame * frame_subsampling_factor_) + i;
        delta_weights->push_back(std::make_pair(input_frame, weight_diff));
      }
    }
  }
}

void OnlineSilenceWeighting::GetNonsilenceFrames(
    int32_t num_frames_ready, int32_t first_decoder_frame,
    std::vector<int32_t> *frames) {
  // num_frames_ready is at the feature frame-rate, most of the code
  // in this function is at the decoder frame-rate.
  KHG_ASSERT(num_frames_ready > first_decoder_frame || num_frames_ready == 0);
  int32_t fs = frame_subsampling_factor_;
  int32_t num_decoder_frames_ready =
      (num_frames_ready - first_decoder_frame + fs - 1) / fs;

  frames->clear();

  int32_t prev_num_frames_processed = static_cast<int32_t>(frame_info_.size());
  if (static_cast<int32_t>(frame_info_.size()) < num_decoder_frames_ready)
    frame_info_.resize(num_decoder_frames_ready);

  // Don't go further backward into the past then 500 frames before the most
  // recent frame
  int32_t begin_frame = std::max<int32_t>(0, prev_num_frames_processed - 500);
  int32_t frames_out = static_cast<int32_t>(frame_info_.size()) - begin_frame;
  // frames_out is the number of frames we will output.
  KHG_ASSERT(frames_out >= 0);

  for (int32_t offset = 0; offset < frames_out; offset++) {
    int32_t frame = begin_frame + offset;
    int32_t transition_id = frame_info_[frame].transition_id;
    if (transition_id != -1) {
      int32_t phone = trans_model_.TransitionIdToPhone(transition_id);
      bool is_silence = (silence_phones_.count(phone) != 0);
      if (!is_silence) frames->push_back(frame);
    }
  }
}

}  // namespace khg
