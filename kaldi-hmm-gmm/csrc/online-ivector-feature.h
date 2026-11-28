// kaldi-hmm-gmm/csrc/online-ivector-feature.h
//
// Copyright 2013-2014  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/online2/online-ivector-feature.h
// (classes OnlineIvectorFeature, OnlineSilenceWeightingConfig and
// OnlineSilenceWeighting).

#ifndef KALDI_HMM_GMM_CSRC_ONLINE_IVECTOR_FEATURE_H_
#define KALDI_HMM_GMM_CSRC_ONLINE_IVECTOR_FEATURE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/ivector-extractor.h"
#include "kaldi-hmm-gmm/csrc/online-ivector-extractor.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "kaldi-native-fbank/csrc/online-cmvn.h"

/**
 * \file online-ivector-feature.h
 * \brief kaldi's online2 i-vector feature: the i-vector the nnet3 online
 * decoders (and vosk) feed to the acoustic model, with optional
 * silence weighting driven by the decoder's best path.
 *
 * Typical streaming loop (what kaldi's online2-wav-nnet3-latgen-faster and
 * vosk's Recognizer do):
 *
 * \code
 *   OnlineIvectorFeature ivector(config, extractor, ubm);
 *   OnlineSilenceWeighting silence(trans_model, silence_config, 3);
 *   for each chunk of audio:
 *     ivector.AcceptFeatures(new_mfcc_frames);
 *     if (silence.Active() && ivector.NumFramesReady() > 0) {
 *       silence.ComputeCurrentTraceback(best_path_transition_ids);
 *       std::vector<std::pair<int32_t, float>> delta_weights;
 *       silence.GetDeltaWeights(ivector.NumFramesReady(), 0, &delta_weights);
 *       ivector.UpdateFrameWeights(delta_weights);
 *     }
 *     ... for each nnet chunk: ivector.GetFrame(t, &iv) ...
 * \endcode
 */

namespace khg {

/**
 * kaldi's OnlineIvectorFeature (online2/online-ivector-feature.h).
 *
 * The raw (base) features, e.g. MFCC, are pushed with AcceptFeatures() /
 * InputFinished() instead of being pulled from an OnlineFeatureInterface.
 * Two chains are built on top of them, exactly as in kaldi:
 *
 *   - posteriors: online CMVN -> splice -> LDA -> UBM posteriors
 *     (top num_gselect Gaussians, pruned by min_post, scaled by
 *     posterior_scale and the frame weight);
 *   - stats: splice -> LDA (or the CMVN chain if online_cmvn_iextractor).
 *
 * Stats are accumulated lazily, up to the frame asked for in GetFrame().
 * With use_most_recent_ivector (the default, and what vosk uses) GetFrame()
 * returns the most recent estimate; otherwise the one of frame
 * (frame / ivector_period) * ivector_period.  Frame weights (silence
 * weighting) are provided with UpdateFrameWeights(); once they are, every
 * frame needs a weight before it is used (kaldi's semantics: frames get a
 * weight of zero until a delta-weight for them arrives).
 *
 * Not ported: the speaker adaptation state (SetAdaptationState /
 * GetAdaptationState), the UBM log-likelihood diagnostics and the
 * max_remembered_frames limit; every utterance starts from the global CMVN
 * stats.
 */
class OnlineIvectorFeature {
 public:
  /// The config fields used are: left_context, right_context, lda_mat,
  /// global_cmvn_stats, cmvn_opts, online_cmvn_iextractor, max_count,
  /// num_cg_iters, ivector_period, num_gselect, min_post, posterior_scale,
  /// use_most_recent_ivector and greedy_ivector_extractor.
  OnlineIvectorFeature(const OnlineIvectorExtractorConfig &config,
                       const IvectorExtractor &extractor, const DiagGmm &ubm);

  /// Append raw feature frames (rows = frames), e.g. MFCC.
  void AcceptFeatures(const FloatMatrix &raw_feats);

  /// No more frames will be accepted (kaldi: the base feature's last frame
  /// is known).  NumFramesReady() then includes the right-context frames.
  void InputFinished();

  /// The i-vector dimension.
  int32_t Dim() const;

  /// Frames whose i-vector can be requested: the number of raw frames,
  /// minus the splice right context while input is not finished
  /// (kaldi's OnlineSpliceFrames::NumFramesReady()).
  int32_t NumFramesReady() const;

  /// True if `frame` is the last frame of a finished input.
  bool IsLastFrame(int32_t frame) const;

  /// The i-vector for `frame` (requires 0 <= frame < NumFramesReady()), with
  /// the prior offset removed from dimension 0, as in kaldi.  This updates
  /// the stats up to `frame` (or to the last ready frame with
  /// greedy_ivector_extractor).
  void GetFrame(int32_t frame, FloatVector *feat);

  /// Provide frame weights as (frame, delta-weight) pairs (kaldi's
  /// UpdateFrameWeights).  The weights are deltas: a frame whose weight
  /// changes from w1 to w2 gets an entry with w2 - w1; frames already used
  /// in the stats are corrected the next time the stats are updated.
  void UpdateFrameWeights(
      const std::vector<std::pair<int32_t, float>> &delta_weights);

  /// Data count after posterior scaling, divided by posterior_scale
  /// (kaldi's NumFrames(); a weighted frame count).
  float NumFrames() const;

  /// Average objective-function improvement per frame of the current
  /// estimate (kaldi's ObjfImprPerFrame(); a diagnostic).
  float ObjfImprPerFrame() const;

 private:
  // The LDA-transformed, spliced features of `frame` on the stats chain
  // (lda_ in kaldi) and on the CMVN chain (lda_normalized_).  Cached on
  // first use, like kaldi's OnlineCacheFeature.
  const FloatVector &LdaFrame(int32_t frame);
  const FloatVector &LdaNormalizedFrame(int32_t frame);
  FloatVector SpliceAndTransform(const std::vector<FloatVector> &frames,
                                 int32_t frame) const;

  void UpdateStatsForFrames(
      const std::vector<std::pair<int32_t, float>> &frame_weights);
  float GetMinPost(float weight) const;
  void UpdateStatsUntilFrame(int32_t frame);
  void UpdateStatsUntilFrameWeighted(int32_t frame);

  OnlineIvectorExtractorConfig config_;
  const IvectorExtractor &extractor_;
  const DiagGmm &ubm_;

  int32_t raw_feat_dim_ = 0;
  int32_t spliced_dim_ = 0;
  bool lda_affine_ = false;

  std::unique_ptr<knf::OnlineCmvn> cmvn_;
  std::vector<FloatVector> raw_frames_;
  std::vector<FloatVector> cmvn_frames_;
  bool input_finished_ = false;

  std::vector<FloatVector> lda_cache_;
  std::vector<bool> lda_cached_;
  std::vector<FloatVector> lda_normalized_cache_;
  std::vector<bool> lda_normalized_cached_;

  OnlineIvectorEstimationStats ivector_stats_;

  // num_frames_stats_ is the number of frames of data we have already
  // accumulated from this utterance and put in ivector_stats_.
  int32_t num_frames_stats_ = 0;

  // delta_weights_ is written to by UpdateFrameWeights, and read by
  // UpdateStatsUntilFrameWeighted.  The top element is the lowest-numbered
  // frame.
  std::priority_queue<std::pair<int32_t, float>,
                      std::vector<std::pair<int32_t, float>>,
                      std::greater<std::pair<int32_t, float>>>
      delta_weights_;

  // True once UpdateFrameWeights() was called.
  bool delta_weights_provided_ = false;
  // True if stats were updated without frame weights (then weights may
  // not be provided any more).
  bool updated_with_no_delta_weights_ = false;
  // The highest frame index that got a weight so far.
  int32_t most_recent_frame_with_weight_ = -1;

  // The current iVector (with the prior offset still in dimension 0).
  DoubleVector current_ivector_;

  // With use_most_recent_ivector == false: the iVector of frame
  // t = i * ivector_period, as ivectors_history_[i].
  std::vector<FloatVector> ivectors_history_;
};

/// kaldi's OnlineSilenceWeightingConfig.
struct OnlineSilenceWeightingConfig {
  /// Colon- (or comma-) separated integer ids of the silence phones, e.g.
  /// "1:2:3:4:5:6:7:8:9:10" (vosk: --endpoint.silence-phones).
  std::string silence_phones_str;
  /// Weight of the frames the decoder traceback identifies as silence
  /// (vosk: 0.001).
  float silence_weight = 1.0f;
  /// Transition-ids that get repeated at least this many times (if > 0) are
  /// treated as silence.
  float max_state_duration = -1.0f;

  /// Whether silence weighting is in effect.
  bool Active() const {
    return !silence_phones_str.empty() && silence_weight != 1.0f;
  }
};

/**
 * kaldi's OnlineSilenceWeighting: turns the decoder's best-path traceback
 * into per-frame weights for OnlineIvectorFeature::UpdateFrameWeights(),
 * down-weighting silence frames.
 *
 * Differences from kaldi: ComputeCurrentTraceback() takes the best path as
 * transition-ids (one per decoder frame) instead of the decoder object, so
 * this library stays independent of the decoder; get them from
 * kaldi-decoder's GetBestPathTransitionIds() (LatticeFasterOnlineDecoder).
 * The resulting weights are the same.
 */
class OnlineSilenceWeighting {
 public:
  /// `frame_subsampling_factor`: decoder frames per input feature frame
  /// (3 for chain models).  The config is copied.
  OnlineSilenceWeighting(const TransitionModel &trans_model,
                         const OnlineSilenceWeightingConfig &config,
                         int32_t frame_subsampling_factor = 1);

  bool Active() const { return config_.Active(); }

  /// Update the stored traceback with the current best path: one
  /// transition-id per decoder frame (all frames decoded so far).  kaldi
  /// calls it with use_final_probs == false while decoding.
  void ComputeCurrentTraceback(const std::vector<int32_t> &transition_ids);

  /// Output (frame, delta-weight) pairs for OnlineIvectorFeature (input
  /// frame indexes).  `num_frames_ready` is the number of input frames
  /// ready (the i-vector's NumFramesReady()); `first_decoder_frame` is the
  /// input-frame offset of decoder frame 0 (0 unless the decoder was
  /// restarted mid-stream).
  void GetDeltaWeights(int32_t num_frames_ready, int32_t first_decoder_frame,
                       std::vector<std::pair<int32_t, float>> *delta_weights);

  /// Decoder frames (subsampled) of the traceback that are not silence.
  void GetNonsilenceFrames(int32_t num_frames_ready,
                           int32_t first_decoder_frame,
                           std::vector<int32_t> *frames);

 private:
  const TransitionModel &trans_model_;
  OnlineSilenceWeightingConfig config_;
  int32_t frame_subsampling_factor_;
  std::unordered_set<int32_t> silence_phones_;

  struct FrameInfo {
    // the transition-id of the traceback, or -1 if no traceback yet
    int32_t transition_id = -1;
    // the weight we have output for this frame so far
    float current_weight = 0.0f;
  };

  // frame_info_ is indexed by decoder frame.
  std::vector<FrameInfo> frame_info_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_ONLINE_IVECTOR_FEATURE_H_
