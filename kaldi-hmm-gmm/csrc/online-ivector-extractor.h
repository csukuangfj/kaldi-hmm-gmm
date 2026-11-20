// kaldi-hmm-gmm/kaldi-hmm-gmm/csrc/online-ivector-extractor.h
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

#ifndef KALDI_HMM_GMM_CSRC_ONLINE_IVECTOR_EXTRACTOR_H_
#define KALDI_HMM_GMM_CSRC_ONLINE_IVECTOR_EXTRACTOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/ivector-extractor.h"
#include "kaldi-native-fbank/csrc/online-cmvn.h"

// Streaming i-vector extractor reproducing the dataflow of kaldi's
// OnlineIvectorFeature (online2/online-ivector-feature.cc):
//
//   posteriors: running-CMVN -> splice(+-L,R) -> lda -> UBM (DiagGmm)
//   stats:      [running-CMVN ->] splice(+-L,R) -> lda      (raw by default,
//               i.e. online_cmvn_iextractor=false like vosk's models)
//   then IvectorEstimationStats accumulation, iVector re-estimated every
//   ivector_period frames (warm-started CG, max_count prior scaling).

namespace khg {

struct OnlineIvectorExtractorConfig {
  /// Splice context (ivector/splice.conf).
  int32_t left_context = 3;
  int32_t right_context = 3;
  /// Linear transform applied to spliced features (ivector/final.mat).
  /// [lda_dim][spliced_dim] or [lda_dim][spliced_dim + 1] (affine offset).
  FloatMatrix lda_mat;
  /// Global CMVN stats [2][feat_dim + 1] (ivector/global_cmvn_stats).
  DoubleMatrix global_cmvn_stats;
  /// Running-CMVN options for the posterior chain (online_cmvn.conf).
  knf::OnlineCmvnOptions cmvn_opts;
  /// If true, also apply running CMVN on the stats chain (kaldi's
  /// online_cmvn_iextractor; false in vosk's models).
  bool online_cmvn_iextractor = false;
  /// Prior scaling kicks in above this frame count (vosk: 100).
  double max_count = 0.0;
  /// Warm-started conjugate-gradient iterations per estimate (kaldi: 15).
  int32_t num_cg_iters = 15;
  /// Re-estimate the iVector every N frames (kaldi/vosk: 10).
  int32_t ivector_period = 10;
  /// UBM pruning: keep top-N Gaussians (0 = all).
  int32_t num_gselect = 0;
  /// UBM pruning: drop posteriors below min_post * total (0 = keep all kept).
  double min_post = 0.0;
  /// Scale applied to UBM posteriors before accumulating stats (0 < s <= 1).
  double posterior_scale = 1.0;
};

class OnlineIvectorExtractor {
 public:
  OnlineIvectorExtractor(const OnlineIvectorExtractorConfig &config,
                         const IvectorExtractor &extractor,
                         const DiagGmm &ubm);

  /// Feed raw feature frames (e.g. MFCC, rows = frames).  Frames are
  /// processed with the same right-context lookahead as kaldi; call Flush()
  /// at the end of the utterance to process the tail.
  void AcceptFeatures(const FloatMatrix &raw_feats);
  void Flush();

  /// Start a new utterance (running CMVN restarts from global stats).
  void Reset();

  int32_t FeatDim() const { return feat_dim_; }
  int32_t NumFrames() const { return num_frames_fed_; }
  /// Number of iVectors estimated so far (one per ivector_period frames,
  /// computed at frames 0, period, 2*period, ...).
  int32_t NumIvectorsReady() const {
    return static_cast<int32_t>(ivectors_.size());
  }
  /// The iVector estimated after frame index * ivector_period frames.
  /// Like kaldi's OnlineIvectorFeature output (and ivector-extract-online),
  /// PriorOffset() is removed from dimension 0 so the vector is roughly
  /// zero-mean; the raw estimate keeps the offset internally for CG warm
  /// starts.
  DoubleVector GetIvector(int32_t index) const;
  /// All iVectors as rows [num_ivectors][ivector_dim] (dim 0 offset removed).
  DoubleMatrix GetIvectors() const;
  /// The most recently estimated iVector (dim 0 offset removed).
  DoubleVector LatestIvector() const;
  /// Weighted frame count accumulated in the stats.
  double Count() const { return stats_.Count(); }

 private:
  void ProcessFrame(int32_t t, bool last_chunk);
  FloatMatrix SpliceChain(const std::vector<FloatVector> &frames, int32_t t,
                          bool last_chunk) const;

  OnlineIvectorExtractorConfig config_;
  const IvectorExtractor &extractor_;
  const DiagGmm &ubm_;

  int32_t raw_feat_dim_ = 0;  // input feature dim (e.g. 30 for German vosk)
  int32_t feat_dim_ = 0;      // extractor (post-lda) dim
  int32_t spliced_dim_ = 0;   // raw_feat_dim_ * (1 + L + R)
  int32_t lda_dim_ = 0;
  bool lda_affine_ = false;   // lda_mat has an extra offset column

  std::unique_ptr<knf::OnlineCmvn> posterior_cmvn_;
  std::unique_ptr<knf::OnlineCmvn> stats_cmvn_;  // only if configured

  std::vector<FloatVector> raw_frames_;    // raw feats of this utterance
  std::vector<FloatVector> cmvn_frames_;   // running-CMVN'd feats

  int32_t num_frames_fed_ = 0;       // frames accepted
  int32_t num_frames_processed_ = 0; // frames with stats accumulated

  OnlineIvectorEstimationStats stats_;
  std::vector<DoubleVector> ivectors_;
  DoubleVector cur_ivector_;  // warm start for CG
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_ONLINE_IVECTOR_EXTRACTOR_H_
