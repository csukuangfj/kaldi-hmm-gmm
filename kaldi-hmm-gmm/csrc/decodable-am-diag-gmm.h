// kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h
//
// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Lukas Burget
//                2013  Johns Hopkins Universith (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

/**
 * \file decodable-am-diag-gmm.h
 * \brief Decodable objects for AmDiagGmm acoustic models.
 *
 * This file is copied/modified from kaldi/src/gmm/decodable-am-diag-gmm.h.
 *
 * This module provides the decodable objects used to score frames with a
 * diagonal-covariance GMM acoustic model (AmDiagGmm); they implement the
 * \c kaldi_decoder::DecodableInterface from the kaldi-decoder project and are
 * what kaldi's gmm-align-compiled and decoding binaries plug into the
 * alignment/decoding machinery:
 *
 * - DecodableAmDiagGmmUnmapped: indices are pdf-ids plus one, so no
 *   transition model is involved.
 * - DecodableAmDiagGmmScaled: indices are transition-ids (one-based); the
 *   GMM log-likelihood of the transition-id's pdf is returned multiplied by
 *   an acoustic scale, so acoustic scores can be weighted relative to the
 *   graph (transition and language model) scores.
 *
 * Usage notes: frames are numbered from zero while indices are one-based (for
 * compatibility with OpenFst).  The decodables cache per-pdf likelihoods and
 * support optional pruning of the LogSumExp over mixture components.
 * Transition probabilities are not part of these scores; they live on the
 * graph arcs (see hmm-utils.h).
 *
 * See also kaldi/src/gmm/decodable-am-diag-gmm.h.
 */
#ifndef KALDI_HMM_GMM_CSRC_DECODABLE_AM_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_DECODABLE_AM_DIAG_GMM_H_

#include <vector>

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-decoder/csrc/decodable-itf.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"

namespace khg {

/**
 * DecodableAmDiagGmmUnmapped is a decodable object that
 * takes indices that correspond to pdf-id's plus one.
 * This may be used in future in a decoder that doesn't need
 * to output alignments, if we create FSTs that have the pdf-ids
 * plus one as the input labels (we couldn't use the pdf-ids
 * themselves because they start from zero, and the graph might
 * have epsilon transitions).
 */
class DecodableAmDiagGmmUnmapped : public kaldi_decoder::DecodableInterface {
 public:
  /**
   * \brief Constructor.
   *
   * If you set log_sum_exp_prune to a value greater than 0 it will prune
   * in the LogSumExp operation (larger = more exact); I suggest 5.
   * This is advisable if it's spending a long time doing exp
   * operations.
   *
   * \param am                The acoustic model (not owned; must outlive
   *                          this object).
   * \param feats             2-D feature matrix, (num_frames, feature_dim).
   * \param log_sum_exp_prune Pruning threshold for the LogSumExp over
   *                          mixture components; negative disables pruning.
   */
  DecodableAmDiagGmmUnmapped(const AmDiagGmm &am,
                             const FloatMatrix &feats,
                             float log_sum_exp_prune = -1.0)
      : acoustic_model_(am),
        feature_matrix_(feats),
        log_sum_exp_prune_(log_sum_exp_prune) {
    ResetLogLikeCache();
  }

  DecodableAmDiagGmmUnmapped(const DecodableAmDiagGmmUnmapped &) = delete;
  DecodableAmDiagGmmUnmapped &operator=(const DecodableAmDiagGmmUnmapped &) =
      delete;

  /// Note, frames are numbered from zero.  But state_index is numbered
  /// from one (this routine is called by FSTs).
  /// \param frame       Frame index (zero-based).
  /// \param state_index Pdf-id plus one (one-based).
  /// \return The log-likelihood \f$\log p(\mathbf{x}_t \mid \mathrm{pdf})\f$
  ///         of the given pdf at the given frame.
  float LogLikelihood(int32_t frame, int32_t state_index) override {
    return LogLikelihoodZeroBased(frame, state_index - 1);
  }

  /// \return The number of frames available, i.e. the number of rows of the
  ///         feature matrix.
  int32_t NumFramesReady() const override { return feature_matrix_.rows(); }

  /// Indices are one-based!  This is for compatibility with OpenFst.
  /// \return The number of pdfs in the acoustic model.
  int32_t NumIndices() const override { return acoustic_model_.NumPdfs(); }

  /// \param frame  Frame index (zero-based).
  /// \return True if \p frame is the last frame of the utterance.
  bool IsLastFrame(int32_t frame) const override {
    KHG_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 protected:
  /// Clears the per-pdf likelihood cache (called at construction).
  void ResetLogLikeCache();

  /// The zero-based implementation of LogLikelihood(): \p state_index is the
  /// pdf-id.  Applies the likelihood cache and the log-sum-exp pruning.
  virtual float LogLikelihoodZeroBased(int32_t frame, int32_t state_index);

  /// The acoustic model (not owned).
  const AmDiagGmm &acoustic_model_;

  /// The feature matrix, (num_frames, feature_dim).
  const FloatMatrix feature_matrix_;

  /// Log-sum-exp pruning threshold (larger = more exact); <= 0 disables
  /// pruning.
  float log_sum_exp_prune_;

  /// Defines a cache record for a state
  struct LikelihoodCacheRecord {
    float log_like;    ///< Cache value
    int32_t hit_time;  ///< Frame for which this value is relevant
  };
  /// Cache of the last computed log-likelihood of each pdf.
  std::vector<LikelihoodCacheRecord> log_like_cache_;
};

/**
 * \brief Decodable for an AmDiagGmm whose indices are transition-ids, with
 *  acoustic scaling.
 *
 * This is the decodable used by kaldi's alignment and decoding tools (e.g.
 * gmm-align-compiled).  It maps the transition-id to its pdf via the
 * TransitionModel (the self-loop pdf for self-loops, the forward pdf
 * otherwise) and returns the GMM log-likelihood of that pdf, multiplied by
 * the acoustic scale \f$s\f$:
 * \f[
 *   \mathrm{LogLikelihood}(t, \mathrm{tid}) \;=\; s \cdot
 *   \log p\!\left(\mathbf{x}_t \mid \mathrm{pdf}(\mathrm{tid})\right).
 * \f]
 * The acoustic scale weights the acoustic log-likelihoods relative to the
 * graph scores (transition and language-model probabilities live on the
 * graph arcs and are not included in these scores).  Frames are numbered
 * from zero but transition-ids from one.
 */
class DecodableAmDiagGmmScaled : public DecodableAmDiagGmmUnmapped {
 public:
  /**
   * \brief Constructor.
   *
   * \param am                The acoustic model (not owned).
   * \param tm                The transition model (not owned); used for the
   *                          transition-id to pdf-id mapping.
   * \param feats             2-D feature matrix, (num_frames, feature_dim).
   * \param scale             The acoustic scale \f$s\f$ applied to all
   *                          log-likelihoods.
   * \param log_sum_exp_prune Pruning threshold for the LogSumExp over
   *                          mixture components (see the base class);
   *                          negative disables pruning.
   */
  DecodableAmDiagGmmScaled(const AmDiagGmm &am, const TransitionModel &tm,
                           const FloatMatrix &feats,
                           float scale, float log_sum_exp_prune = -1.0)
      : DecodableAmDiagGmmUnmapped(am, feats, log_sum_exp_prune),
        trans_model_(tm),
        scale_(scale) {}

  DecodableAmDiagGmmScaled(const DecodableAmDiagGmmScaled &) = delete;
  DecodableAmDiagGmmScaled &operator=(const DecodableAmDiagGmmScaled &) =
      delete;

  /// Note, frames are numbered from zero but transition-ids from one.
  /// \param frame  Frame index (zero-based).
  /// \param tid    Transition-id (one-based).
  /// \return The scaled log-likelihood of the pdf of \p tid at \p frame,
  ///         i.e. \f$s \cdot \log p(\mathbf{x}_t \mid \mathrm{pdf})\f$.
  float LogLikelihood(int32_t frame, int32_t tid) override {
    return scale_ *
           LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid));
  }
  /// Indices are one-based!  This is for compatibility with OpenFst.
  /// \return The number of transition-ids in the transition model.
  int32_t NumIndices() const override {
    return trans_model_.NumTransitionIds();
  }

  /// \return The transition model (not owned).
  const TransitionModel *TransModel() { return &trans_model_; }

 private:  // want to access it public to have pdf id information
  /// The transition model, for transition-id to pdf mapping (not owned).
  const TransitionModel &trans_model_;

  /// The acoustic scale \f$s\f$.
  float scale_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DECODABLE_AM_DIAG_GMM_H_
