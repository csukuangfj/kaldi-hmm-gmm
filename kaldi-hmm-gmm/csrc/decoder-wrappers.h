// kaldi-hmm-gmm/csrc/decoder-wrappers.h

// Copyright   2014  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/decoder/decoder-wrappers.h
#ifndef KALDI_HMM_GMM_CSRC_DECODER_WRAPPERS_H_
#define KALDI_HMM_GMM_CSRC_DECODER_WRAPPERS_H_

#include <string>
#include <vector>

#include "fst/fst.h"
#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/decodable-itf.h"
#include "kaldi-hmm-gmm/csrc/lattice-faster-decoder.h"
#include "kaldi-hmm-gmm/csrc/lattice-simple-decoder.h"
#include "kaldi-hmm-gmm/csrc/transition-information.h"

namespace khg {

struct AlignConfig {
  // Decoding beam used in alignment
  float beam;
  // Decoding beam for second try at alignment
  float retry_beam;

  // If true, do 'careful' alignment, which is better at detecting
  // alignment failure (involves loop to start of decoding graph).
  bool careful;

  /*implicit*/ AlignConfig(float beam = 200.0,      // NOLINT
                           float retry_beam = 0.0,  // NOLINT
                           bool careful = false)    // NOLINT
      : beam(beam), retry_beam(retry_beam), careful(careful) {}
};

/// AlignUtteranceWapper is a wrapper for alignment code used in training, that
/// is called from many different binaries, e.g. gmm-align, gmm-align-compiled,
/// sgmm-align, etc.  The writers for alignments and words will only be written
/// to if they are open.  The num_done, num_error, num_retried, tot_like and
/// frame_count pointers will (if non-NULL) be incremented or added to, not set,
/// by this function.
void AlignUtteranceWrapper(
    const AlignConfig &config, const std::string &utt,
    float acoustic_scale,  // affects scores written to scores_writer, if
                           // present
    fst::VectorFst<fst::StdArc> *fst,  // non-const in case config.careful ==
                                       // true, we add loop.
    DecodableInterface *decodable,     // not const but is really an input.
    int32_t *num_done, int32_t *num_error, int32_t *num_retried,
    double *tot_like, int64_t *frame_count, std::vector<int32_t> *alignment,
    std::vector<int32_t> *words);

/// This function modifies the decoding graph for what we call "careful
/// alignment".  The problem we are trying to solve is that if the decoding eats
/// up the words in the graph too fast, it can get stuck at the end, and produce
/// what looks like a valid alignment even though there was really a failure.
/// So what we want to do is to introduce, after the final-states of the graph,
/// a "blind alley" with no final-probs reachable, where the decoding can go to
/// get lost.  Our basic idea is to append the decoding-graph to itself using
/// the fst Concat operation; but in order that there should be final-probs at
/// the end of the first but not the second FST, we modify the right-hand
/// argument to the Concat operation so that it has none of the original
/// final-probs, and add a "pre-initial" state that is final.
void ModifyGraphForCarefulAlignment(fst::VectorFst<fst::StdArc> *fst);

// This function DecodeUtteranceLatticeSimple is used in several decoders, and
// we have moved it here.  Note: this is really "binary-level" code as it
// involves table readers and writers; we've just put it here as there is no
// other obvious place to put it.  If determinize == false, it writes to
// lattice_writer, else to compact_lattice_writer.  The writers for
// alignments and words will only be written to if they are open.
bool DecodeUtteranceLatticeSimple(
    LatticeSimpleDecoder &decoder,  // NOLINT not const but is really an input.
    DecodableInterface &decodable,  // NOLINT not const but is really an input.
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, std::vector<int32_t> *alignments,
    std::vector<int32_t> *words,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.

/// This function DecodeUtteranceLatticeFaster is used in several decoders, and
/// we have moved it here.  Note: this is really "binary-level" code as it
/// involves table readers and writers; we've just put it here as there is no
/// other obvious place to put it.  If determinize == false, it writes to
/// lattice_writer, else to compact_lattice_writer.  The writers for
/// alignments and words will only be written to if they are open.
///
/// Caution: this will only link correctly if FST is either
/// fst::Fst<fst::StdArc>, or fst::GrammarFst, as the template function is
/// defined in the .cc file and only instantiated for those two types.
template <typename FST>
bool DecodeUtteranceLatticeFaster(
    LatticeFasterDecoderTpl<FST>
        &decoder,                   // NOLINT not const but is really an input.
    DecodableInterface &decodable,  // NOLINT not const but is really an input.
    const TransitionInformation &trans_model, const std::string &utt,
    bool allow_partial, std::vector<int32_t> *alignments,
    std::vector<int32_t> *words,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DECODER_WRAPPERS_H_
