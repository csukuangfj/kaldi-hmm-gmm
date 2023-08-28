// kaldi-hmm-gmm/csrc/decoder-wrappers.cc

// Copyright   2014  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/decoder-wrappers.h"

#include "kaldi-hmm-gmm/csrc/faster-decoder.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldifst/csrc/fstext-utils.h"

namespace khg {

void AlignUtteranceWrapper(
    const AlignConfig &config, const std::string &utt,
    float acoustic_scale,  // affects scores written to scores_writer, if
                           // present
    fst::VectorFst<fst::StdArc> *fst,  // non-const in case config.careful ==
                                       // true, we add loop.
    DecodableInterface *decodable,     // not const but is really an input.
    int32_t *num_done, int32_t *num_error, int32_t *num_retried,
    double *tot_like, int64_t *frame_count, std::vector<int32_t> *alignment,
    std::vector<int32_t> *words) {
  alignment->clear();
  words->clear();

  if ((config.retry_beam != 0 && config.retry_beam <= config.beam) ||
      config.beam <= 0.0) {
    KHG_ERR << "Beams do not make sense: beam " << config.beam
            << ", retry-beam " << config.retry_beam;
  }

  if (fst->Start() == fst::kNoStateId) {
    KHG_WARN << "Empty decoding graph for " << utt;
    if (num_error != nullptr) {
      (*num_error)++;
    }
    return;
  }

  if (config.careful) {
    ModifyGraphForCarefulAlignment(fst);
  }

  FasterDecoderOptions decode_opts;
  decode_opts.beam = config.beam;

  FasterDecoder decoder(*fst, decode_opts);
  decoder.Decode(decodable);

  bool ans = decoder.ReachedFinal();  // consider only final states.

  if (!ans && config.retry_beam != 0.0) {
    if (num_retried != nullptr) {
      (*num_retried)++;
    }

    KHG_WARN << "Retrying utterance " << utt << " with beam "
             << config.retry_beam;

    decode_opts.beam = config.retry_beam;
    decoder.SetOptions(decode_opts);
    decoder.Decode(decodable);
    ans = decoder.ReachedFinal();
  }

  if (!ans) {  // Still did not reach final state.
    KHG_WARN << "Did not successfully decode file " << utt
             << ", len = " << decodable->NumFramesReady();
    if (num_error != nullptr) {
      (*num_error)++;
    }

    return;
  }

  fst::VectorFst<fst::LatticeArc> decoded;  // linear FST.
  decoder.GetBestPath(&decoded);

  if (decoded.NumStates() == 0) {
    KHG_WARN << "Error getting best path from decoder (likely a bug)";
    if (num_error != NULL) {
      (*num_error)++;
    }

    return;
  }

  fst::LatticeWeight weight;

  fst::GetLinearSymbolSequence(decoded, alignment, words, &weight);

  float like = -(weight.Value1() + weight.Value2()) / acoustic_scale;

  if (num_done != nullptr) {
    (*num_done)++;
  }

  if (tot_like != nullptr) {
    (*tot_like) += like;
  }

  if (frame_count != nullptr) {
    (*frame_count) += decodable->NumFramesReady();
  }
}

// see comment in header.
void ModifyGraphForCarefulAlignment(fst::VectorFst<fst::StdArc> *fst) {
  typedef fst::StdArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;
  typedef Arc::Weight Weight;
  StateId num_states = fst->NumStates();
  if (num_states == 0) {
    KHG_WARN << "Empty FST input.";
    return;
  }

  Weight zero = Weight::Zero();
  // fst_rhs will be the right hand side of the Concat operation.
  fst::VectorFst<fst::StdArc> fst_rhs(*fst);
  // first remove the final-probs from fst_rhs.
  for (StateId state = 0; state < num_states; ++state) {
    fst_rhs.SetFinal(state, zero);
  }

  StateId pre_initial = fst_rhs.AddState();
  Arc to_initial(0, 0, Weight::One(), fst_rhs.Start());
  fst_rhs.AddArc(pre_initial, to_initial);
  fst_rhs.SetStart(pre_initial);
  // make the pre_initial state final with probability one;
  // this is equivalent to keeping the final-probs of the first
  // FST when we do concat (otherwise they would get deleted).
  fst_rhs.SetFinal(pre_initial, Weight::One());

  fst::Concat(fst, fst_rhs);
}

}  // namespace khg
