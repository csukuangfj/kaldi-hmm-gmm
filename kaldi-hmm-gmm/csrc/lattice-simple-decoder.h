// kaldi-hmm-gmm/csrc/lattice-simple-decoder.h

// Copyright 2009-2012  Microsoft Corporation
//           2012-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/decoder/lattice-simple-decoder.h

#ifndef KALDI_HMM_GMM_CSRC_LATTICE_SIMPLE_DECODER_H_
#define KALDI_HMM_GMM_CSRC_LATTICE_SIMPLE_DECODER_H_

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "kaldi-hmm-gmm/csrc/decodable-itf.h"
#include "kaldi-hmm-gmm/csrc/determinize-lattice-pruned.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldifst/csrc/lattice-weight.h"

namespace khg {

struct LatticeSimpleDecoderConfig {
  float beam;  // decoding beam

  float lattice_beam;  // lattice generation beam

  // Interval (in frames) at which to prune tokens
  int32_t prune_interval;

  // if true, determinize the lattice (in a special sense, keeping only
  // best pdf-sequence for each word-sequence).
  bool determinize_lattice;  // not inspected by this class... used in

  // command-line program.
  bool prune_lattice;
  float beam_ratio;
  float prune_scale;  // Note: we don't make this configurable on the command
                      // line, it's not a very important parameter.  It affects
                      // the algorithm that prunes the tokens as we go.
  DeterminizeLatticePhonePrunedOptions det_opts;

  LatticeSimpleDecoderConfig(
      float beam = 16.0, float lattice_beam = 10.0, int32_t prune_interval = 25,
      bool determinize_lattice = true, float beam_ratio = 0.9,
      float prune_scale = 0.1,
      const DeterminizeLatticePhonePrunedOptions &det_opts = {})
      : beam(beam),
        lattice_beam(lattice_beam),
        prune_interval(prune_interval),
        determinize_lattice(determinize_lattice),
        beam_ratio(beam_ratio),
        prune_scale(prune_scale),
        det_opts(det_opts) {}

  std::string ToString() const {
    std::ostringstream os;

    os << "LatticeSimpleDecoderConfig(";
    os << "beam=" << beam << ", ";
    os << "lattice_beam=" << lattice_beam << ", ";
    os << "prune_interval=" << prune_interval << ", ";
    os << "determinize_lattice=" << (determinize_lattice ? "True" : "False")
       << ", ";
    os << "prune_lattice=" << (prune_lattice ? "True" : "False") << ", ";
    os << "beam_ratio=" << beam_ratio << ", ";
    os << "prune_scale=" << prune_scale << ", ";
    os << "det_opts=" << det_opts.ToString() << ")";

    return os.str();
  }

  void Check() const {
    KHG_ASSERT(beam > 0.0 && lattice_beam > 0.0 && prune_interval > 0);
  }
};

/** Simplest possible decoder, included largely for didactic purposes and as a
    means to debug more highly optimized decoders.  See \ref decoders_simple
    for more information.
 */
class LatticeSimpleDecoder {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  // instantiate this class once for each thing you have to decode.
  LatticeSimpleDecoder(const fst::Fst<fst::StdArc> &fst,
                       const LatticeSimpleDecoderConfig &config)
      : fst_(fst), config_(config), num_toks_(0) {
    config.Check();
  }

  ~LatticeSimpleDecoder() { ClearActiveTokens(); }

  void ClearActiveTokens();  // a cleanup routine, at utt end/begin

  const LatticeSimpleDecoderConfig &GetOptions() const { return config_; }

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need
  /// to call this.  You can call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  void InitDecoding();

  /// This function may be optionally called after AdvanceDecoding(), when you
  /// do not plan to decode any further.  It does an extra pruning step that
  /// will help to prune the lattices output by GetLattice and (particularly)
  /// GetRawLattice more accurately, particularly toward the end of the
  /// utterance.  It does this by using the final-probs in pruning (if any
  /// final-state survived); it also does a final pruning step that visits all
  /// states (the pruning that is done during decoding may fail to prune states
  /// that are within kPruningScale = 0.1 outside of the beam).  If you call
  /// this, you cannot call AdvanceDecoding again (it will fail), and you
  /// cannot call GetLattice() and related functions with use_final_probs =
  /// false.
  /// Used to be called PruneActiveTokensFinal().
  void FinalizeDecoding();

  // Returns true if any kind of traceback is available (not necessarily from
  // a final state).
  bool Decode(DecodableInterface *decodable);

  inline int32_t NumFramesDecoded() const {
    return static_cast<int32_t>(active_toks_.size()) - 1;
  }

  /// says whether a final-state was active on the last frame.  If it was not,
  /// the lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<float>::infinity();
  }

  /// FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost
  /// plus cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were
  /// present on the final frame.  It will usually be nonnegative.  If it not
  /// too positive (e.g. < 5 is my first guess, but this is not tested) you can
  /// take it as a good indication that we reached the final-state with
  /// reasonable likelihood.
  float FinalRelativeCost() const;

  // Outputs an FST corresponding to the single best path
  // through the lattice.  Returns true if result is nonempty
  // (using the return status is deprecated, it will become void).
  // If "use_final_probs" is true AND we reached the final-state
  // of the graph then it will include those as final-probs, else
  // it will treat all final-probs as one.
  bool GetBestPath(fst::Lattice *lat, bool use_final_probs = true) const;

  // Outputs an FST corresponding to the raw, state-level
  // tracebacks.  Returns true if result is nonempty
  // (using the return status is deprecated, it will become void).
  // If "use_final_probs" is true AND we reached the final-state
  // of the graph then it will include those as final-probs, else
  // it will treat all final-probs as one.
  bool GetRawLattice(fst::Lattice *lat, bool use_final_probs = true) const;

 private:
  struct Token;

  // ForwardLinks are the links from a token to a token on the next frame.
  // or sometimes on the current frame (for input-epsilon links).
  struct ForwardLink {
    Token *next_tok;      // the next token [or NULL if represents final-state]
    Label ilabel;         // ilabel on link.
    Label olabel;         // olabel on link.
    float graph_cost;     // graph cost of traversing link (contains LM, etc.)
    float acoustic_cost;  // acoustic cost (pre-scaled) of traversing link
    ForwardLink *next;    // next in singly-linked list of forward links from a
                          // token.
    ForwardLink(Token *next_tok, Label ilabel, Label olabel, float graph_cost,
                float acoustic_cost, ForwardLink *next)
        : next_tok(next_tok),
          ilabel(ilabel),
          olabel(olabel),
          graph_cost(graph_cost),
          acoustic_cost(acoustic_cost),
          next(next) {}
  };

  // Token is what's resident in a particular state at a particular time.
  // In this decoder a Token actually contains *forward* links.
  // When first created, a Token just has the (total) cost.    We add forward
  // links from it when we process the next frame.
  struct Token {
    float tot_cost;    // would equal weight.Value()... cost up to this point.
    float extra_cost;  // >= 0.  After calling PruneForwardLinks, this equals
    // the minimum difference between the cost of the best path this is on,
    // and the cost of the absolute best path, under the assumption
    // that any of the currently active states at the decoding front may
    // eventually succeed (e.g. if you were to take the currently active states
    // one by one and compute this difference, and then take the minimum).

    ForwardLink *links;  // Head of singly linked list of ForwardLinks

    Token *next;  // Next in list of tokens for this frame.

    Token(float tot_cost, float extra_cost, ForwardLink *links, Token *next)
        : tot_cost(tot_cost),
          extra_cost(extra_cost),
          links(links),
          next(next) {}

    Token() = default;

    void DeleteForwardLinks() {
      ForwardLink *l = links, *m;
      while (l != nullptr) {
        m = l->next;
        delete l;
        l = m;
      }
      links = nullptr;
    }
  };

  // head and tail of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;

    TokenList()
        : toks(nullptr),
          must_prune_forward_links(true),
          must_prune_tokens(true) {}
  };

 private:
  // FindOrAddToken either locates a token in cur_toks_, or if necessary inserts
  // a new, empty token (i.e. with no forward links) for the current frame.
  // [note: it's inserted if necessary into cur_toks_ and also into the singly
  // linked list of tokens active on this frame (whose head is at
  // active_toks_[frame]).
  //
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  inline Token *FindOrAddToken(StateId state, int32_t frame_plus_one,
                               float tot_cost, bool emitting, bool *changed);

  // delta is the amount by which the extra_costs must
  // change before it sets "extra_costs_changed" to true.  If delta is larger,
  // we'll tend to go back less far toward the beginning of the file.
  void PruneForwardLinks(int32_t frame, bool *extra_costs_changed,
                         bool *links_pruned, float delta);

  // Prune away any tokens on this frame that have no forward links. [we don't
  // do this in PruneForwardLinks because it would give us a problem with
  // dangling pointers].
  void PruneTokensForFrame(int32_t frame);

  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.  It's
  // possible to prove that this is "correct" in the sense that we won't lose
  // anything outside of lat_beam, regardless of what happens in the future.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse
  // less far.
  void PruneActiveTokens(float delta);

  void ProcessNonemitting();

  void ProcessEmitting(DecodableInterface *decodable);

  // PruneCurrentTokens deletes the tokens from the "toks" map, but not
  // from the active_toks_ list, which could cause dangling forward pointers
  // (will delete it during regular pruning operation).
  void PruneCurrentTokens(float beam,
                          std::unordered_map<StateId, Token *> *toks);

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses the
  // final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal();

  // This function computes the final-costs for tokens active on the final
  // frame.  It outputs to final-costs, if non-NULL, a map from the Token*
  // pointer to the final-prob of the corresponding state, or zero for all
  // states if none were final.  It outputs to final_relative_cost, if non-NULL,
  // the difference between the best forward-cost including the final-prob cost,
  // and the best forward-cost without including the final-prob cost (this will
  // usually be positive), or infinity if there were no final-probs.  It outputs
  // to final_best_cost, if non-NULL, the lowest for any token t active on the
  // final frame, of t + final-cost[t], where final-cost[t] is the final-cost
  // in the graph of the state corresponding to token t, or zero if there
  // were no final-probs active on the final frame.
  // You cannot call this after FinalizeDecoding() has been called; in that
  // case you should get the answer from class-member variables.
  void ComputeFinalCosts(std::unordered_map<Token *, float> *final_costs,
                         float *final_relative_cost,
                         float *final_best_cost) const;

 private:
  const fst::Fst<fst::StdArc> &fst_;
  LatticeSimpleDecoderConfig config_;
  int32_t num_toks_;  // current total #toks allocated...

  // Lists of tokens, indexed by
  // frame_plus_one
  std::vector<TokenList> active_toks_;

  // indexed by state
  std::unordered_map<StateId, Token *> cur_toks_;

  std::unordered_map<StateId, Token *> prev_toks_;

  bool warned_;

  /// decoding_finalized_ is true if someone called FinalizeDecoding().  [note,
  /// calling this is optional].  If true, it's forbidden to decode more.  Also,
  /// if this is set, then the output of ComputeFinalCosts() is in the next
  /// three variables.  The reason we need to do this is that after
  /// FinalizeDecoding() calls PruneTokensForFrame() for the final frame, some
  /// of the tokens on the last frame are freed, so we free the list from
  /// cur_toks_ to avoid having dangling pointers hanging around.
  bool decoding_finalized_;

  /// For the meaning of the next 3 variables, see the comment for
  /// decoding_finalized_ above., and ComputeFinalCosts().
  std::unordered_map<Token *, float> final_costs_;
  float final_relative_cost_;
  float final_best_cost_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_LATTICE_SIMPLE_DECODER_H_
