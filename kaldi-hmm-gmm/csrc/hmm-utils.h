// kaldi-hmm-gmm/csrc/hmm-utils.h
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/hmm/hmm-utils.h
#ifndef KALDI_HMM_GMM_CSRC_HMM_UTILS_H_
#define KALDI_HMM_GMM_CSRC_HMM_UTILS_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
namespace khg {

/// Configuration class for the GetHTransducer() function; see
/// \ref hmm_graph_config for context.
struct HTransducerConfig {
  /// Transition log-prob scale, see \ref hmm_scale.
  /// Note this doesn't apply to self-loops; GetHTransducer() does
  /// not include self-loops.
  /// Scale of transition probs (relative to LM)
  float transition_scale = 1.0;

  // The integer id of #nonterm_bos in phones.txt, if present.
  // Only needs to be set if you are doing grammar decoding,
  // see doc/grammar.dox.
  int32_t nonterm_phones_offset = -1;

  HTransducerConfig(float transition_scale = 1.0,
                    int32_t nonterm_phones_offset = -1)
      : transition_scale(transition_scale),
        nonterm_phones_offset(nonterm_phones_offset) {}
  std::string ToString() const {
    std::ostringstream os;
    os << "HTransducerConfig(";
    os << "transition_scale=" << transition_scale << ", ";
    os << "nonterm_phones_offset=" << nonterm_phones_offset << ")";
    return os.str();
  }
};

/**
 * Returns the H tranducer; result owned by caller.  Caution: our version of
 * the H transducer does not include self-loops; you have to add those later.
 * See \ref hmm_graph_get_h_transducer.  The H transducer has on the
 * input transition-ids, and also possibly some disambiguation symbols, which
 * will be put in disambig_syms.  The output side contains the identifiers that
 * are indexes into "ilabel_info" (these represent phones-in-context or
 * disambiguation symbols).  The ilabel_info vector allows GetHTransducer to map
 * from symbols to phones-in-context (i.e. phonetic context windows).  Any
 * singleton symbols in the ilabel_info vector which are not phones, will be
 * treated as disambiguation symbols.  [Not all recipes use these].  The output
 * "disambig_syms_left" will be set to a list of the disambiguation symbols on
 * the input of the transducer (i.e. same symbol type as whatever is on the
 * input of the transducer
 */
fst::VectorFst<fst::StdArc> *GetHTransducer(
    const std::vector<std::vector<int32_t>> &ilabel_info,
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model, const HTransducerConfig &config,
    std::vector<int32_t> *disambig_syms_left);

struct HmmCacheHash {
  int operator()(const std::pair<int32_t, std::vector<int32_t>> &p) const {
    VectorHasher<int32_t> v;
    int32_t prime = 103049;
    return prime * p.first + v(p.second);
  }
};

/// HmmCacheType is a map from (central-phone, sequence of pdf-ids) to FST, used
/// as cache in GetHmmAsFsa, as an optimization.
typedef unordered_map<std::pair<int32_t, std::vector<int32_t>>,
                      fst::VectorFst<fst::StdArc> *, HmmCacheHash>
    HmmCacheType;

/**
 * For context, see \ref hmm_graph_add_self_loops.  Expands an FST that has been
 * built without self-loops, and adds the self-loops (it also needs to modify
 * the probability of the non-self-loop ones, as the graph without self-loops
 * was created in such a way that it was stochastic).  Note that the
 * disambig_syms will be empty in some recipes (e.g.  if you already removed
 * the disambiguation symbols).
 * This function will treat numbers over 10000000 (kNontermBigNumber) the
 * same as disambiguation symbols, assuming they are special symbols for
 * grammar decoding.
 *
 * @param trans_model [in] Transition model
 * @param disambig_syms [in] Sorted, uniq list of disambiguation symbols,
 * required if the graph contains disambiguation symbols but only needed for
 * sanity checks.
 * @param self_loop_scale [in] Transition-probability scale for self-loops; c.f.
 *                    \ref hmm_scale
 * @param reorder [in] If true, reorders the transitions (see \ref hmm_reorder).
 *                     You'll normally want this to be true.
 * @param check_no_self_loops [in]  If true, it will check that there are no
 *                      self-loops in the original graph; you'll normally want
 *                      this to be true.  If false, it will allow them, and
 *                      will add self-loops after the original self-loop
 *                      transitions, assuming reorder==true... this happens to
 *                      be what we want when converting normal to unconstrained
 *                      chain examples.  WARNING: this was added in 2018;
 *                      if you get a compilation error, add this as 'true',
 *                      which emulates the behavior of older code.
 * @param  fst [in, out] The FST to be modified.
 */
void AddSelfLoops(
    const TransitionModel &trans_model,
    const std::vector<int32_t> &disambig_syms,  // used as a check only.
    float self_loop_scale, bool reorder, bool check_no_self_loops,
    fst::VectorFst<fst::StdArc> *fst);

/**
 * Adds transition-probs, with the supplied
 * scales (see \ref hmm_scale), to the graph.
 * Useful if you want to create a graph without transition probs, then possibly
 * train the model (including the transition probs) but keep the graph fixed,
 * and add back in the transition probs.  It assumes the fst has transition-ids
 * on it.  It is not an error if the FST has no states (nothing will be done).
 * @param trans_model [in] The transition model
 * @param disambig_syms [in] A list of disambiguation symbols, required if the
 *                       graph has disambiguation symbols on its input but only
 *                       used for checks.
 * @param transition_scale [in] A scale on transition-probabilities apart from
 *                      those involving self-loops; see \ref hmm_scale.
 * @param self_loop_scale [in] A scale on self-loop transition probabilities;
 *                      see \ref hmm_scale.
 * @param  fst [in, out] The FST to be modified.
 */
void AddTransitionProbs(const TransitionModel &trans_model,
                        const std::vector<int32_t> &disambig_syms,
                        float transition_scale, float self_loop_scale,
                        fst::VectorFst<fst::StdArc> *fst);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_HMM_UTILS_H_
