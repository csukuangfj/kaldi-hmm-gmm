// kaldi-hmm-gmm/csrc/hmm-utils.h
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2023  Xiaomi Corporation

/**
 * \file hmm-utils.h
 * \brief Building FSTs from context-dependent HMMs, and alignment utilities.
 *
 * This file is copied/modified from kaldi/src/hmm/hmm-utils.h.
 *
 * This module turns context-dependent HMMs into FSTs and manipulates the
 * transition probabilities on such FSTs:
 *
 * - GetHTransducer() builds the H transducer: input labels are
 *   transition-ids (plus possibly disambiguation symbols), output labels are
 *   indexes into the \c ilabel_info vector (phones-in-context or
 *   disambiguation symbols).
 * - AddSelfLoops() expands a graph built without self-loops by adding them.
 * - AddTransitionProbs() puts (or re-puts) scaled transition probabilities
 *   onto a graph whose input labels are transition-ids.
 * - SplitToPhones() splits an alignment (a sequence of transition-ids) into
 *   per-phone segments; GetPdfToPhonesMap() maps pdf-ids to the monophones
 *   they can belong to.
 *
 * Cost conventions (matching kaldi):
 *
 * - Transition probabilities enter graphs as tropical weights (costs)
 *   \f$-s \log p\f$, where \f$p\f$ is the transition probability and \f$s\f$
 *   the applicable scale (HTransducerConfig::transition_scale for forward
 *   transitions, the \c self_loop_scale argument for self-loops); a scale of
 *   0 disables the corresponding probabilities.
 * - In GetHTransducer() the transition costs are baked ONLY on the HMM
 *   transition arcs (the arcs labeled with transition-ids); the
 *   context-expansion arcs (whose output labels index into \c ilabel_info)
 *   have weight zero.  Self-loops are not included at all.  This matches
 *   kaldi; the placement of the costs is intentional.
 * - AddTransitionProbs() multiplies the transition probabilities into the
 *   existing arc weights; it does not compensate for probabilities that are
 *   already present.  In the standard training pipeline the training graphs
 *   are compiled with both scales set to 0 (the compile-train-graphs
 *   convention; see TrainingGraphCompilerOptions) and AddTransitionProbs() is
 *   applied at alignment time.  If it is applied to a graph that already has
 *   transition probabilities baked in, they end up applied twice; this double
 *   application matches kaldi and is intentional - do not "fix" it.
 *
 * See also kaldi/src/hmm/hmm-utils.h.
 */

#ifndef KALDI_HMM_GMM_CSRC_HMM_UTILS_H_
#define KALDI_HMM_GMM_CSRC_HMM_UTILS_H_

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
namespace khg {

/**
 * \brief Configuration class for the GetHTransducer() function.
 *
 * The scales follow the convention described in the file documentation:
 * transition probabilities appear in the graph as costs
 * \f$-s \log p\f$.
 */
struct HTransducerConfig {
  /// Transition log-prob scale (scale of transition probs, relative to LM).
  /// Note this doesn't apply to self-loops; GetHTransducer() does
  /// not include self-loops.
  float transition_scale = 1.0;

  /// The integer id of \c nonterm_bos in phones.txt, if present.
  /// Only needs to be set if you are doing grammar decoding
  /// (see doc/grammar.dox in the kaldi source).
  int32_t nonterm_phones_offset = -1;

  /// Constructor with the defaults shown on the members.
  HTransducerConfig(float transition_scale = 1.0,
                    int32_t nonterm_phones_offset = -1)
      : transition_scale(transition_scale),
        nonterm_phones_offset(nonterm_phones_offset) {}

  /// Returns a human-readable string representation, for logging.
  std::string ToString() const {
    std::ostringstream os;
    os << "HTransducerConfig(";
    os << "transition_scale=" << transition_scale << ", ";
    os << "nonterm_phones_offset=" << nonterm_phones_offset << ")";
    return os.str();
  }
};

/**
 * \brief Returns the H transducer; result owned by caller.
 *
 * Caution: our version of the H transducer does not include self-loops; you
 * have to add those later (see AddSelfLoops()).  The H transducer has on the
 * input transition-ids, and also possibly some disambiguation symbols, which
 * will be put in disambig_syms.  The output side contains the identifiers
 * that are indexes into "ilabel_info" (these represent phones-in-context or
 * disambiguation symbols).  The ilabel_info vector allows GetHTransducer to
 * map from symbols to phones-in-context (i.e. phonetic context windows).  Any
 * singleton symbols in the ilabel_info vector which are not phones, will be
 * treated as disambiguation symbols.  [Not all recipes use these].  The
 * output "disambig_syms_left" will be set to a list of the disambiguation
 * symbols on the input of the transducer (i.e. same symbol type as whatever
 * is on the input of the transducer).
 *
 * The transition costs are baked ONLY on the HMM transition arcs: an arc
 * labeled with a transition-id carries cost \f$-s_t \log \tilde{p}\f$, where
 * \f$\tilde{p}\f$ is the transition probability renormalized to exclude the
 * self-loop probability mass of its state (see
 * TransitionModel::GetTransitionLogProbIgnoringSelfLoops) and \f$s_t\f$ is
 * \p config's transition_scale.  The context-expansion arcs (labeled on the
 * output side with indexes into \p ilabel_info) have weight zero.  This
 * matches kaldi; the cost placement is intentional.
 *
 * \param ilabel_info [in] The ilabel info: element \c i is the
 *        phone-in-context (or singleton disambiguation symbol) that output
 *        symbol \c i stands for.
 * \param ctx_dep     [in] The phonetic decision tree; maps pdf-classes to
 *        pdf-ids.
 * \param trans_model [in] The transition model; provides the transition-ids
 *        and the transition probabilities.
 * \param config      [in] Configuration object; see HTransducerConfig.
 * \param disambig_syms_left [out] Set to the list of disambiguation symbols
 *        appearing on the input side of the transducer.
 * \return The H transducer (ownership passed to the caller).
 */
fst::VectorFst<fst::StdArc> *GetHTransducer(
    const std::vector<std::vector<int32_t>> &ilabel_info,
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model, const HTransducerConfig &config,
    std::vector<int32_t> *disambig_syms_left);

/// Hash function for the keys of HmmCacheType: combines the central phone
/// and the sequence of pdf-ids of a context-dependent HMM.
struct HmmCacheHash {
  /// Hashes a (central-phone, pdf-id sequence) pair.
  int operator()(const std::pair<int32_t, std::vector<int32_t>> &p) const {
    VectorHasher<int32_t> v;
    int32_t prime = 103049;
    return prime * p.first + v(p.second);
  }
};

/// HmmCacheType is a map from (central-phone, sequence of pdf-ids) to FST,
/// used as cache in GetHmmAsFsa, as an optimization.
typedef std::unordered_map<std::pair<int32_t, std::vector<int32_t>>,
                      fst::VectorFst<fst::StdArc> *, HmmCacheHash>
    HmmCacheType;

/**
 * \brief Adds self-loops to an FST that was built without them.
 *
 * For context, this is the last step of building HCLG-style graphs from the H
 * transducer of GetHTransducer().  Expands an FST that has been built without
 * self-loops, and adds the self-loops (it also needs to modify the
 * probability of the non-self-loop ones, as the graph without self-loops was
 * created in such a way that it was stochastic).  Concretely, for every state
 * with a transition-state, a self-loop is added with cost
 * \f$-s_{\mathrm{sl}} \log p_{\mathrm{sl}}\f$ and the other out-arcs (and the
 * final weight) of the state are multiplied by the forward probability mass,
 * i.e. their cost is increased by
 * \f$-s_{\mathrm{sl}} \log (1 - p_{\mathrm{sl}})\f$, where \f$p_{\mathrm{sl}}\f$
 * is the self-loop probability and \f$s_{\mathrm{sl}}\f$ is \p self_loop_scale.
 * Note that the disambig_syms will be empty in some recipes (e.g. if you
 * already removed the disambiguation symbols).
 * This function will treat numbers over 10000000 (kNontermBigNumber) the
 * same as disambiguation symbols, assuming they are special symbols for
 * grammar decoding.
 *
 * \param trans_model [in] Transition model
 * \param disambig_syms [in] Sorted, uniq list of disambiguation symbols,
 *        required if the graph contains disambiguation symbols but only
 *        needed for sanity checks.
 * \param self_loop_scale [in] Transition-probability scale for self-loops;
 *        see the scale conventions in the file documentation.
 * \param reorder [in] If true, reorders the transitions (the transition-id of
 *        a forward transition is placed on the arc entering its destination
 *        state, before the self-loop; Dan-style graphs).  You'll normally
 *        want this to be true.
 * \param check_no_self_loops [in]  If true, it will check that there are no
 *        self-loops in the original graph; you'll normally want this to be
 *        true.  If false, it will allow them, and will add self-loops after
 *        the original self-loop transitions, assuming reorder==true... this
 *        happens to be what we want when converting normal to unconstrained
 *        chain examples.  WARNING: this was added in 2018; if you get a
 *        compilation error, add this as 'true', which emulates the behavior
 *        of older code.
 * \param fst [in, out] The FST to be modified.
 */
void AddSelfLoops(
    const TransitionModel &trans_model,
    const std::vector<int32_t> &disambig_syms,
    float self_loop_scale, bool reorder, bool check_no_self_loops,
    fst::VectorFst<fst::StdArc> *fst);

/**
 * \brief Adds transition-probs to the graph.
 *
 * Adds transition-probs, with the supplied scales (see the scale
 * conventions in the file documentation), to the graph.  Every arc labeled
 * with a transition-id gets its weight multiplied by the corresponding
 * transition probability: if \p transition_scale and \p self_loop_scale are
 * equal this is just \f$s \log p\f$; otherwise self-loops get
 * \f$s_{\mathrm{sl}} \log p\f$ and non-self-loop transitions get
 * \f$s_{\mathrm{sl}} \log(1 - p_{\mathrm{sl}}) + s_t \log \tilde{p}\f$ (the
 * renormalized scheme of GetHTransducer()/AddSelfLoops()).  In all cases the
 * log-probs are simply added to what the arcs already carry.
 *
 * Useful if you want to create a graph without transition probs, then
 * possibly train the model (including the transition probs) but keep the
 * graph fixed, and add back in the transition probs.  It assumes the fst has
 * transition-ids on it.  It is not an error if the FST has no states (nothing
 * will be done).  Note that in the standard training pipeline this is called
 * at alignment time (the training graphs are compiled with both scales set to
 * 0, because the transition probs change at every iteration while the graphs
 * stay fixed); applying it to a graph that already has transition probs
 * applies them a second time, which matches kaldi and is intentional - do not
 * "fix" it.
 *
 * \param trans_model [in] The transition model
 * \param disambig_syms [in] A list of disambiguation symbols, required if the
 *        graph has disambiguation symbols on its input but only used for
 *        checks.
 * \param transition_scale [in] A scale on transition-probabilities apart from
 *        those involving self-loops.
 * \param self_loop_scale [in] A scale on self-loop transition probabilities.
 * \param fst [in, out] The FST to be modified.
 */
void AddTransitionProbs(const TransitionModel &trans_model,
                        const std::vector<int32_t> &disambig_syms,
                        float transition_scale, float self_loop_scale,
                        fst::VectorFst<fst::StdArc> *fst);

/// SplitToPhones splits up the TransitionIds in "alignment" into their
/// individual phones (one vector per instance of a phone).  At output,
/// the sum of the sizes of the vectors in split_alignment will be the same
/// as the corresponding sum for "alignment".  The function returns
/// true on success.  If the alignment appears to be incomplete, e.g.
/// not ending at the end-state of a phone, it will still break it up into
/// phones but it will return false.  For more serious errors it will
/// die or throw an exception.
/// This function works out by itself whether the graph was created
/// with "reordering", and just does the right thing.
///
/// This is ported from ../kaldi/src/hmm/hmm-utils.cc.
///
/// \param trans_model    [in] The transition model.
/// \param alignment      [in] The alignment: a sequence of transition-ids.
/// \param split_alignment [out] The alignment split into per-phone segments,
///        in order; one vector of transition-ids per instance of a phone.
/// \return True on success; false if the alignment was incomplete (e.g. not
///         ending at the end-state of a phone), in which case the split is
///         still produced.
bool SplitToPhones(const TransitionModel &trans_model,
                   const std::vector<int32_t> &alignment,
                   std::vector<std::vector<int32_t>> *split_alignment);

/// GetPdfToPhonesMap creates a map which maps each pdf-id into its
/// corresponding monophones.
///
/// This is ported from ../kaldi/src/hmm/hmm-utils.cc.
///
/// \param trans_model [in] The transition model.
/// \param pdf2phones  [out] Indexed by pdf-id; each element is the set of
///        phones the pdf-id can occur in.
void GetPdfToPhonesMap(const TransitionModel &trans_model,
                       std::vector<std::set<int32_t>> *pdf2phones);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_HMM_UTILS_H_
