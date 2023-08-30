// kaldi-hmm-gmm/csrc/hmm-utils.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/hmm/hmm-utils.cc

#include "kaldi-hmm-gmm/csrc/hmm-utils.h"

#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldifst/csrc/fstext-utils.h"
#include "kaldifst/csrc/grammar-context-fst.h"
#include "kaldifst/csrc/remove-eps-local.h"

namespace khg {

/// This utility function, used in GetHTransducer(), creates an FSA (finite
/// state acceptor, i.e. an FST with ilabels equal to olabels) with a single
/// successful path, with a single label on it.
static inline fst::VectorFst<fst::StdArc> *MakeTrivialAcceptor(int32_t label) {
  typedef fst::StdArc Arc;
  typedef Arc::Weight Weight;
  fst::VectorFst<Arc> *ans = new fst::VectorFst<Arc>;
  ans->AddState();
  ans->AddState();
  ans->SetStart(0);
  ans->SetFinal(1, Weight::One());
  ans->AddArc(0, Arc(label, label, Weight::One(), 1));
  return ans;
}

static fst::VectorFst<fst::StdArc> *GetHmmAsFsa(
    std::vector<int32_t> phone_window,
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model, const HTransducerConfig &config,
    HmmCacheType *cache) {
  using namespace fst;  // NOLINT

  if (static_cast<int32_t>(phone_window.size()) != ctx_dep.ContextWidth())
    KHG_ERR << "Context size mismatch, ilabel-info [from context FST is "
            << phone_window.size()
            << ", context-dependency object "
               "expects "
            << ctx_dep.ContextWidth();

  int32_t P = ctx_dep.CentralPosition();
  int32_t phone = phone_window[P];
  if (phone == 0)
    KHG_ERR << "phone == 0.  Some mismatch happened, or there is "
               "a code error.";

  const HmmTopology &topo = trans_model.GetTopo();
  const HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(phone);

  // vector of the pdfs, indexed by pdf-class (pdf-classes must start from zero
  // and be contiguous).
  std::vector<int32_t> pdfs(topo.NumPdfClasses(phone));
  for (int32_t pdf_class = 0; pdf_class < static_cast<int32_t>(pdfs.size());
       pdf_class++) {
    if (!ctx_dep.Compute(phone_window, pdf_class, &(pdfs[pdf_class]))) {
      std::ostringstream ctx_ss;
      for (size_t i = 0; i < phone_window.size(); i++)
        ctx_ss << phone_window[i] << ' ';
      KHG_ERR << "GetHmmAsFsa: context-dependency object could not produce "
              << "an answer: pdf-class = " << pdf_class
              << " ctx-window = " << ctx_ss.str()
              << ".  This probably points "
                 "to either a coding error in some graph-building process, "
                 "a mismatch of topology with context-dependency object, the "
                 "wrong FST being passed on a command-line, or something of "
                 " that general nature.";
    }
  }
  std::pair<int32_t, std::vector<int32_t>> cache_index(phone, pdfs);
  if (cache != nullptr) {
    HmmCacheType::iterator iter = cache->find(cache_index);
    if (iter != cache->end()) return iter->second;
  }

  VectorFst<StdArc> *ans = new VectorFst<StdArc>;

  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  std::vector<StateId> state_ids;
  for (size_t i = 0; i < entry.size(); i++)
    state_ids.push_back(ans->AddState());

  KHG_ASSERT(state_ids.size() != 0);  // Or empty topology entry.
  ans->SetStart(state_ids[0]);
  StateId final = state_ids.back();
  ans->SetFinal(final, Weight::One());

  for (int32_t hmm_state = 0; hmm_state < static_cast<int32_t>(entry.size());
       hmm_state++) {
    int32_t forward_pdf_class = entry[hmm_state].forward_pdf_class, forward_pdf;
    int32_t self_loop_pdf_class = entry[hmm_state].self_loop_pdf_class,
            self_loop_pdf;
    if (forward_pdf_class == kNoPdf) {  // nonemitting state.
      forward_pdf = kNoPdf;
      self_loop_pdf = kNoPdf;
    } else {
      KHG_ASSERT(forward_pdf_class < static_cast<int32_t>(pdfs.size()));
      KHG_ASSERT(self_loop_pdf_class < static_cast<int32_t>(pdfs.size()));
      forward_pdf = pdfs[forward_pdf_class];
      self_loop_pdf = pdfs[self_loop_pdf_class];
    }
    int32_t trans_idx;
    for (trans_idx = 0;
         trans_idx < static_cast<int32_t>(entry[hmm_state].transitions.size());
         trans_idx++) {
      float log_prob;
      Label label;
      int32_t dest_state = entry[hmm_state].transitions[trans_idx].first;
      bool is_self_loop = (dest_state == hmm_state);
      if (is_self_loop)
        continue;  // We will add self-loops in at a later stage of processing,
      // not in this function.
      if (forward_pdf_class == kNoPdf) {
        // no pdf, hence non-estimated probability.
        // [would not happen with normal topology] .  There is no
        // transition-state involved in this case.
        log_prob = log(entry[hmm_state].transitions[trans_idx].second);
        label = 0;
      } else {  // normal probability.
        int32_t trans_state = trans_model.TupleToTransitionState(
            phone, hmm_state, forward_pdf, self_loop_pdf);
        int32_t trans_id =
            trans_model.PairToTransitionId(trans_state, trans_idx);
        log_prob = trans_model.GetTransitionLogProbIgnoringSelfLoops(trans_id);
        // log_prob is a negative number (or zero)...
        label = trans_id;
      }
      // Will add probability-scale later (we may want to push first).
      ans->AddArc(state_ids[hmm_state],
                  Arc(label, label, Weight(-log_prob), state_ids[dest_state]));
    }
  }

  fst::RemoveEpsLocal(ans);  // this is safe and will not blow up.

  // Now apply probability scale.
  // We waited till after the possible weight-pushing steps,
  // because weight-pushing needs "real" weights in order to work.
  ApplyProbabilityScale(config.transition_scale, ans);
  if (cache != nullptr) (*cache)[cache_index] = ans;
  return ans;
}

// The H transducer has a separate outgoing arc for each of the symbols in
// ilabel_info.
fst::VectorFst<fst::StdArc> *GetHTransducer(
    const std::vector<std::vector<int32_t>> &ilabel_info,
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model, const HTransducerConfig &config,
    std::vector<int32_t> *disambig_syms_left) {
  KHG_ASSERT(ilabel_info.size() >= 1 &&
             ilabel_info[0].size() == 0);  // make sure that eps == eps.
  HmmCacheType cache;
  // "cache" is an optimization that prevents GetHmmAsFsa repeating work
  // unnecessarily.
  using namespace fst;  // NOLINT
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  std::vector<const ExpandedFst<Arc> *> fsts(ilabel_info.size(), nullptr);
  const std::vector<int32_t> &phones = trans_model.GetPhones();

  KHG_ASSERT(disambig_syms_left != 0);
  disambig_syms_left->clear();

  int32_t first_disambig_sym =
      trans_model.NumTransitionIds() +
      1;  // First disambig symbol we can have on the input side.
  int32_t next_disambig_sym = first_disambig_sym;

  if (ilabel_info.size() > 0)
    KHG_ASSERT(ilabel_info[0].size() == 0);  // make sure epsilon is epsilon...

  for (int32_t j = 1; j < static_cast<int32_t>(ilabel_info.size());
       j++) {  // zero is eps.
    KHG_ASSERT(!ilabel_info[j].empty());
    if (ilabel_info[j][0] < 0 ||
        (ilabel_info[j][0] == 0 && ilabel_info[j].size() == 1)) {
      // disambig symbol or special symbol for grammar FSTs.
      if (ilabel_info[j].size() == 1) {
        // disambiguation symbol.
        int32_t disambig_sym_left = next_disambig_sym++;
        disambig_syms_left->push_back(disambig_sym_left);
        fsts[j] = MakeTrivialAcceptor(disambig_sym_left);
      } else if (ilabel_info[j].size() == 2) {
        if (config.nonterm_phones_offset <= 0) {
          KHG_ERR << "ilabel-info seems to be for grammar-FST.  You need to "
                     "supply the --nonterm-phones-offset option.";
        }
        int32_t nonterm_phones_offset = config.nonterm_phones_offset,
                nonterminal = -ilabel_info[j][0],
                left_context_phone = ilabel_info[j][1];
        if (nonterminal <= nonterm_phones_offset || left_context_phone <= 0 ||
            left_context_phone > nonterm_phones_offset) {
          KHG_ERR << "Could not interpret this ilabel-info with "
                     "--nonterm-phones-offset="
                  << nonterm_phones_offset
                  << ": nonterminal,left-context-phone=" << nonterminal << ','
                  << left_context_phone;
        }
        int32_t big_number = static_cast<int32_t>(fst::kNontermBigNumber),
                encoding_multiple =
                    fst::GetEncodingMultiple(nonterm_phones_offset);
        int32_t encoded_symbol =
            big_number + nonterminal * encoding_multiple + left_context_phone;
        fsts[j] = MakeTrivialAcceptor(encoded_symbol);
      } else {
        KHG_ERR << "Could not decode this ilabel_info entry.";
      }
    } else {  // Real phone-in-context.
      std::vector<int32_t> phone_window = ilabel_info[j];

      VectorFst<Arc> *fst =
          GetHmmAsFsa(phone_window, ctx_dep, trans_model, config, &cache);
      fsts[j] = fst;
    }
  }

  VectorFst<Arc> *ans = MakeLoopFst(fsts);
  SortAndUniq(&fsts);  // remove duplicate pointers, which we will have
  // in general, since we used the cache.
  DeletePointers(&fsts);
  return ans;
}

class TidToTstateMapper {
 public:
  // Function object used in MakePrecedingInputSymbolsSameClass and
  // MakeFollowingInputSymbolsSameClass (as called by AddSelfLoopsReorder and
  // AddSelfLoopsNoReorder).  It maps transition-ids to transition-states (and
  // -1 to -1, 0 to 0 and disambiguation symbols to 0).  If check_no_self_loops
  // == true, it also checks that there are no self-loops in the graph (i.e. in
  // the labels it is called with).  This is just a convenient place to put this
  // check.

  // This maps valid transition-ids to transition states, maps kNoLabel to -1,
  // and maps all other symbols (i.e. epsilon symbols, disambig symbols, and
  // symbols with values over 100000/kNontermBigNumber) to zero. Its point is to
  // provide an equivalence class on labels that's relevant to what the
  // self-loop will be on the following (or preceding) state.
  TidToTstateMapper(const TransitionModel &trans_model,
                    const std::vector<int32_t> &disambig_syms,
                    bool check_no_self_loops)
      : trans_model_(trans_model),
        disambig_syms_(disambig_syms),
        check_no_self_loops_(check_no_self_loops) {}
  typedef int32_t Result;
  int32_t operator()(int32_t label) const {
    if (label == static_cast<int32_t>(fst::kNoLabel)) {
      return -1;  // -1 -> -1
    } else if (label >= 1 && label <= trans_model_.NumTransitionIds()) {
      if (check_no_self_loops_ && trans_model_.IsSelfLoop(label)) {
        KHG_ERR << "AddSelfLoops: graph already has self-loops.";
      }
      return trans_model_.TransitionIdToTransitionState(label);
    } else {  // 0 or (presumably) disambiguation symbol.  Map to zero
      int32_t big_number = fst::kNontermBigNumber;  // 1000000
      if (label != 0 && label < big_number)
        KHG_ASSERT(std::binary_search(disambig_syms_.begin(),
                                      disambig_syms_.end(),
                                      label));  // or invalid label
      return 0;
    }
  }

 private:
  const TransitionModel &trans_model_;
  const std::vector<int32_t> &disambig_syms_;  // sorted.
  bool check_no_self_loops_;
};

// This is the code that expands an FST from transition-states to
// transition-ids, in the case where reorder == true, i.e. the non-optional
// transition is before the self-loop.
static void AddSelfLoopsReorder(const TransitionModel &trans_model,
                                const std::vector<int32_t> &disambig_syms,
                                float self_loop_scale, bool check_no_self_loops,
                                fst::VectorFst<fst::StdArc> *fst) {
  using namespace fst;  // NOLINT
  typedef StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  TidToTstateMapper f(trans_model, disambig_syms, check_no_self_loops);
  // Duplicate states as necessary so that each state will require at most one
  // self-loop to be added to it.  Approximately this means that if a
  // state has multiple different symbols on arcs entering it, it will be
  // duplicated, with one copy per incoming symbol.
  MakePrecedingInputSymbolsSameClass(true, fst, f);

  int32_t kNoTransState = f(kNoLabel);
  KHG_ASSERT(kNoTransState == -1);

  // use the following to keep track of the transition-state for each state.
  std::vector<int32_t> state_in(fst->NumStates(), kNoTransState);

  // This first loop just works out the label into each state,
  // and converts the transitions in the graph from transition-states
  // to transition-ids.

  for (StateIterator<VectorFst<Arc>> siter(*fst); !siter.Done(); siter.Next()) {
    StateId s = siter.Value();
    for (MutableArcIterator<VectorFst<Arc>> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      int32_t trans_state = f(arc.ilabel);
      if (state_in[arc.nextstate] == kNoTransState) {
        state_in[arc.nextstate] = trans_state;
      } else {
        KHG_ASSERT(state_in[arc.nextstate] == trans_state);
        // or probably an error in MakePrecedingInputSymbolsSame.
      }
    }
  }

  KHG_ASSERT(state_in[fst->Start()] == kNoStateId ||
             state_in[fst->Start()] == 0);
  // or MakePrecedingInputSymbolsSame failed.

  // The next loop looks at each graph state, adds the self-loop [if needed] and
  // multiples all the out-transitions' probs (and final-prob) by the
  // forward-prob for that state (which is one minus self-loop-prob).  We do it
  // like this to maintain stochasticity (i.e. rather than multiplying the arcs
  // with the corresponding labels on them by this probability).

  for (StateId s = 0; s < static_cast<StateId>(state_in.size()); s++) {
    if (state_in[s] >
        0) {  // defined, and not eps or a disambiguation symbol or a
              // nonterminal-related sybol for grammar decoding...
      int32_t trans_state = static_cast<int32_t>(state_in[s]);
      // First multiply all probabilities by "forward" probability.
      float log_prob = trans_model.GetNonSelfLoopLogProb(trans_state);
      fst->SetFinal(s,
                    Times(fst->Final(s), Weight(-log_prob * self_loop_scale)));
      for (MutableArcIterator<MutableFst<Arc>> aiter(fst, s); !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        arc.weight = Times(arc.weight, Weight(-log_prob * self_loop_scale));
        aiter.SetValue(arc);
      }
      // Now add self-loop, if needed.
      int32_t trans_id = trans_model.SelfLoopOf(trans_state);
      if (trans_id != 0) {  // has self-loop.
        float log_prob = trans_model.GetTransitionLogProb(trans_id);
        fst->AddArc(s,
                    Arc(trans_id, 0, Weight(-log_prob * self_loop_scale), s));
      }
    }
  }
}

// this is the code that expands an FST from transition-states to
// transition-ids, in the case where reorder == false, i.e. non-optional
// transition is after the self-loop.
static void AddSelfLoopsNoReorder(const TransitionModel &trans_model,
                                  const std::vector<int32_t> &disambig_syms,
                                  float self_loop_scale,
                                  bool check_no_self_loops,
                                  fst::VectorFst<fst::StdArc> *fst) {
  using namespace fst;  // NOLINT
  typedef StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  // Duplicate states as necessary so that each state has at most one self-loop
  // on it.
  TidToTstateMapper f(trans_model, disambig_syms, check_no_self_loops);
  MakeFollowingInputSymbolsSameClass(true, fst, f);

  StateId num_states = fst->NumStates();
  for (StateId s = 0; s < num_states; s++) {
    int32_t my_trans_state = f(kNoLabel);
    KHG_ASSERT(my_trans_state == -1);
    for (MutableArcIterator<VectorFst<Arc>> aiter(fst, s); !aiter.Done();
         aiter.Next()) {
      Arc arc = aiter.Value();
      if (my_trans_state == -1)
        my_trans_state = f(arc.ilabel);
      else
        KHG_ASSERT(
            my_trans_state ==
            f(arc.ilabel));  // or MakeFollowingInputSymbolsSameClass failed.
      if (my_trans_state > 0) {  // transition-id; multiply weight...
        float log_prob = trans_model.GetNonSelfLoopLogProb(my_trans_state);
        arc.weight = Times(arc.weight, Weight(-log_prob * self_loop_scale));
        aiter.SetValue(arc);
      }
    }
    if (fst->Final(s) != Weight::Zero()) {
      KHG_ASSERT(my_trans_state == kNoLabel ||
                 my_trans_state ==
                     0);  // or MakeFollowingInputSymbolsSameClass failed.
    }
    if (my_trans_state != kNoLabel && my_trans_state != 0) {
      // a transition-state;  add self-loop, if it has one.
      int32_t trans_id = trans_model.SelfLoopOf(my_trans_state);
      if (trans_id != 0) {  // has self-loop.
        float log_prob = trans_model.GetTransitionLogProb(trans_id);
        fst->AddArc(s,
                    Arc(trans_id, 0, Weight(-log_prob * self_loop_scale), s));
      }
    }
  }
}

void AddSelfLoops(const TransitionModel &trans_model,
                  const std::vector<int32_t> &disambig_syms,
                  float self_loop_scale, bool reorder, bool check_no_self_loops,
                  fst::VectorFst<fst::StdArc> *fst) {
  KHG_ASSERT(fst->Start() != fst::kNoStateId);
  if (reorder) {
    AddSelfLoopsReorder(trans_model, disambig_syms, self_loop_scale,
                        check_no_self_loops, fst);
  } else {
    AddSelfLoopsNoReorder(trans_model, disambig_syms, self_loop_scale,
                          check_no_self_loops, fst);
  }
}

// Returns the scaled, but not negated, log-prob, with the given scaling
// factors.
static float GetScaledTransitionLogProb(const TransitionModel &trans_model,
                                        int32_t trans_id,
                                        float transition_scale,
                                        float self_loop_scale) {
  if (transition_scale == self_loop_scale) {
    return trans_model.GetTransitionLogProb(trans_id) * transition_scale;
  } else {
    if (trans_model.IsSelfLoop(trans_id)) {
      return self_loop_scale * trans_model.GetTransitionLogProb(trans_id);
    } else {
      int32_t trans_state = trans_model.TransitionIdToTransitionState(trans_id);
      return self_loop_scale * trans_model.GetNonSelfLoopLogProb(trans_state) +
             transition_scale *
                 trans_model.GetTransitionLogProbIgnoringSelfLoops(trans_id);
      // This could be simplified to
      // (self_loop_scale - transition_scale) *
      // trans_model.GetNonSelfLoopLogProb(trans_state)
      // + trans_model.GetTransitionLogProb(trans_id);
      // this simplifies if self_loop_scale == 0.0
    }
  }
}

void AddTransitionProbs(
    const TransitionModel &trans_model,
    const std::vector<int32_t> &disambig_syms,  // may be empty
    float transition_scale, float self_loop_scale,
    fst::VectorFst<fst::StdArc> *fst) {
  using namespace fst;  // NOLINT
  KHG_ASSERT(IsSortedAndUniq(disambig_syms));

  int num_tids = trans_model.NumTransitionIds();
  for (StateIterator<VectorFst<StdArc>> siter(*fst); !siter.Done();
       siter.Next()) {
    for (MutableArcIterator<VectorFst<StdArc>> aiter(fst, siter.Value());
         !aiter.Done(); aiter.Next()) {
      StdArc arc = aiter.Value();
      StdArc::Label l = arc.ilabel;
      if (l >= 1 && l <= num_tids) {  // a transition-id.
        float scaled_log_prob = GetScaledTransitionLogProb(
            trans_model, l, transition_scale, self_loop_scale);
        arc.weight = Times(arc.weight, TropicalWeight(-scaled_log_prob));
      } else if (l != 0) {
        if (!std::binary_search(disambig_syms.begin(), disambig_syms.end(),
                                arc.ilabel))
          KHG_ERR << "AddTransitionProbs: invalid symbol " << arc.ilabel
                  << " on graph input side.";
      }
      aiter.SetValue(arc);
    }
  }
}

}  // namespace khg
