// kaldi-hmm-gmm/csrc/transition-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

// this if is copied and modified from
// kaldi/src/hmm/transition-model.cc

#include "kaldi-hmm-gmm/csrc/transition-model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi_native_io/csrc/io-funcs.h"

namespace khg {

TransitionModel::TransitionModel(const ContextDependencyInterface &ctx_dep,
                                 const HmmTopology &hmm_topo)
    : topo_(hmm_topo) {
  // First thing is to get all possible tuples.
  ComputeTuples(ctx_dep);
  ComputeDerived();
  InitializeProbs();
  Check();
}

void TransitionModel::Write(std::ostream &os, bool binary) const {
  bool is_hmm = IsHmm();
  kaldiio::WriteToken(os, binary, "<TransitionModel>");
  if (!binary) os << "\n";

  topo_.Write(os, binary);

  if (is_hmm) {
    kaldiio::WriteToken(os, binary, "<Triples>");
  } else {
    kaldiio::WriteToken(os, binary, "<Tuples>");
  }

  kaldiio::WriteBasicType(os, binary, static_cast<int32_t>(tuples_.size()));

  if (!binary) os << "\n";

  for (int32_t i = 0; i < static_cast<int32_t>(tuples_.size()); i++) {
    kaldiio::WriteBasicType(os, binary, tuples_[i].phone);
    kaldiio::WriteBasicType(os, binary, tuples_[i].hmm_state);
    kaldiio::WriteBasicType(os, binary, tuples_[i].forward_pdf);

    if (!is_hmm) kaldiio::WriteBasicType(os, binary, tuples_[i].self_loop_pdf);

    if (!binary) os << "\n";
  }
  if (is_hmm) {
    kaldiio::WriteToken(os, binary, "</Triples>");
  } else {
    kaldiio::WriteToken(os, binary, "</Tuples>");
  }
  if (!binary) os << "\n";

  kaldiio::WriteToken(os, binary, "<LogProbs>");
  if (!binary) os << "\n";

  log_probs_.Write(os, binary);
  kaldiio::WriteToken(os, binary, "</LogProbs>");

  if (!binary) os << "\n";

  kaldiio::WriteToken(os, binary, "</TransitionModel>");

  if (!binary) os << "\n";
}

void TransitionModel::Read(std::istream &is, bool binary) {
  kaldiio::ExpectToken(is, binary, "<TransitionModel>");
  topo_.Read(is, binary);

  std::string token;
  kaldiio::ReadToken(is, binary, &token);

  int32_t size;
  kaldiio::ReadBasicType(is, binary, &size);
  tuples_.resize(size);

  for (int32_t i = 0; i < size; i++) {
    kaldiio::ReadBasicType(is, binary, &(tuples_[i].phone));
    kaldiio::ReadBasicType(is, binary, &(tuples_[i].hmm_state));
    kaldiio::ReadBasicType(is, binary, &(tuples_[i].forward_pdf));

    if (token == "<Tuples>") {
      kaldiio::ReadBasicType(is, binary, &(tuples_[i].self_loop_pdf));
    } else if (token == "<Triples>") {
      tuples_[i].self_loop_pdf = tuples_[i].forward_pdf;
    }
  }

  kaldiio::ReadToken(is, binary, &token);
  KHG_ASSERT(token == "</Triples>" || token == "</Tuples>");
  ComputeDerived();

  kaldiio::ExpectToken(is, binary, "<LogProbs>");
  log_probs_.Read(is, binary);
  kaldiio::ExpectToken(is, binary, "</LogProbs>");
  kaldiio::ExpectToken(is, binary, "</TransitionModel>");
  ComputeDerivedOfProbs();
  Check();
}

bool TransitionModel::IsHmm() const { return topo_.IsHmm(); }

void TransitionModel::ComputeTuples(const ContextDependencyInterface &ctx_dep) {
  if (IsHmm()) {
    ComputeTuplesIsHmm(ctx_dep);
  } else {
    ComputeTuplesNotHmm(ctx_dep);
  }

  // now tuples_ is populated with all possible tuples of (phone, hmm_state,
  // pdf, self_loop_pdf).
  std::sort(tuples_.begin(), tuples_.end());  // sort to enable reverse lookup.
  // this sorting defines the transition-ids.
}

void TransitionModel::ComputeTuplesIsHmm(
    const ContextDependencyInterface &ctx_dep) {
  const std::vector<int32_t> &phones = topo_.GetPhones();
  KHG_ASSERT(!phones.empty());

  // this is the case for normal models. but not for chain models
  std::vector<std::vector<std::pair<int32_t, int32_t>>> pdf_info;
  std::vector<int32_t> num_pdf_classes(
      1 + *std::max_element(phones.begin(), phones.end()), -1);
  for (size_t i = 0; i < phones.size(); i++)
    num_pdf_classes[phones[i]] = topo_.NumPdfClasses(phones[i]);
  ctx_dep.GetPdfInfo(phones, num_pdf_classes, &pdf_info);
  // pdf_info is list indexed by pdf of which (phone, pdf_class) it
  // can correspond to.

  std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> to_hmm_state_list;
  // to_hmm_state_list is a map from (phone, pdf_class) to the list
  // of hmm-states in the HMM for that phone that that (phone, pdf-class)
  // can correspond to.
  for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
    int32_t phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    for (int32_t j = 0; j < static_cast<int32_t>(entry.size());
         j++) {  // for each state...
      int32_t pdf_class = entry[j].forward_pdf_class;
      if (pdf_class != kNoPdf) {
        to_hmm_state_list[std::make_pair(phone, pdf_class)].push_back(j);
      }
    }
  }

  for (int32_t pdf = 0; pdf < static_cast<int32_t>(pdf_info.size()); pdf++) {
    for (size_t j = 0; j < pdf_info[pdf].size(); j++) {
      int32_t phone = pdf_info[pdf][j].first,
              pdf_class = pdf_info[pdf][j].second;
      const std::vector<int32_t> &state_vec =
          to_hmm_state_list[std::make_pair(phone, pdf_class)];
      KHG_ASSERT(!state_vec.empty());
      // state_vec is a list of the possible HMM-states that emit this
      // pdf_class.
      for (size_t k = 0; k < state_vec.size(); k++) {
        int32_t hmm_state = state_vec[k];
        tuples_.push_back(Tuple(phone, hmm_state, pdf, pdf));
      }
    }
  }
}

void TransitionModel::ComputeTuplesNotHmm(
    const ContextDependencyInterface &ctx_dep) {
  const std::vector<int32_t> &phones = topo_.GetPhones();
  KHG_ASSERT(!phones.empty());

  // pdf_info is a set of lists indexed by phone. Each list is indexed by
  // (pdf-class, self-loop pdf-class) of each state of that phone, and the
  // element is a list of possible (pdf, self-loop pdf) pairs that (pdf-class,
  // self-loop pdf-class) pair generates.
  std::vector<std::vector<std::vector<std::pair<int32_t, int32_t>>>> pdf_info;
  // pdf_class_pairs is a set of lists indexed by phone. Each list stores
  // (pdf-class, self-loop pdf-class) of each state of that phone.
  std::vector<std::vector<std::pair<int32_t, int32_t>>> pdf_class_pairs;
  pdf_class_pairs.resize(1 + *std::max_element(phones.begin(), phones.end()));
  for (size_t i = 0; i < phones.size(); i++) {
    int32_t phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    for (int32_t j = 0; j < static_cast<int32_t>(entry.size());
         j++) {  // for each state...
      int32_t forward_pdf_class = entry[j].forward_pdf_class,
              self_loop_pdf_class = entry[j].self_loop_pdf_class;
      if (forward_pdf_class != kNoPdf)
        pdf_class_pairs[phone].push_back(
            std::make_pair(forward_pdf_class, self_loop_pdf_class));
    }
  }
  ctx_dep.GetPdfInfo(phones, pdf_class_pairs, &pdf_info);

  std::vector<std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>>>
      to_hmm_state_list;
  to_hmm_state_list.resize(1 + *std::max_element(phones.begin(), phones.end()));
  // to_hmm_state_list is a phone-indexed set of maps from (pdf-class, self-loop
  // pdf_class) to the list of hmm-states in the HMM for that phone that that
  // (pdf-class, self-loop pdf-class) can correspond to.
  for (size_t i = 0; i < phones.size(); i++) {  // setting up to_hmm_state_list.
    int32_t phone = phones[i];
    const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
    std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>>
        phone_to_hmm_state_list;
    for (int32_t j = 0; j < static_cast<int32_t>(entry.size());
         j++) {  // for each state...
      int32_t forward_pdf_class = entry[j].forward_pdf_class,
              self_loop_pdf_class = entry[j].self_loop_pdf_class;
      if (forward_pdf_class != kNoPdf) {
        phone_to_hmm_state_list[std::make_pair(forward_pdf_class,
                                               self_loop_pdf_class)]
            .push_back(j);
      }
    }
    to_hmm_state_list[phone] = phone_to_hmm_state_list;
  }

  for (int32_t i = 0; i < phones.size(); i++) {
    int32_t phone = phones[i];
    for (int32_t j = 0; j < static_cast<int32_t>(pdf_info[phone].size()); j++) {
      int32_t pdf_class = pdf_class_pairs[phone][j].first,
              self_loop_pdf_class = pdf_class_pairs[phone][j].second;
      const std::vector<int32_t> &state_vec =
          to_hmm_state_list[phone]
                           [std::make_pair(pdf_class, self_loop_pdf_class)];
      KHG_ASSERT(!state_vec.empty());
      for (size_t k = 0; k < state_vec.size(); k++) {
        int32_t hmm_state = state_vec[k];
        for (size_t m = 0; m < pdf_info[phone][j].size(); m++) {
          int32_t pdf = pdf_info[phone][j][m].first,
                  self_loop_pdf = pdf_info[phone][j][m].second;
          tuples_.push_back(Tuple(phone, hmm_state, pdf, self_loop_pdf));
        }
      }
    }
  }
}

void TransitionModel::ComputeDerived() {
  state2id_.resize(tuples_.size() + 2);  // indexed by transition-state, which
  // is one based, but also an entry for one past end of list.

  int32_t cur_transition_id = 1;
  num_pdfs_ = 0;
  for (int32_t tstate = 1;
       tstate <= static_cast<int32_t>(tuples_.size() + 1);  // not a typo.
       ++tstate) {
    state2id_[tstate] = cur_transition_id;
    if (static_cast<size_t>(tstate) <= tuples_.size()) {
      int32_t phone = tuples_[tstate - 1].phone,
              hmm_state = tuples_[tstate - 1].hmm_state,
              forward_pdf = tuples_[tstate - 1].forward_pdf,
              self_loop_pdf = tuples_[tstate - 1].self_loop_pdf;
      num_pdfs_ = std::max(num_pdfs_, 1 + forward_pdf);
      num_pdfs_ = std::max(num_pdfs_, 1 + self_loop_pdf);
      const HmmTopology::HmmState &state =
          topo_.TopologyForPhone(phone)[hmm_state];
      int32_t my_num_ids = static_cast<int32_t>(state.transitions.size());
      cur_transition_id += my_num_ids;  // # trans out of this state.
    }
  }

  id2state_.resize(
      cur_transition_id);  // cur_transition_id is #transition-ids+1.
  id2pdf_id_.resize(cur_transition_id);

  for (int32_t tstate = 1; tstate <= static_cast<int32_t>(tuples_.size());
       tstate++) {
    for (int32_t tid = state2id_[tstate]; tid < state2id_[tstate + 1]; tid++) {
      id2state_[tid] = tstate;
      if (IsSelfLoop(tid)) {
        id2pdf_id_[tid] = tuples_[tstate - 1].self_loop_pdf;
      } else {
        id2pdf_id_[tid] = tuples_[tstate - 1].forward_pdf;
      }
    }
  }

  // The following statements put copies a large number in the region of memory
  // past the end of the id2pdf_id_ array, while leaving the array as it was
  // before.  The goal of this is to speed up decoding by disabling a check
  // inside TransitionIdToPdf() that the transition-id was within the correct
  // range.
  int32_t num_big_numbers = std::min<int32_t>(2000, cur_transition_id);
  id2pdf_id_.resize(cur_transition_id + num_big_numbers,
                    std::numeric_limits<int32_t>::max());
  id2pdf_id_.resize(cur_transition_id);
}

bool TransitionModel::IsSelfLoop(int32_t trans_id) const {
  KHG_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32_t trans_state = id2state_[trans_id];
  int32_t trans_index = trans_id - state2id_[trans_state];
  const Tuple &tuple = tuples_[trans_state - 1];
  int32_t phone = tuple.phone, hmm_state = tuple.hmm_state;
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
  KHG_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  return (static_cast<size_t>(trans_index) <
              entry[hmm_state].transitions.size() &&
          entry[hmm_state].transitions[trans_index].first == hmm_state);
}

void TransitionModel::InitializeProbs() {
  log_probs_.Resize(NumTransitionIds() +
                    1);  // one-based array, zeroth element empty.
  for (int32_t trans_id = 1; trans_id <= NumTransitionIds(); trans_id++) {
    int32_t trans_state = id2state_[trans_id];
    int32_t trans_index = trans_id - state2id_[trans_state];
    const Tuple &tuple = tuples_[trans_state - 1];
    const HmmTopology::TopologyEntry &entry =
        topo_.TopologyForPhone(tuple.phone);
    KHG_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
    float prob = entry[tuple.hmm_state].transitions[trans_index].second;
    if (prob <= 0.0)
      KHG_ERR << "TransitionModel::InitializeProbs, zero "
                 "probability [should remove that entry in the topology]";
    if (prob > 1.0)
      KHG_WARN << "TransitionModel::InitializeProbs, prob greater than one.";
    log_probs_(trans_id) = std::log(prob);
  }
  ComputeDerivedOfProbs();
}

void TransitionModel::ComputeDerivedOfProbs() {
  non_self_loop_log_probs_.Resize(NumTransitionStates() +
                                  1);  // this array indexed
  //  by transition-state with nothing in zeroth element.
  for (int32_t tstate = 1; tstate <= NumTransitionStates(); tstate++) {
    int32_t tid = SelfLoopOf(tstate);
    if (tid == 0) {                            // no self-loop
      non_self_loop_log_probs_(tstate) = 0.0;  // log(1.0)
    } else {
      float self_loop_prob = std::exp(GetTransitionLogProb(tid)),
            non_self_loop_prob = 1.0 - self_loop_prob;
      if (non_self_loop_prob <= 0.0) {
        KHG_WARN << "ComputeDerivedOfProbs(): non-self-loop prob is "
                 << non_self_loop_prob;
        non_self_loop_prob = 1.0e-10;  // just so we can continue...
      }
      non_self_loop_log_probs_(tstate) =
          std::log(non_self_loop_prob);  // will be negative.
    }
  }
}

float TransitionModel::GetTransitionProb(int32_t trans_id) const {
  return std::exp(log_probs_(trans_id));
}

const std::vector<int32_t> &TransitionModel::TransitionIdToPdfArray() const {
  return id2pdf_id_;
}
// returns the self-loop transition-id,
int32_t TransitionModel::SelfLoopOf(int32_t trans_state) const {
  KHG_ASSERT(static_cast<size_t>(trans_state - 1) < tuples_.size());
  const Tuple &tuple = tuples_[trans_state - 1];
  // or zero if does not exist.
  int32_t phone = tuple.phone, hmm_state = tuple.hmm_state;
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(phone);
  KHG_ASSERT(static_cast<size_t>(hmm_state) < entry.size());
  for (int32_t trans_index = 0;
       trans_index < static_cast<int32_t>(entry[hmm_state].transitions.size());
       trans_index++)
    if (entry[hmm_state].transitions[trans_index].first == hmm_state)
      return PairToTransitionId(trans_state, trans_index);

  return 0;  // invalid transition id.
}

int32_t TransitionModel::PairToTransitionId(int32_t trans_state,
                                            int32_t trans_index) const {
  KHG_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  KHG_ASSERT(trans_index < state2id_[trans_state + 1] - state2id_[trans_state]);
  return state2id_[trans_state] + trans_index;
}

float TransitionModel::GetTransitionLogProb(int32_t trans_id) const {
  return log_probs_(trans_id);
}

void TransitionModel::Check() const {
  KHG_ASSERT(NumTransitionIds() != 0 && NumTransitionStates() != 0);
  {
    int32_t sum = 0;
    for (int32_t ts = 1; ts <= NumTransitionStates(); ts++)
      sum += NumTransitionIndices(ts);
    KHG_ASSERT(sum == NumTransitionIds());
  }
  for (int32_t tid = 1; tid <= NumTransitionIds(); tid++) {
    int32_t tstate = TransitionIdToTransitionState(tid),
            index = TransitionIdToTransitionIndex(tid);
    KHG_ASSERT(tstate > 0 && tstate <= NumTransitionStates() && index >= 0);
    KHG_ASSERT(tid == PairToTransitionId(tstate, index));
    int32_t phone = TransitionStateToPhone(tstate),
            hmm_state = TransitionStateToHmmState(tstate),
            forward_pdf = TransitionStateToForwardPdf(tstate),
            self_loop_pdf = TransitionStateToSelfLoopPdf(tstate);
    KHG_ASSERT(tstate == TupleToTransitionState(phone, hmm_state, forward_pdf,
                                                self_loop_pdf));
    KHG_ASSERT(log_probs_(tid) <= 0.0 &&
               log_probs_(tid) - log_probs_(tid) == 0.0);
    // checking finite and non-positive (and not out-of-bounds).
  }
}

int32_t TransitionModel::TransitionIdToTransitionState(int32_t trans_id) const {
  KHG_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  return id2state_[trans_id];
}

int32_t TransitionModel::NumTransitionIndices(int32_t trans_state) const {
  KHG_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return static_cast<int32_t>(state2id_[trans_state + 1] -
                              state2id_[trans_state]);
}

int32_t TransitionModel::TupleToTransitionState(int32_t phone,
                                                int32_t hmm_state, int32_t pdf,
                                                int32_t self_loop_pdf) const {
  Tuple tuple(phone, hmm_state, pdf, self_loop_pdf);
  // Note: if this ever gets too expensive, which is unlikely, we can refactor
  // this code to sort first on pdf, and then index on pdf, so those
  // that have the same pdf are in a contiguous range.
  std::vector<Tuple>::const_iterator iter =
      std::lower_bound(tuples_.begin(), tuples_.end(), tuple);
  if (iter == tuples_.end() || !(*iter == tuple)) {
    KHG_ERR << "TransitionModel::TupleToTransitionState, tuple not found."
            << " (incompatible tree and model?)";
  }
  // tuples_ is indexed by transition_state-1, so add one.
  return static_cast<int32_t>((iter - tuples_.begin())) + 1;
}

int32_t TransitionModel::TransitionStateToSelfLoopPdf(
    int32_t trans_state) const {
  KHG_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state - 1].self_loop_pdf;
}

int32_t TransitionModel::TransitionStateToForwardPdf(
    int32_t trans_state) const {
  KHG_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state - 1].forward_pdf;
}

int32_t TransitionModel::TransitionStateToHmmState(int32_t trans_state) const {
  KHG_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state - 1].hmm_state;
}

int32_t TransitionModel::TransitionStateToPhone(int32_t trans_state) const {
  KHG_ASSERT(static_cast<size_t>(trans_state) <= tuples_.size());
  return tuples_[trans_state - 1].phone;
}

int32_t TransitionModel::TransitionIdToTransitionIndex(int32_t trans_id) const {
  KHG_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  return trans_id - state2id_[id2state_[trans_id]];
}

bool TransitionModel::TransitionIdsEquivalent(int32_t trans_id1,
                                              int32_t trans_id2) const {
  return TransitionIdToTransitionState(trans_id1) ==
         TransitionIdToTransitionState(trans_id2);
}

bool TransitionModel::TransitionIdIsStartOfPhone(int32_t trans_id) const {
  return TransitionIdToHmmState(trans_id) == 0;
}

int32_t TransitionModel::TransitionIdToHmmState(int32_t trans_id) const {
  KHG_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  int32_t trans_state = id2state_[trans_id];
  const Tuple &t = tuples_[trans_state - 1];
  return t.hmm_state;
}

int32_t TransitionModel::TransitionIdToPhone(int32_t trans_id) const {
  KHG_ASSERT(trans_id != 0 && static_cast<size_t>(trans_id) < id2state_.size());
  int32_t trans_state = id2state_[trans_id];
  return tuples_[trans_state - 1].phone;
}

bool TransitionModel::IsFinal(int32_t trans_id) const {
  KHG_ASSERT(static_cast<size_t>(trans_id) < id2state_.size());
  int32_t trans_state = id2state_[trans_id];
  int32_t trans_index = trans_id - state2id_[trans_state];
  const Tuple &tuple = tuples_[trans_state - 1];
  const HmmTopology::TopologyEntry &entry = topo_.TopologyForPhone(tuple.phone);
  KHG_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
  KHG_ASSERT(static_cast<size_t>(tuple.hmm_state) < entry.size());
  KHG_ASSERT(static_cast<size_t>(trans_index) <
             entry[tuple.hmm_state].transitions.size());
  // return true if the transition goes to the final state of the
  // topology entry.
  return (entry[tuple.hmm_state].transitions[trans_index].first + 1 ==
          static_cast<int32_t>(entry.size()));
}

float TransitionModel::GetNonSelfLoopLogProb(int32_t trans_state) const {
  KHG_ASSERT(trans_state != 0);
  return non_self_loop_log_probs_(trans_state);
}

float TransitionModel::GetTransitionLogProbIgnoringSelfLoops(
    int32_t trans_id) const {
  KHG_ASSERT(trans_id != 0);
  KHG_ASSERT(!IsSelfLoop(trans_id));
  return log_probs_(trans_id) -
         GetNonSelfLoopLogProb(TransitionIdToTransitionState(trans_id));
}

/// This version of the Update() function is for if the user specifies
/// --share-for-pdfs=true.  We share the transitions for all states that
/// share the same pdf.
void TransitionModel::MleUpdateShared(const DoubleVector &stats,
                                      const MleTransitionUpdateConfig &cfg,
                                      float *objf_impr_out, float *count_out) {
  KHG_ASSERT(cfg.share_for_pdfs);

  float count_sum = 0.0, objf_impr_sum = 0.0;
  int32_t num_skipped = 0, num_floored = 0;
  KHG_ASSERT(stats.size() == NumTransitionIds() + 1);
  std::map<int32_t, std::set<int32_t>> pdf_to_tstate;

  for (int32_t tstate = 1; tstate <= NumTransitionStates(); ++tstate) {
    int32_t pdf = TransitionStateToForwardPdf(tstate);
    pdf_to_tstate[pdf].insert(tstate);
    if (!IsHmm()) {
      pdf = TransitionStateToSelfLoopPdf(tstate);
      pdf_to_tstate[pdf].insert(tstate);
    }
  }

  for (auto map_iter = pdf_to_tstate.begin(); map_iter != pdf_to_tstate.end();
       ++map_iter) {
    // map_iter->first is pdf-id... not needed.
    const auto &tstates = map_iter->second;

    KHG_ASSERT(!tstates.empty());

    int32_t one_tstate = *(tstates.begin());
    int32_t n = NumTransitionIndices(one_tstate);
    KHG_ASSERT(n >= 1);

    if (n > 1) {  // Only update if >1 transition...
      std::vector<double> counts(n, 0);
      double pdf_tot = 0;

      for (auto iter = tstates.begin(); iter != tstates.end(); ++iter) {
        int32_t tstate = *iter;

        if (NumTransitionIndices(tstate) != n) {
          KHG_ERR << "Mismatch in #transition indices: you cannot "
                     "use the --share-for-pdfs option with this topology "
                     "and sharing scheme.";
        }

        for (int32_t tidx = 0; tidx < n; ++tidx) {
          int32_t tid = PairToTransitionId(tstate, tidx);
          auto acc = stats[tid];
          counts[tidx] += acc;
          pdf_tot += acc;
        }
      }

      count_sum += pdf_tot;

      if (pdf_tot < cfg.mincount) {
        ++num_skipped;
      } else {
        // Note: when calculating objf improvement, we
        // assume we previously had the same tying scheme so
        // we can get the params from one_tstate and they're valid
        // for all.
        std::vector<float> old_probs(n, 0);

        FloatVector new_probs(n);

        for (int32_t tidx = 0; tidx < n; ++tidx) {
          int32_t tid = PairToTransitionId(one_tstate, tidx);
          old_probs[tidx] = GetTransitionProb(tid);
        }

        for (int32_t tidx = 0; tidx < n; ++tidx) {
          new_probs[tidx] = counts[tidx] / pdf_tot;
        }

        for (int32_t i = 0; i < 3; i++) {
          // keep flooring+renormalizing for 3 times..
          new_probs /= new_probs.sum();

          for (int32_t tidx = 0; tidx < n; ++tidx) {
            new_probs[tidx] = std::max(new_probs[tidx], cfg.floor);
          }
        }

        // Compute objf change
        for (int32_t tidx = 0; tidx < n; ++tidx) {
          if (new_probs[tidx] == cfg.floor) {
            num_floored++;
          }

          double objf_change = counts[tidx] * (std::log(new_probs[tidx]) -
                                               std::log(old_probs[tidx]));

          objf_impr_sum += objf_change;
        }

        // Commit updated values.
        for (auto iter = tstates.begin(); iter != tstates.end(); ++iter) {
          int32_t tstate = *iter;
          for (int32_t tidx = 0; tidx < n; ++tidx) {
            int32_t tid = PairToTransitionId(tstate, tidx);
            log_probs_(tid) = std::log(new_probs[tidx]);
            if (log_probs_(tid) - log_probs_(tid) != 0.0)
              KHG_ERR
                  << "Log probs is inf or NaN: error in update or bad stats?";
          }
        }
      }
    }
  }

  KHG_LOG << "Objf change is " << (objf_impr_sum / count_sum)
          << " per frame over " << count_sum << " frames; " << num_floored
          << " probabilities floored, " << num_skipped
          << " pdf-ids skipped due to insufficient data.";

  if (objf_impr_out) {
    *objf_impr_out = objf_impr_sum;
  }

  if (count_out) {
    *count_out = count_sum;
  }

  ComputeDerivedOfProbs();
}

// stats are counts/weights, indexed by transition-id.
void TransitionModel::MleUpdate(const DoubleVector &stats,
                                const MleTransitionUpdateConfig &cfg,
                                float *objf_impr_out, float *count_out) {
  if (cfg.share_for_pdfs) {
    MleUpdateShared(stats, cfg, objf_impr_out, count_out);
    return;
  }

  float count_sum = 0.0, objf_impr_sum = 0.0;
  int32_t num_skipped = 0, num_floored = 0;

  KHG_ASSERT(stats.size() == NumTransitionIds() + 1);

  for (int32_t tstate = 1; tstate <= NumTransitionStates(); ++tstate) {
    int32_t n = NumTransitionIndices(tstate);
    KHG_ASSERT(n >= 1);
    if (n > 1) {  // no point updating if only one transition...
      std::vector<double> counts(n, 0);
      double tstate_tot = 0;

      for (int32_t tidx = 0; tidx < n; ++tidx) {
        int32_t tid = PairToTransitionId(tstate, tidx);
        auto acc = stats[tid];
        counts[tidx] = acc;
        tstate_tot += acc;
      }

      count_sum += tstate_tot;
      if (tstate_tot < cfg.mincount) {
        num_skipped++;
      } else {
        std::vector<float> old_probs(n, 0);

        FloatVector new_probs(n);

        for (int32_t tidx = 0; tidx < n; ++tidx) {
          int32_t tid = PairToTransitionId(tstate, tidx);
          old_probs[tidx] = GetTransitionProb(tid);
        }

        for (int32_t tidx = 0; tidx < n; tidx++) {
          new_probs[tidx] = counts[tidx] / tstate_tot;
        }

        for (int32_t i = 0; i < 3; i++) {
          // keep flooring+renormalizing for 3 times..
          new_probs /= new_probs.sum();

          for (int32_t tidx = 0; tidx < n; tidx++)
            new_probs[tidx] = std::max(new_probs[tidx], cfg.floor);
        }

        // Compute objf change
        for (int32_t tidx = 0; tidx < n; ++tidx) {
          if (new_probs[tidx] == cfg.floor) {
            ++num_floored;
          }

          double objf_change = counts[tidx] * (std::log(new_probs[tidx]) -
                                               std::log(old_probs[tidx]));
          objf_impr_sum += objf_change;
        }

        // Commit updated values.
        for (int32_t tidx = 0; tidx < n; ++tidx) {
          int32_t tid = PairToTransitionId(tstate, tidx);
          log_probs_(tid) = std::log(new_probs[tidx]);

          if (log_probs_(tid) - log_probs_(tid) != 0.0)
            KHG_ERR << "Log probs is inf or NaN: error in update or bad stats?";
        }
      }
    }
  }

  KHG_LOG << "TransitionModel::Update, objf change is "
          << (objf_impr_sum / count_sum) << " per frame over " << count_sum
          << " frames. ";
  KHG_LOG
      << num_floored << " probabilities floored, " << num_skipped << " out of "
      << NumTransitionStates()
      << " transition-states "
         "skipped due to insuffient data (it is normal to have some skipped.)";

  if (objf_impr_out) {
    *objf_impr_out = objf_impr_sum;
  }

  if (count_out) {
    *count_out = count_sum;
  }

  ComputeDerivedOfProbs();
}

bool GetPdfsForPhones(const TransitionModel &trans_model,
                      const std::vector<int32_t> &phones,
                      std::vector<int32_t> *pdfs) {
  KHG_ASSERT(IsSortedAndUniq(phones));
  KHG_ASSERT(pdfs != nullptr);

  pdfs->clear();

  for (int32_t tstate = 1; tstate <= trans_model.NumTransitionStates();
       ++tstate) {
    if (std::binary_search(phones.begin(), phones.end(),
                           trans_model.TransitionStateToPhone(tstate))) {
      pdfs->push_back(trans_model.TransitionStateToForwardPdf(tstate));
      pdfs->push_back(trans_model.TransitionStateToSelfLoopPdf(tstate));
    }
  }

  SortAndUniq(pdfs);

  for (int32_t tstate = 1; tstate <= trans_model.NumTransitionStates();
       ++tstate) {
    if ((std::binary_search(pdfs->begin(), pdfs->end(),
                            trans_model.TransitionStateToForwardPdf(tstate)) ||
         std::binary_search(
             pdfs->begin(), pdfs->end(),
             trans_model.TransitionStateToSelfLoopPdf(tstate))) &&
        !std::binary_search(phones.begin(), phones.end(),
                            trans_model.TransitionStateToPhone(tstate))) {
      return false;
    }
  }

  return true;
}

}  // namespace khg
