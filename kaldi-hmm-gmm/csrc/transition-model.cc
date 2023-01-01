// kaldi-hmm-gmm/csrc/transition-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

// this if is copied and modified from
// kaldi/src/hmm/transition-model.cc

#include "kaldi-hmm-gmm/csrc/transition-model.h"

#include <map>

#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

TransitionModel::TransitionModel(const ContextDependencyInterface &ctx_dep,
                                 const HmmTopology &hmm_topo)
    : topo_(hmm_topo) {
  // First thing is to get all possible tuples.
  ComputeTuples(ctx_dep);
  ComputeDerived();
  // InitializeProbs();
  // Check();
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
       tstate++) {
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

}  // namespace khg
