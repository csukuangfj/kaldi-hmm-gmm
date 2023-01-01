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
  // ComputeDerived();
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

}  // namespace khg
