// kaldi-hmm-gmm/csrc/context-dep.cc
//
// Copyright (c)  2022  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/tree/context-dep.cc

#include "kaldi-hmm-gmm/csrc/context-dep.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/build-tree-utils.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi_native_io/csrc/io-funcs.h"

namespace khg {

bool ContextDependency::Compute(const std::vector<int32_t> &phoneseq,
                                int32_t pdf_class, int32_t *pdf_id) const {
  KHG_ASSERT(static_cast<int32_t>(phoneseq.size()) == N_);

  EventType event_vec;
  event_vec.reserve(N_ + 1);
  event_vec.push_back(
      std::make_pair(static_cast<EventKeyType>(kPdfClass),  // -1
                     static_cast<EventValueType>(pdf_class)));

  static_assert(kPdfClass < 0, "");  // or it would not be sorted.

  for (int32_t i = 0; i < N_; ++i) {
    event_vec.push_back(
        std::make_pair(static_cast<EventKeyType>(i),
                       static_cast<EventValueType>(phoneseq[i])));
    KHG_ASSERT(static_cast<EventAnswerType>(phoneseq[i]) >= 0);
  }

  KHG_ASSERT(pdf_id != nullptr);
  return to_pdf_->Map(event_vec, pdf_id);
}

void ContextDependency::Write(std::ostream &os, bool binary) const {
  kaldiio::WriteToken(os, binary, "ContextDependency");
  kaldiio::WriteBasicType(os, binary, N_);
  kaldiio::WriteBasicType(os, binary, P_);
  kaldiio::WriteToken(os, binary, "ToPdf");
  to_pdf_->Write(os, binary);
  kaldiio::WriteToken(os, binary, "EndContextDependency");
}

void ContextDependency::Read(std::istream &is, bool binary) {
  if (to_pdf_) {
    delete to_pdf_;
    to_pdf_ = nullptr;
  }

  kaldiio::ExpectToken(is, binary, "ContextDependency");
  kaldiio::ReadBasicType(is, binary, &N_);
  kaldiio::ReadBasicType(is, binary, &P_);

  EventMap *to_pdf = nullptr;
  std::string token;
  kaldiio::ReadToken(is, binary, &token);

  if (token == "ToLength") {  // back-compat.
    EventMap *to_num_pdf_classes = EventMap::Read(is, binary);
    delete to_num_pdf_classes;
    kaldiio::ReadToken(is, binary, &token);
  }

  if (token == "ToPdf") {
    to_pdf = EventMap::Read(is, binary);
  } else {
    KHG_ERR << "Got unexpected token " << token
            << " reading context-dependency object.";
  }

  kaldiio::ExpectToken(is, binary, "EndContextDependency");
  to_pdf_ = to_pdf;
}

void ContextDependency::EnumeratePairs(
    const std::vector<int32_t> &phones, int32_t self_loop_pdf_class,
    int32_t forward_pdf_class, const std::vector<int32_t> &phone_window,
    std::unordered_set<std::pair<int32_t, int32_t>, PairHasher<int32_t>> *pairs)
    const {
  std::vector<int32_t> new_phone_window(phone_window);
  EventType vec;

  std::vector<EventAnswerType> forward_pdfs, self_loop_pdfs;

  // get list of possible forward pdfs
  vec.clear();
  for (size_t i = 0; i < N_; i++)
    if (phone_window[i] >= 0)
      vec.push_back(
          std::make_pair(static_cast<EventKeyType>(i),
                         static_cast<EventValueType>(phone_window[i])));
  vec.push_back(std::make_pair(kPdfClass,
                               static_cast<EventValueType>(forward_pdf_class)));
  std::sort(vec.begin(), vec.end());
  to_pdf_->MultiMap(vec, &forward_pdfs);
  SortAndUniq(&forward_pdfs);

  if (self_loop_pdf_class < 0) {
    // Invalid pdf-class because there was no self-loop.  Return pairs
    // where the self-loop pdf-id is -1.
    for (int32_t forward_pdf : forward_pdfs) {
      pairs->insert(std::pair<int32_t, int32_t>(forward_pdf, -1));
    }
    return;
  }

  // get list of possible self-loop pdfs
  vec.clear();
  for (size_t i = 0; i < N_; i++)
    if (phone_window[i] >= 0)
      vec.push_back(
          std::make_pair(static_cast<EventKeyType>(i),
                         static_cast<EventValueType>(phone_window[i])));
  vec.push_back(std::make_pair(
      kPdfClass, static_cast<EventValueType>(self_loop_pdf_class)));
  std::sort(vec.begin(), vec.end());
  to_pdf_->MultiMap(vec, &self_loop_pdfs);
  SortAndUniq(&self_loop_pdfs);

  if (forward_pdfs.size() == 1 || self_loop_pdfs.size() == 1) {
    for (size_t m = 0; m < forward_pdfs.size(); m++)
      for (size_t n = 0; n < self_loop_pdfs.size(); n++)
        pairs->insert(std::make_pair(forward_pdfs[m], self_loop_pdfs[n]));
  } else {
    // Choose 'position' as a phone position in 'context' that's currently
    // -1, and that is as close as possible to the central position P.
    int32_t position = 0;
    int32_t min_dist = N_ - 1;
    for (int32_t i = 0; i < N_; i++) {
      int32_t dist = (P_ - i > 0) ? (P_ - i) : (i - P_);
      if (phone_window[i] == -1 && dist < min_dist) {
        position = i;
        min_dist = dist;
      }
    }
    KHG_ASSERT(min_dist < N_);
    KHG_ASSERT(position != P_);

    // The next two lines have to do with how BOS/EOS effects are handled in
    // phone context.  Zero phone value in a non-central position (i.e. not
    // position P_...  and 'position' will never equal P_) means 'there is no
    // phone here because we're at BOS or EOS'.
    new_phone_window[position] = 0;
    EnumeratePairs(phones, self_loop_pdf_class, forward_pdf_class,
                   new_phone_window, pairs);

    for (size_t i = 0; i < phones.size(); i++) {
      new_phone_window[position] = phones[i];
      EnumeratePairs(phones, self_loop_pdf_class, forward_pdf_class,
                     new_phone_window, pairs);
    }
  }
}

void ContextDependency::GetPdfInfo(
    const std::vector<int32_t> &phones,
    const std::vector<int32_t> &num_pdf_classes,  // indexed by phone,
    std::vector<std::vector<std::pair<int32_t, int32_t>>> *pdf_info) const {
  EventType vec;
  KHG_ASSERT(pdf_info != nullptr);
  pdf_info->resize(NumPdfs());

  for (size_t i = 0; i < phones.size(); ++i) {
    int32_t phone = phones[i];
    vec.clear();
    vec.push_back(std::make_pair(static_cast<EventKeyType>(P_),
                                 static_cast<EventValueType>(phone)));
    // Now get length.
    KHG_ASSERT(static_cast<size_t>(phone) < num_pdf_classes.size());
    EventAnswerType len = num_pdf_classes[phone];

    for (int32_t pos = 0; pos < len; ++pos) {
      vec.resize(2);
      vec[0] = std::make_pair(static_cast<EventKeyType>(P_),
                              static_cast<EventValueType>(phone));
      vec[1] = std::make_pair(kPdfClass, static_cast<EventValueType>(pos));
      std::sort(vec.begin(), vec.end());
      std::vector<EventAnswerType>
          pdfs;  // pdfs that can be at this pos as this phone.
      to_pdf_->MultiMap(vec, &pdfs);
      SortAndUniq(&pdfs);
      if (pdfs.empty()) {
        KHG_WARN
            << "ContextDependency::GetPdfInfo, no pdfs returned for position "
            << pos << " of phone " << phone
            << ".   Continuing but this is a serious error.";
      }
      for (size_t j = 0; j < pdfs.size(); j++) {
        KHG_ASSERT(static_cast<size_t>(pdfs[j]) < pdf_info->size());
        (*pdf_info)[pdfs[j]].push_back(std::make_pair(phone, pos));
      }
    }
  }
  for (size_t i = 0; i < pdf_info->size(); ++i) {
    std::sort(((*pdf_info)[i]).begin(), ((*pdf_info)[i]).end());
    KHG_ASSERT(IsSortedAndUniq(((*pdf_info)[i])));  // should have no dups.
  }
}

void ContextDependency::GetPdfInfo(
    const std::vector<int32_t> &phones,
    const std::vector<std::vector<std::pair<int32_t, int32_t>>>
        &pdf_class_pairs,
    std::vector<std::vector<std::vector<std::pair<int32_t, int32_t>>>>
        *pdf_info) const {
  KHG_ASSERT(pdf_info != nullptr);
  pdf_info->resize(1 + *std::max_element(phones.begin(), phones.end()));
  std::vector<int32_t> phone_window(N_, -1);
  EventType vec;
  for (size_t i = 0; i < phones.size(); i++) {
    // loop over phones
    int32_t phone = phones[i];
    (*pdf_info)[phone].resize(pdf_class_pairs[phone].size());
    for (size_t j = 0; j < pdf_class_pairs[phone].size(); j++) {
      // loop over pdf_class pairs
      int32_t pdf_class = pdf_class_pairs[phone][j].first,
              self_loop_pdf_class = pdf_class_pairs[phone][j].second;
      phone_window[P_] = phone;

      std::unordered_set<std::pair<int32_t, int32_t>, PairHasher<int32_t>>
          pairs;
      EnumeratePairs(phones, self_loop_pdf_class, pdf_class, phone_window,
                     &pairs);
      auto iter = pairs.begin(), end = pairs.end();
      for (; iter != end; ++iter) (*pdf_info)[phone][j].push_back(*iter);
      std::sort(((*pdf_info)[phone][j]).begin(), ((*pdf_info)[phone][j]).end());
    }
  }
}

ContextDependency *MonophoneContextDependency(
    const std::vector<int32_t> &phones,
    const std::vector<int32_t> &phone2num_pdf_classes) {
  std::vector<std::vector<int32_t>> phone_sets(phones.size());

  for (size_t i = 0; i < phones.size(); i++) phone_sets[i].push_back(phones[i]);

  std::vector<bool> share_roots(phones.size(), false);  // don't share roots.

  // N is context size, P = position of central phone (must be 0).
  int32_t num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = GetStubMap(P, phone_sets, phone2num_pdf_classes,
                                 share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}

ContextDependency *MonophoneContextDependencyShared(
    const std::vector<std::vector<int32_t>> &phone_sets,
    const std::vector<int32_t> &phone2num_pdf_classes) {
  // don't share roots.
  std::vector<bool> share_roots(phone_sets.size(), false);

  // N is context size, P = position of central phone (must be 0).
  int32_t num_leaves = 0, P = 0, N = 1;
  EventMap *pdf_map = GetStubMap(P, phone_sets, phone2num_pdf_classes,
                                 share_roots, &num_leaves);
  return new ContextDependency(N, P, pdf_map);
}

}  // namespace khg
