// kaldi-hmm-gmm/csrc/build-tree-utils.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/build-tree-utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/event-map.h"
#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

EventMap *GetStubMap(int32_t P,
                     const std::vector<std::vector<int32_t>> &phone_sets,
                     const std::vector<int32_t> &phone2num_pdf_classes,
                     const std::vector<bool> &share_roots,
                     int32_t *num_leaves_out) {
  {  // Checking inputs.
    KHG_ASSERT(!phone_sets.empty() && share_roots.size() == phone_sets.size());
    std::set<int32_t> all_phones;
    for (size_t i = 0; i < phone_sets.size(); ++i) {
      KHG_ASSERT(IsSortedAndUniq(phone_sets[i]));
      KHG_ASSERT(!phone_sets[i].empty());

      for (auto p : phone_sets[i]) {
        KHG_ASSERT(all_phones.count(p) == 0);  // check not present.
        all_phones.insert(p);
      }
    }
  }

  // Initially create a single leaf for each phone set.

  size_t max_set_size = 0;
  int32_t highest_numbered_phone = 0;
  for (size_t i = 0; i < phone_sets.size(); ++i) {
    max_set_size = std::max(max_set_size, phone_sets[i].size());
    highest_numbered_phone =
        std::max(highest_numbered_phone,
                 *std::max_element(phone_sets[i].begin(), phone_sets[i].end()));
  }

  if (phone_sets.size() ==
      1) {                 // there is only one set so the recursion finishes.
    if (share_roots[0]) {  // if "shared roots" return a single leaf.
      return new ConstantEventMap((*num_leaves_out)++);
    } else {  // not sharing roots -> work out the length and return a
              // TableEventMap splitting on length.
      EventAnswerType max_len = 0;
      for (size_t i = 0; i < phone_sets[0].size(); ++i) {
        EventAnswerType len;
        EventValueType phone = phone_sets[0][i];
        KHG_ASSERT(static_cast<size_t>(phone) < phone2num_pdf_classes.size());
        len = phone2num_pdf_classes[phone];
        KHG_ASSERT(len > 0);
        if (i == 0) {
          max_len = len;
        } else {
          if (len != max_len) {
            KHG_WARN << "Mismatching lengths within a phone set: " << len
                     << " vs. " << max_len
                     << " [unusual, but not necessarily fatal]. ";
            max_len = std::max(len, max_len);
          }
        }
      }
      std::map<EventValueType, EventAnswerType> m;
      for (EventAnswerType p = 0; p < max_len; p++) m[p] = (*num_leaves_out)++;
      return new TableEventMap(kPdfClass,  // split on hmm-position
                               m);
    }
  } else if (max_set_size == 1 && static_cast<int32_t>(phone_sets.size()) <=
                                      2 * highest_numbered_phone) {
    // create table map splitting on phone-- more efficient.
    // the part after the && checks that this would not contain a very sparse
    // vector.
    std::map<EventValueType, EventMap *> m;
    for (size_t i = 0; i < phone_sets.size(); ++i) {
      std::vector<std::vector<int32_t>> phone_sets_tmp;
      phone_sets_tmp.push_back(phone_sets[i]);
      std::vector<bool> share_roots_tmp;
      share_roots_tmp.push_back(share_roots[i]);
      EventMap *this_stub = GetStubMap(P, phone_sets_tmp, phone2num_pdf_classes,
                                       share_roots_tmp, num_leaves_out);
      KHG_ASSERT(m.count(phone_sets_tmp[0][0]) == 0);
      m[phone_sets_tmp[0][0]] = this_stub;
    }
    return new TableEventMap(P, m);
  } else {
    // Do a split.  Recurse.
    size_t half_sz = phone_sets.size() / 2;
    std::vector<std::vector<int32_t>>::const_iterator half_phones =
        phone_sets.begin() + half_sz;
    std::vector<bool>::const_iterator half_share =
        share_roots.begin() + half_sz;
    std::vector<std::vector<int32_t>> phone_sets_1, phone_sets_2;
    std::vector<bool> share_roots_1, share_roots_2;
    phone_sets_1.insert(phone_sets_1.end(), phone_sets.begin(), half_phones);
    phone_sets_2.insert(phone_sets_2.end(), half_phones, phone_sets.end());
    share_roots_1.insert(share_roots_1.end(), share_roots.begin(), half_share);
    share_roots_2.insert(share_roots_2.end(), half_share, share_roots.end());

    EventMap *map1 = GetStubMap(P, phone_sets_1, phone2num_pdf_classes,
                                share_roots_1, num_leaves_out);
    EventMap *map2 = GetStubMap(P, phone_sets_2, phone2num_pdf_classes,
                                share_roots_2, num_leaves_out);

    std::vector<EventKeyType> all_in_first_set;
    for (size_t i = 0; i < half_sz; i++)
      for (size_t j = 0; j < phone_sets[i].size(); j++)
        all_in_first_set.push_back(phone_sets[i][j]);
    std::sort(all_in_first_set.begin(), all_in_first_set.end());
    KHG_ASSERT(IsSortedAndUniq(all_in_first_set));
    return new SplitEventMap(P, all_in_first_set, map1, map2);
  }
}

}  // namespace khg
