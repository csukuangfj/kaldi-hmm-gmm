// kaldi-hmm-gmm/csrc/build-tree-utils.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_BUILD_TREE_UTILS_H_
#define KALDI_HMM_GMM_CSRC_BUILD_TREE_UTILS_H_
// this file is copied and modified from
// kaldi/src/tree/build-tree-utils.h

#include <vector>

#include "kaldi-hmm-gmm/csrc/event-map.h"

namespace khg {

/// GetStubMap is used in tree-building functions to get the initial
/// to-states map, before the decision-tree-building process.  It creates
/// a simple map that splits on groups of phones.  For the set of phones in
/// phone_sets[i] it creates either: if share_roots[i] == true, a single
/// leaf node, or if share_roots[i] == false, separate root nodes for
/// each HMM-position (it goes up to the highest position for any
/// phone in the set, although it will warn if you share roots between
/// phones with different numbers of states, which is a weird thing to
/// do but should still work.  If any phone is present
/// in "phone_sets" but "phone2num_pdf_classes" does not map it to a length,
/// it is an error.  Note that the behaviour of the resulting map is
/// undefined for phones not present in "phone_sets".
/// At entry, this function should be called with (*num_leaves == 0).
/// It will number the leaves starting from (*num_leaves).

EventMap *GetStubMap(
    int32_t P, const std::vector<std::vector<int32_t>> &phone_sets,
    const std::vector<int32_t> &phone2num_pdf_classes,
    const std::vector<bool> &share_roots,  // indexed by index into phone_sets.
    int32_t *num_leaves);
/// Note: GetStubMap with P = 0 can be used to get a standard monophone system.

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_BUILD_TREE_UTILS_H_
