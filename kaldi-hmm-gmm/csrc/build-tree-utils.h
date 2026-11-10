// kaldi-hmm-gmm/csrc/build-tree-utils.h
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_BUILD_TREE_UTILS_H_
#define KALDI_HMM_GMM_CSRC_BUILD_TREE_UTILS_H_
// This file is copied/modified from kaldi/src/tree/build-tree-utils.h

#include <vector>

#include "kaldi-hmm-gmm/csrc/event-map.h"

/**
 * \file build-tree-utils.h
 * \brief Helpers for building decision trees from statistics.
 *
 * This file is copied/modified from kaldi/src/tree/build-tree-utils.h.
 *
 * It provides the tree-building helper \c GetStubMap, which creates the
 * initial to-states (to-pdf) map that seeds the decision-tree-building
 * process: a trivial \c EventMap that splits only on groups of phones and on
 * HMM position.  Kaldi's build-tree-utils.h contains many more helpers for
 * building and refining trees from statistics (see also cluster-utils.h).
 */

namespace khg {

/**
 * \brief Creates the initial to-states map, before the decision-tree-building
 *        process.
 *
 * GetStubMap is used in tree-building functions to get the initial
 * to-states map, before the decision-tree-building process.  It creates
 * a simple map that splits on groups of phones.  For the set of phones in
 * phone_sets[i] it creates either: if share_roots[i] == true, a single
 * leaf node, or if share_roots[i] == false, separate root nodes for
 * each HMM-position (it goes up to the highest position for any
 * phone in the set, although it will warn if you share roots between
 * phones with different numbers of states, which is a weird thing to
 * do but should still work.  If any phone is present
 * in "phone_sets" but "phone2num_pdf_classes" does not map it to a length,
 * it is an error.  Note that the behaviour of the resulting map is
 * undefined for phones not present in "phone_sets".
 * At entry, this function should be called with (*num_leaves == 0).
 * It will number the leaves starting from (*num_leaves).
 *
 * \param P          The central position within the context window.  Note:
 *                   GetStubMap with P = 0 can be used to get a standard
 *                   monophone system.
 * \param phone_sets  Groups of phones; each group gets one shared root (or
 *                   one root per HMM-position).
 * \param phone2num_pdf_classes  Maps from the phone id to the number of
 *                   pdf-classes (i.e. HMM positions) for that phone; every
 *                   phone in \c phone_sets must be present.
 * \param share_roots  Indexed by index into \c phone_sets: if true, the
 *                   phones in that set share a single leaf node; if false,
 *                   each HMM-position gets a separate root node.
 * \param num_leaves  [in/out] Must be 0 at entry; on exit it holds the next
 *                   unused leaf index (the leaves created here are numbered
 *                   starting from its entry value).
 * \return  The newly created stub map; the caller owns it.  Its behaviour is
 *          undefined for phones not present in \c phone_sets.
 */
EventMap *GetStubMap(
    int32_t P, const std::vector<std::vector<int32_t>> &phone_sets,
    const std::vector<int32_t> &phone2num_pdf_classes,
    const std::vector<bool> &share_roots,  // indexed by index into phone_sets.
    int32_t *num_leaves);
/// Note: GetStubMap with P = 0 can be used to get a standard monophone system.

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_BUILD_TREE_UTILS_H_
