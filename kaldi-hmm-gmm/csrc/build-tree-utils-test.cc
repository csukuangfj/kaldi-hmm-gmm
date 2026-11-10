// kaldi-hmm-gmm/csrc/build-tree-utils-test.cc
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2026  Xiaomi Corporation
//
// New tests for GetStubMap (kaldi/src/tree/build-tree-utils.{h,cc} behavior).
//
// Tests for GetStubMap (../kaldi/src/tree/build-tree-utils.cc):
//   - single phone set with shared roots -> one ConstantEventMap leaf
//   - single phone set with non-shared roots -> TableEventMap on kPdfClass,
//     one leaf per HMM position, numbered sequentially from *num_leaves
//   - several single-phone sets (|sets| <= 2 * highest phone) ->
//     TableEventMap on the phone key
//   - sets of size > 1 -> recursive SplitEventMap on the phone key

#include "kaldi-hmm-gmm/csrc/build-tree-utils.h"

#include <cstdint>
#include <map>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/event-map.h"

namespace khg {

TEST(BuildTreeUtils, GetStubMap) {
  srand(1);
  const int32_t P = 0;  // phone key used in the events

  // Case 1: one phone set with shared roots -> a single leaf.
  {
    std::vector<std::vector<int32_t>> phone_sets;
    phone_sets.push_back(std::vector<int32_t>{2, 5});
    std::vector<int32_t> phone2num_pdf_classes = {0, 0, 3, 0, 0, 2};
    std::vector<bool> share_roots;
    share_roots.push_back(true);
    int32_t num_leaves = 0;

    EventMap *root =
        GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots,
                   &num_leaves);

    EXPECT_EQ(num_leaves, 1);
    EXPECT_NE(dynamic_cast<ConstantEventMap *>(root), nullptr);

    EventAnswerType ans = -100;
    EventType event;  // the constant map ignores the event
    EXPECT_TRUE(root->Map(event, &ans));
    EXPECT_EQ(ans, 0);
    delete root;
  }

  // Case 2: one phone set with non-shared roots -> one leaf per HMM
  // position (kPdfClass), numbered sequentially from *num_leaves.
  {
    std::vector<std::vector<int32_t>> phone_sets;
    phone_sets.push_back(std::vector<int32_t>{1, 2});
    std::vector<int32_t> phone2num_pdf_classes = {0, 2, 2};
    std::vector<bool> share_roots;
    share_roots.push_back(false);
    int32_t num_leaves = 10;  // leaves must be numbered starting from here

    EventMap *root =
        GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots,
                   &num_leaves);

    EXPECT_EQ(num_leaves, 12);
    EXPECT_NE(dynamic_cast<TableEventMap *>(root), nullptr);

    for (int32_t pos = 0; pos < 2; ++pos) {
      EventType event;
      event.push_back(MakeEventPair(kPdfClass, pos));
      EventAnswerType ans = -100;
      EXPECT_TRUE(root->Map(event, &ans));
      EXPECT_EQ(ans, 10 + pos);
    }
    // no HMM position 2 for these phones
    EventType event;
    event.push_back(MakeEventPair(kPdfClass, 2));
    EventAnswerType ans = -100;
    EXPECT_FALSE(root->Map(event, &ans));
    delete root;
  }

  // Case 3: several single-phone sets (2 <= 2 * highest phone = 10) ->
  // TableEventMap on the phone key; each phone gets its own stub.
  {
    std::vector<std::vector<int32_t>> phone_sets;
    phone_sets.push_back(std::vector<int32_t>{3});
    phone_sets.push_back(std::vector<int32_t>{5});
    std::vector<int32_t> phone2num_pdf_classes = {0, 0, 0, 1, 0, 2};
    std::vector<bool> share_roots;
    share_roots.push_back(true);
    share_roots.push_back(false);
    int32_t num_leaves = 0;

    EventMap *root =
        GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots,
                   &num_leaves);

    // table split on the phone key (not the recursive split)
    EXPECT_NE(dynamic_cast<TableEventMap *>(root), nullptr);
    // leaf 0: shared root of phone 3; leaves 1 and 2: positions of phone 5
    EXPECT_EQ(num_leaves, 3);

    {
      EventType event;
      event.push_back(MakeEventPair(P, 3));
      EventAnswerType ans = -100;
      EXPECT_TRUE(root->Map(event, &ans));
      EXPECT_EQ(ans, 0);
    }
    for (int32_t pos = 0; pos < 2; ++pos) {
      EventType event;  // sorted on key: kPdfClass (-1) < P (0)
      event.push_back(MakeEventPair(kPdfClass, pos));
      event.push_back(MakeEventPair(P, 5));
      EventAnswerType ans = -100;
      EXPECT_TRUE(root->Map(event, &ans));
      EXPECT_EQ(ans, 1 + pos);
    }
    delete root;
  }

  // Case 4: sets of size > 1 -> recursive split on the phone key; the
  // first-half phones map to the first sub-tree.
  {
    std::vector<std::vector<int32_t>> phone_sets;
    phone_sets.push_back(std::vector<int32_t>{1, 2});
    phone_sets.push_back(std::vector<int32_t>{3, 4});
    std::vector<int32_t> phone2num_pdf_classes = {0, 2, 2, 2, 2};
    std::vector<bool> share_roots;
    share_roots.push_back(true);
    share_roots.push_back(true);
    int32_t num_leaves = 0;

    EventMap *root =
        GetStubMap(P, phone_sets, phone2num_pdf_classes, share_roots,
                   &num_leaves);

    // max set size 2 -> SplitEventMap, one shared leaf per set
    EXPECT_NE(dynamic_cast<SplitEventMap *>(root), nullptr);
    EXPECT_EQ(num_leaves, 2);

    const int32_t expected[4] = {0, 0, 1, 1};
    for (int32_t phone = 1; phone <= 4; ++phone) {
      EventType event;
      event.push_back(MakeEventPair(P, phone));
      EventAnswerType ans = -100;
      EXPECT_TRUE(root->Map(event, &ans));
      EXPECT_EQ(ans, expected[phone - 1]);
    }
    delete root;
  }
}

}  // namespace khg
