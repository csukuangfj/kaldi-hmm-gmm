// kaldi-hmm-gmm/csrc/hmm-utils-test.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2015  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/hmm/hmm-utils-test.cc
//

#include "kaldi-hmm-gmm/csrc/hmm-utils.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"

namespace khg {

// Two 3-emitting-state phones (1 and 2); each state has a self-loop (index 0)
// and a forward transition (index 1).
static const char kTopoStr[] = R"(
<Topology>
<TopologyEntry>
<ForPhones> 1 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones> 2 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
</Topology>
)";

static void BuildTransitionModel(HmmTopology *topo,
                                 std::vector<ContextDependency *> *ctx_dep,
                                 TransitionModel **trans_model) {
  std::istringstream is(kTopoStr);
  bool binary = false;
  topo->Read(is, binary);

  std::vector<int32_t> phone2num_pdf_classes;
  topo->GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

  ContextDependency *dep =
      MonophoneContextDependency(topo->GetPhones(), phone2num_pdf_classes);
  ctx_dep->push_back(dep);
  *trans_model = new TransitionModel(*dep, *topo);
}

// non-reordered visit of one state: [self* k, forward]
static void AppendStateVisit(const TransitionModel &tm, int32_t trans_state,
                             int32_t num_self_loops, bool reordered,
                             std::vector<int32_t> *alignment) {
  int32_t self_loop = tm.SelfLoopOf(trans_state);
  int32_t forward = tm.PairToTransitionId(trans_state, 1);
  KHG_ASSERT(self_loop == tm.PairToTransitionId(trans_state, 0));
  if (reordered) {
    alignment->push_back(forward);
    for (int32_t i = 0; i < num_self_loops; ++i) alignment->push_back(self_loop);
  } else {
    for (int32_t i = 0; i < num_self_loops; ++i) alignment->push_back(self_loop);
    alignment->push_back(forward);
  }
}

// One full instance of "phone" (3 emitting states), non-reordered or
// reordered layout.
static std::vector<int32_t> PhoneInstance(const TransitionModel &tm,
                                          int32_t phone, bool reordered) {
  // Collect this phone's transition-states ordered by hmm-state.
  std::vector<std::pair<int32_t, int32_t>> hmm_state2trans_state;
  for (int32_t trans_state = 1; trans_state <= tm.NumTransitionStates();
       ++trans_state) {
    if (tm.TransitionStateToPhone(trans_state) != phone) continue;
    int32_t hmm_state = tm.TransitionStateToHmmState(trans_state);
    if (hmm_state < 3)  // only the emitting states of our 3-state topology
      hmm_state2trans_state.emplace_back(hmm_state, trans_state);
  }
  std::sort(hmm_state2trans_state.begin(), hmm_state2trans_state.end());

  std::vector<int32_t> alignment;
  for (const auto &p : hmm_state2trans_state)
    AppendStateVisit(tm, p.second, 1, reordered, &alignment);
  return alignment;
}

TEST(HmmUtils, SplitToPhonesNonReordered) {
  HmmTopology topo;
  std::vector<ContextDependency *> deps;
  TransitionModel *tm = nullptr;
  BuildTransitionModel(&topo, &deps, &tm);

  // phone 1 then phone 2, both non-reordered.
  std::vector<int32_t> ali = PhoneInstance(*tm, 1, false);
  std::vector<int32_t> ali2 = PhoneInstance(*tm, 2, false);
  ali.insert(ali.end(), ali2.begin(), ali2.end());

  std::vector<std::vector<int32_t>> split;
  bool ok = SplitToPhones(*tm, ali, &split);
  EXPECT_TRUE(ok);
  ASSERT_EQ(split.size(), 2u);
  EXPECT_EQ(split[0].size() + split[1].size(), ali.size());
  EXPECT_EQ(tm->TransitionIdToPhone(split[0][0]), 1);
  EXPECT_EQ(tm->TransitionIdToPhone(split[1][0]), 2);

  delete tm;
  delete deps[0];
}

TEST(HmmUtils, SplitToPhonesAdjacentSamePhoneNotMerged) {
  HmmTopology topo;
  std::vector<ContextDependency *> deps;
  TransitionModel *tm = nullptr;
  BuildTransitionModel(&topo, &deps, &tm);

  // two consecutive instances of the SAME phone must NOT be merged.
  std::vector<int32_t> instance1 = PhoneInstance(*tm, 1, false);
  std::vector<int32_t> instance2 = PhoneInstance(*tm, 1, false);
  std::vector<int32_t> ali = instance1;
  ali.insert(ali.end(), instance2.begin(), instance2.end());

  std::vector<std::vector<int32_t>> split;
  bool ok = SplitToPhones(*tm, ali, &split);
  EXPECT_TRUE(ok);
  ASSERT_EQ(split.size(), 2u);
  EXPECT_EQ(split[0], instance1);
  EXPECT_EQ(split[1], instance2);

  delete tm;
  delete deps[0];
}

TEST(HmmUtils, SplitToPhonesReordered) {
  HmmTopology topo;
  std::vector<ContextDependency *> deps;
  TransitionModel *tm = nullptr;
  BuildTransitionModel(&topo, &deps, &tm);

  // phone 1 + phone 2, reordered layout (out-transition before self-loops).
  std::vector<int32_t> ali = PhoneInstance(*tm, 1, true);
  std::vector<int32_t> ali2 = PhoneInstance(*tm, 2, true);
  ali.insert(ali.end(), ali2.begin(), ali2.end());

  std::vector<std::vector<int32_t>> split;
  bool ok = SplitToPhones(*tm, ali, &split);
  EXPECT_TRUE(ok);
  ASSERT_EQ(split.size(), 2u);
  EXPECT_EQ(tm->TransitionIdToPhone(split[0][0]), 1);
  EXPECT_EQ(tm->TransitionIdToPhone(split[1][0]), 2);

  delete tm;
  delete deps[0];
}

TEST(HmmUtils, SplitToPhonesIncomplete) {
  HmmTopology topo;
  std::vector<ContextDependency *> deps;
  TransitionModel *tm = nullptr;
  BuildTransitionModel(&topo, &deps, &tm);

  // drop the last transition-id (phone does not end at a final transition):
  // kaldi semantics: still splits, but returns false.
  std::vector<int32_t> ali = PhoneInstance(*tm, 1, false);
  ali.pop_back();

  std::vector<std::vector<int32_t>> split;
  bool ok = SplitToPhones(*tm, ali, &split);
  EXPECT_FALSE(ok);
  EXPECT_EQ(split.size(), 1u);

  delete tm;
  delete deps[0];
}

TEST(HmmUtils, GetPdfToPhonesMap) {
  HmmTopology topo;
  std::vector<ContextDependency *> deps;
  TransitionModel *tm = nullptr;
  BuildTransitionModel(&topo, &deps, &tm);

  std::vector<std::set<int32_t>> pdf2phones;
  GetPdfToPhonesMap(*tm, &pdf2phones);
  EXPECT_EQ(pdf2phones.size(), tm->NumPdfs());
  // 2 phones x 3 pdf-classes, no sharing.
  for (const auto &phones : pdf2phones) EXPECT_EQ(phones.size(), 1u);

  delete tm;
  delete deps[0];
}

}  // namespace khg
