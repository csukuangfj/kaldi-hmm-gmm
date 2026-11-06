// kaldi-hmm-gmm/csrc/transition-model-test.cc
//
// Copyright 2014  Johns Hopkins University
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/hmm/transition-model-test.cc (gtest-ified).  It
// covers:
//   - building a TransitionModel from a random HmmTopology plus a tree
//   - text and binary Write/Read roundtrips
//   - the TransitionId <-> TransitionState/Pdf/phone mappings
// kaldi builds its tree with build-tree (GenRandContextDependencyLarge); we
// use the simpler MonophoneContextDependency path from context-dep.h, which
// exercises the same mappings.  kaldi's Compatible() check and MapUpdate()
// are not applicable here: those functions are not part of this repo.

#include "kaldi-hmm-gmm/csrc/transition-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

namespace {

// Uniform integer in [low, high] (kaldi RandInt semantics; test-local).
int32_t RandIntLocal(int32_t low, int32_t high) {
  return low + static_cast<int32_t>(RandUniform() * (high - low + 1));
}

// Relative comparison.  kaldi AssertEqual/ApproxEqual is
// |a-b| <= tol * (|a| + |b|) (../kaldi/src/base/kaldi-math.h);
// we use the stricter |a-b| <= 0.5 * tol * (|a| + |b|).
void ExpectEqual(double a, double b, double tol = 1.0e-6) {
  EXPECT_LE(std::fabs(a - b), 0.5 * tol * (std::fabs(a) + std::fabs(b)))
      << "a = " << a << ", b = " << b;
}

// Test-local port of ../kaldi/src/hmm/hmm-test-utils.cc GetDefaultTopology():
// the normal 3-state topology (self-loop + forward) for the given phones.
HmmTopology GetDefaultTopology(const std::vector<int32_t> &phones_in) {
  std::vector<int32_t> phones(phones_in);
  std::sort(phones.begin(), phones.end());
  KHG_ASSERT(IsSortedAndUniq(phones) && !phones.empty());

  std::ostringstream topo_string;
  topo_string << "<Topology>\n"
              << "<TopologyEntry>\n"
              << "<ForPhones> ";
  for (size_t i = 0; i < phones.size(); ++i) {
    topo_string << phones[i] << " ";
  }
  topo_string << "</ForPhones>\n"
              << "<State> 0 <PdfClass> 0\n"
              << "<Transition> 0 0.5\n"
              << "<Transition> 1 0.5\n"
              << "</State> \n"
              << "<State> 1 <PdfClass> 1 \n"
              << "<Transition> 1 0.5\n"
              << "<Transition> 2 0.5\n"
              << "</State>  \n"
              << " <State> 2 <PdfClass> 2\n"
              << " <Transition> 2 0.5\n"
              << " <Transition> 3 0.5\n"
              << " </State>   \n"
              << " <State> 3 </State>\n"
              << " </TopologyEntry>\n"
              << " </Topology>\n";

  HmmTopology topo;
  std::istringstream iss(topo_string.str());
  topo.Read(iss, false);
  return topo;
}

// Test-local port of ../kaldi/src/hmm/hmm-test-utils.cc
// GenRandTopology(phones, num_pdf_classes): groups the phones by their number
// of pdf-classes; each group gets one topology entry that is randomly either
// ergodic (with a nonemitting start state) or feedforward.
HmmTopology GenRandTopology(const std::vector<int32_t> &phones_in,
                            const std::vector<int32_t> &num_pdf_classes) {
  std::vector<int32_t> phones(phones_in);
  std::sort(phones.begin(), phones.end());
  KHG_ASSERT(IsSortedAndUniq(phones) && !phones.empty());

  std::ostringstream topo_string;

  std::map<int32_t, std::vector<int32_t>> num_pdf_classes_to_phones;
  for (size_t i = 0; i < phones.size(); ++i) {
    int32_t p = phones[i];
    KHG_ASSERT(static_cast<size_t>(p) < num_pdf_classes.size());
    int32_t n = num_pdf_classes[p];
    KHG_ASSERT(n > 0 && "num-pdf-classes cannot be zero.");
    num_pdf_classes_to_phones[n].push_back(p);
  }

  topo_string << "<Topology>\n";
  std::map<int32_t, std::vector<int32_t>>::const_iterator iter =
      num_pdf_classes_to_phones.begin();
  std::map<int32_t, std::vector<int32_t>>::const_iterator end =
      num_pdf_classes_to_phones.end();
  for (; iter != end; ++iter) {
    topo_string << "<TopologyEntry>\n"
                << "<ForPhones> ";
    int32_t this_num_pdf_classes = iter->first;
    const std::vector<int32_t> &group_phones = iter->second;
    for (size_t i = 0; i < group_phones.size(); ++i) {
      topo_string << group_phones[i] << " ";
    }
    topo_string << "</ForPhones> ";
    bool ergodic = (RandIntLocal(0, 1) == 0);
    if (ergodic) {
      // Note, this type of topology is not something we ever use in practice-
      // it has an initial nonemitting state (no PdfClass specified).  But it's
      // supported so we're testing it.
      std::vector<int32_t> state_to_pdf_class;
      state_to_pdf_class.push_back(-1);  // state zero, nonemitting.
      for (int32_t i = 0; i < this_num_pdf_classes; ++i) {
        int32_t num_states = RandIntLocal(1, 2);
        for (int32_t j = 0; j < num_states; ++j) {
          state_to_pdf_class.push_back(i);
        }
      }
      state_to_pdf_class.push_back(-1);  // final non-emitting state.
      {  // state zero is nonemitting.  This is not something used in any
         // current example script.
        topo_string << "<State> 0\n";
        float prob = 1.0 / (state_to_pdf_class.size() - 2);
        for (size_t i = 1; i + 1 < state_to_pdf_class.size(); ++i) {
          topo_string << "<Transition> " << i << ' ' << prob << '\n';
        }
        topo_string << "</State>\n";
      }
      // ergodic part.
      for (size_t i = 1; i + 1 < state_to_pdf_class.size(); ++i) {
        float prob = 1.0 / (state_to_pdf_class.size() - 1);
        topo_string << "<State> " << i << " <PdfClass> "
                    << state_to_pdf_class[i] << '\n';
        for (size_t j = 1; j < state_to_pdf_class.size(); ++j) {
          topo_string << "<Transition> " << j << ' ' << prob << '\n';
        }
        topo_string << "</State>\n";
      }
      // final, nonemitting state.  No pdf-class, no transitions.
      topo_string << "<State> " << (state_to_pdf_class.size() - 1)
                  << " </State>\n";
    } else {
      // feedforward topology.
      int32_t cur_state = 0;
      for (int32_t pdf_class = 0; pdf_class < this_num_pdf_classes;
           ++pdf_class) {
        int32_t this_num_states = RandIntLocal(1, 2);
        for (int32_t s = 0; s < this_num_states; ++s) {
          topo_string << "<State> " << cur_state << " <PdfClass> " << pdf_class
                      << "\n<Transition> " << cur_state << " 0.5\n<Transition> "
                      << (cur_state + 1) << " 0.5\n</State>\n";
          cur_state++;
        }
      }
      // final, non-emitting state.
      topo_string << "<State> " << cur_state << " </State>\n";
    }
    topo_string << "</TopologyEntry>\n";
  }
  topo_string << "</Topology>\n";

  HmmTopology topo;
  std::istringstream iss(topo_string.str());
  topo.Read(iss, false);
  return topo;
}

// Builds a random monophone system: a random topology over a random phone set
// (kaldi's GenRandTransitionModel phone convention: phone 1 plus a random
// subset of 2..19), the matching monophone tree, and the TransitionModel over
// both.  kaldi's test builds the tree via build-tree; we use
// MonophoneContextDependency (context-dep.h) which exercises the same
// TransitionModel mappings.
void BuildRandTransitionModel(HmmTopology *topo, ContextDependency **ctx_dep,
                              TransitionModel **trans_model) {
  std::vector<int32_t> phones;
  phones.push_back(1);
  for (int32_t i = 2; i < 20; ++i) {
    if (RandIntLocal(0, 1) == 0) {
      phones.push_back(i);
    }
  }
  std::vector<int32_t> num_pdf_classes(phones.back() + 1, -1);
  for (size_t i = 0; i < phones.size(); ++i) {
    num_pdf_classes[phones[i]] = RandIntLocal(1, 5);
  }
  *topo = GenRandTopology(phones, num_pdf_classes);

  std::vector<int32_t> phone2num_pdf_classes;
  topo->GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
  *ctx_dep =
      MonophoneContextDependency(topo->GetPhones(), phone2num_pdf_classes);
  *trans_model = new TransitionModel(**ctx_dep, *topo);
}

// Exhaustively checks the TransitionId <-> TransitionState/Pdf/phone mappings
// and the transition probabilities against the topology.  (kaldi relies on
// TransitionModel::Check() for most of this; here we verify the public
// mappings explicitly.)
void CheckMappings(const TransitionModel &tm) {
  const HmmTopology &topo = tm.GetTopo();
  int32_t num_ids_seen = 0;
  std::vector<bool> seen_id(tm.NumTransitionIds() + 1, false);

  for (int32_t tstate = 1; tstate <= tm.NumTransitionStates(); ++tstate) {
    int32_t phone = tm.TransitionStateToPhone(tstate);
    int32_t hmm_state = tm.TransitionStateToHmmState(tstate);
    int32_t forward_pdf = tm.TransitionStateToForwardPdf(tstate);
    int32_t self_loop_pdf = tm.TransitionStateToSelfLoopPdf(tstate);

    // the 4-tuple must map back to this transition-state
    EXPECT_EQ(tm.TupleToTransitionState(phone, hmm_state, forward_pdf,
                                        self_loop_pdf),
              tstate);
    EXPECT_GE(forward_pdf, 0);
    EXPECT_GE(self_loop_pdf, 0);
    EXPECT_LT(forward_pdf, tm.NumPdfs());
    EXPECT_LT(self_loop_pdf, tm.NumPdfs());

    const HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(phone);
    ASSERT_GE(hmm_state, 0);
    ASSERT_LT(hmm_state, static_cast<int32_t>(entry.size()));
    const HmmTopology::HmmState &state = entry[hmm_state];

    int32_t self_loop_id = 0;
    float self_loop_prob = 0.0f;
    for (int32_t idx = 0;
         idx < static_cast<int32_t>(state.transitions.size()); ++idx) {
      int32_t tid = tm.PairToTransitionId(tstate, idx);
      EXPECT_FALSE(seen_id[tid]);
      seen_id[tid] = true;
      ++num_ids_seen;

      EXPECT_EQ(tm.TransitionIdToTransitionState(tid), tstate);
      EXPECT_EQ(tm.TransitionIdToPhone(tid), phone);
      EXPECT_TRUE(tm.TransitionIdsEquivalent(tid, tid));

      int32_t dst_state = state.transitions[idx].first;
      bool is_self_loop = (dst_state == hmm_state);
      EXPECT_EQ(tm.IsSelfLoop(tid), is_self_loop);
      bool is_final = (dst_state + 1 == static_cast<int32_t>(entry.size()));
      EXPECT_EQ(tm.IsFinal(tid), is_final);
      EXPECT_EQ(tm.TransitionIdIsStartOfPhone(tid), hmm_state == 0);

      // pdf of a transition-id: self-loop pdf on self-loops, else forward pdf
      int32_t expected_pdf = is_self_loop ? self_loop_pdf : forward_pdf;
      EXPECT_EQ(tm.TransitionIdToPdf(tid), expected_pdf);

      // default transition probs come straight from the topology
      float prob = state.transitions[idx].second;
      EXPECT_GT(prob, 0.0f);
      ExpectEqual(tm.GetTransitionProb(tid), prob, 1e-4);
      ExpectEqual(tm.GetTransitionLogProb(tid), std::log(prob), 1e-4);
      EXPECT_LE(tm.GetTransitionLogProb(tid), 0.0f);
      EXPECT_TRUE(std::isfinite(tm.GetTransitionLogProb(tid)));

      if (is_self_loop) {
        self_loop_id = tid;
        self_loop_prob = prob;
      }
    }
    EXPECT_EQ(tm.SelfLoopOf(tstate), self_loop_id);

    // Compare probabilities, not log-probs: ExpectEqual is a relative check
    // and gets brittle when the exact answer is zero (e.g. log(1) == 0 after
    // renormalizing) while exp()/log() roundtrips leave float noise.
    ExpectEqual(std::exp(tm.GetNonSelfLoopLogProb(tstate)),
                1.0f - self_loop_prob, 1e-4);
    for (int32_t idx = 0;
         idx < static_cast<int32_t>(state.transitions.size()); ++idx) {
      int32_t tid = tm.PairToTransitionId(tstate, idx);
      if (tm.IsSelfLoop(tid)) {
        continue;
      }
      float non_self_prob = state.transitions[idx].second;
      ExpectEqual(
          std::exp(tm.GetTransitionLogProbIgnoringSelfLoops(tid)),
          non_self_prob / (1.0f - self_loop_prob), 1e-4);
    }
  }

  EXPECT_EQ(num_ids_seen, tm.NumTransitionIds());
  for (int32_t tid = 1; tid <= tm.NumTransitionIds(); ++tid) {
    EXPECT_TRUE(seen_id[tid]);
  }
  if (tm.NumTransitionStates() >= 2) {
    // transition-ids of different transition-states are not equivalent
    int32_t tid1 = tm.PairToTransitionId(1, 0);
    int32_t tid2 = tm.PairToTransitionId(2, 0);
    EXPECT_FALSE(tm.TransitionIdsEquivalent(tid1, tid2));
  }
}

}  // namespace

TEST(TransitionModel, MappingConsistency) {
  srand(1);
  for (int32_t trial = 0; trial < 5; ++trial) {
    HmmTopology topo;
    ContextDependency *ctx_dep = nullptr;
    TransitionModel *trans_model = nullptr;
    BuildRandTransitionModel(&topo, &ctx_dep, &trans_model);

    EXPECT_GT(trans_model->NumTransitionStates(), 0);
    EXPECT_GT(trans_model->NumTransitionIds(), 0);
    EXPECT_EQ(trans_model->NumPdfs(), ctx_dep->NumPdfs());
    CheckMappings(*trans_model);

    delete trans_model;
    delete ctx_dep;
  }
}

TEST(TransitionModel, WriteReadRoundTrip) {
  srand(1);
  for (int32_t trial = 0; trial < 2; ++trial) {
    HmmTopology topo;
    ContextDependency *ctx_dep = nullptr;
    TransitionModel *trans_model = nullptr;
    BuildRandTransitionModel(&topo, &ctx_dep, &trans_model);

    for (int32_t mode = 0; mode < 2; ++mode) {
      bool binary = (mode == 1);
      std::ostringstream oss1;
      trans_model->Write(oss1, false);

      const char *filename = "tmp_transition_model_test";
      {
        kaldiio::Output ko(filename, binary);
        trans_model->Write(ko.Stream(), binary);
      }
      TransitionModel trans_model2;
      {
        bool binary_in = false;
        kaldiio::Input ki(filename, &binary_in);
        trans_model2.Read(ki.Stream(), binary_in);
      }
      std::remove(filename);

      std::ostringstream oss2;
      trans_model2.Write(oss2, false);
      EXPECT_EQ(oss1.str(), oss2.str());

      // the re-read model must behave identically
      EXPECT_EQ(trans_model2.NumTransitionIds(),
                trans_model->NumTransitionIds());
      EXPECT_EQ(trans_model2.NumTransitionStates(),
                trans_model->NumTransitionStates());
      EXPECT_EQ(trans_model2.NumPdfs(), trans_model->NumPdfs());
      for (int32_t tid = 1; tid <= trans_model->NumTransitionIds(); ++tid) {
        EXPECT_FLOAT_EQ(trans_model2.GetTransitionLogProb(tid),
                        trans_model->GetTransitionLogProb(tid));
      }
      CheckMappings(trans_model2);
    }

    delete trans_model;
    delete ctx_dep;
  }
}

TEST(TransitionModel, NoSelfLoopTopology) {
  srand(1);
  // A skippable-style phone whose only transition goes straight to the final
  // state: no transition is a self-loop, so SelfLoopOf() must return 0 and the
  // non-self-loop renormalization must be the identity.
  static const char kTopoStr[] = R"(
<Topology>
<TopologyEntry>
<ForPhones> 1 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 1 1.0 </State>
<State> 1 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones> 2 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 1 1.0 </State>
<State> 1 </State>
</TopologyEntry>
</Topology>
)";
  HmmTopology topo;
  {
    std::istringstream iss(kTopoStr);
    topo.Read(iss, false);
  }
  std::vector<int32_t> phone2num_pdf_classes;
  topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
  ContextDependency *ctx_dep =
      MonophoneContextDependency(topo.GetPhones(), phone2num_pdf_classes);
  TransitionModel trans_model(*ctx_dep, topo);

  EXPECT_EQ(trans_model.NumTransitionStates(), 2);
  EXPECT_EQ(trans_model.NumTransitionIds(), 2);
  for (int32_t tstate = 1; tstate <= trans_model.NumTransitionStates();
       ++tstate) {
    EXPECT_EQ(trans_model.SelfLoopOf(tstate), 0);
    ExpectEqual(std::exp(trans_model.GetNonSelfLoopLogProb(tstate)), 1.0,
                1e-4);
    int32_t tid = trans_model.PairToTransitionId(tstate, 0);
    EXPECT_FALSE(trans_model.IsSelfLoop(tid));
    EXPECT_TRUE(trans_model.IsFinal(tid));
    ExpectEqual(std::exp(trans_model.GetTransitionLogProbIgnoringSelfLoops(tid)),
                1.0, 1e-4);
  }
  CheckMappings(trans_model);

  delete ctx_dep;
}

}  // namespace khg
