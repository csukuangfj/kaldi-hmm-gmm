// kaldi-hmm-gmm/csrc/hmm-topology-test.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2015  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/hmm/hmm-topology-test.cc (gtest-ified).  It covers:
//   - text and binary Read/Write roundtrips (fixed and random topologies)
//   - MinLength()
//   - chain topology (<ForwardPdfClass>/<SelfLoopPdfClass>)
// The random-topology helpers are test-local ports of the generators in
// ../kaldi/src/hmm/hmm-test-utils.cc.  kaldi's main() repeats the whole test
// ten times with a random binary flag; here each aspect is its own TEST case
// and every roundtrip is checked in both text and binary mode.

#include "kaldi-hmm-gmm/csrc/hmm-topology.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

namespace {

// Uniform integer in [low, high] (kaldi RandInt semantics; test-local).
int32_t RandIntLocal(int32_t low, int32_t high) {
  return low + static_cast<int32_t>(RandUniform() * (high - low + 1));
}

// Parses a text-mode topology (the format the kaldi test uses as input).
void ReadFromString(const std::string &str, HmmTopology *topo) {
  std::istringstream iss(str);
  topo->Read(iss, false);
}

// Writes `topo` to a temp file in the given mode, reads it back, and checks
// that the canonical text form is unchanged (kaldi compares the text forms of
// the original and the re-read object).
void CheckRoundTrip(const HmmTopology &topo, bool binary) {
  std::ostringstream oss1;
  topo.Write(oss1, false);

  const char *filename = "tmp_hmm_topology_test";
  {
    kaldiio::Output ko(filename, binary);
    topo.Write(ko.Stream(), binary);
  }
  HmmTopology topo2;
  {
    bool binary_in = false;
    kaldiio::Input ki(filename, &binary_in);
    topo2.Read(ki.Stream(), binary_in);
  }
  std::remove(filename);

  std::ostringstream oss2;
  topo2.Write(oss2, false);
  EXPECT_EQ(oss1.str(), oss2.str());
}

// kaldi's fixed input topology: one 3-emitting-state entry for phones 1-9 and
// one 2-emitting-state entry for phones 10, 11, 13 (verbatim from
// ../kaldi/src/hmm/hmm-topology-test.cc).
const char kInputStr[] =
    "<Topology>\n"
    "<TopologyEntry>\n"
    "<ForPhones> 1 2 3 4 5 6 7 8 9 </ForPhones>\n"
    "<State> 0 <PdfClass> 0\n"
    "<Transition> 0 0.5\n"
    "<Transition> 1 0.5\n"
    "</State> \n"
    "<State> 1 <PdfClass> 1 \n"
    "<Transition> 1 0.5\n"
    "<Transition> 2 0.5\n"
    "</State>  \n"
    " <State> 2 <PdfClass> 2\n"
    " <Transition> 2 0.5\n"
    " <Transition> 3 0.5\n"
    " </State>   \n"
    " <State> 3 </State>\n"
    " </TopologyEntry>\n"
    "  <TopologyEntry>\n"
    "  <ForPhones> 10 11 13  </ForPhones>\n"
    "  <State> 0 <PdfClass> 0\n"
    "  <Transition> 0 0.5\n"
    "  <Transition> 1 0.5\n"
    "  </State> \n"
    "  <State> 1 <PdfClass> 1 \n"
    "  <Transition> 1 0.5\n"
    "  <Transition> 2 0.5\n"
    "  </State>  \n"
    " <State> 2 </State>"
    "  </TopologyEntry>\n"
    "  </Topology>\n";

// kaldi's chain-style topology: one emitting state whose forward and self-loop
// pdf-classes differ (verbatim from ../kaldi/src/hmm/hmm-topology-test.cc).
const char kChainInputStr[] =
    "<Topology>\n"
    "<TopologyEntry>\n"
    "<ForPhones> 1 2 3 4 5 6 7 8 9 </ForPhones>\n"
    " <State> 0 <ForwardPdfClass> 0 <SelfLoopPdfClass> 1\n"
    "  <Transition> 0 0.5\n"
    "  <Transition> 1 0.5\n"
    " </State> \n"
    " <State> 1 </State>\n"
    "</TopologyEntry>\n"
    "</Topology>\n";

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

// Test-local port of ../kaldi/src/hmm/hmm-test-utils.cc GenRandTopology():
// random phone list, random number of pdf-classes per phone, then either the
// default or a random topology.
HmmTopology GenRandTopology() {
  std::vector<int32_t> phones;
  phones.push_back(1);
  for (int32_t i = 2; i < 20; ++i) {
    if (RandIntLocal(0, 1) == 0) {
      phones.push_back(i);
    }
  }
  if (RandIntLocal(0, 1) == 0) {
    return GetDefaultTopology(phones);
  } else {
    std::vector<int32_t> num_pdf_classes(phones.back() + 1, -1);
    for (size_t i = 0; i < phones.size(); ++i) {
      num_pdf_classes[phones[i]] = RandIntLocal(1, 5);
    }
    return GenRandTopology(phones, num_pdf_classes);
  }
}

}  // namespace

TEST(HmmTopology, TextReadWriteRoundTrip) {
  srand(1);
  HmmTopology topo;
  ReadFromString(kInputStr, &topo);
  CheckRoundTrip(topo, false);
  CheckRoundTrip(topo, true);
}

TEST(HmmTopology, MinLength) {
  srand(1);
  HmmTopology topo;
  ReadFromString(kInputStr, &topo);
  // 3 emitting states for phones 1-9, 2 emitting states for phones 10/11/13.
  EXPECT_EQ(topo.MinLength(3), 3);
  EXPECT_EQ(topo.MinLength(11), 2);
  EXPECT_EQ(topo.NumPdfClasses(3), 3);
  EXPECT_EQ(topo.NumPdfClasses(11), 2);

  std::vector<int32_t> phones = topo.GetPhones();
  EXPECT_TRUE(IsSortedAndUniq(phones));
  EXPECT_EQ(phones.size(), 12u);  // phones 1-9, 10, 11, 13
  const HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(3);
  EXPECT_EQ(entry.size(), 4u);  // 3 emitting states + final state

  std::vector<int32_t> phone2num_pdf_classes;
  topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
  EXPECT_EQ(phone2num_pdf_classes[3], 3);
  EXPECT_EQ(phone2num_pdf_classes[13], 2);
}

TEST(HmmTopology, ChainTopology) {
  srand(1);
  HmmTopology topo;
  ReadFromString(kChainInputStr, &topo);
  // forward and self-loop pdf-classes differ -> not hmm-like.
  EXPECT_FALSE(topo.IsHmm());
  EXPECT_EQ(topo.MinLength(3), 1);  // a single emitting state
  EXPECT_EQ(topo.NumPdfClasses(3), 2);

  const HmmTopology::TopologyEntry &entry = topo.TopologyForPhone(3);
  ASSERT_EQ(entry.size(), 2u);
  EXPECT_EQ(entry[0].forward_pdf_class, 0);
  EXPECT_EQ(entry[0].self_loop_pdf_class, 1);
  EXPECT_EQ(entry[1].forward_pdf_class, kNoPdf);

  // the roundtrip exercises the binary format's extended (non-hmm) layout.
  CheckRoundTrip(topo, false);
  CheckRoundTrip(topo, true);
}

TEST(HmmTopology, RandomTopologyRoundTrip) {
  srand(1);
  // kaldi makes sure GetDefaultTopology does not crash.
  std::vector<int32_t> default_phones;
  default_phones.push_back(1);
  default_phones.push_back(2);
  GetDefaultTopology(default_phones);

  for (int32_t trial = 0; trial < 10; ++trial) {
    HmmTopology topo = GenRandTopology();
    CheckRoundTrip(topo, false);
    CheckRoundTrip(topo, true);

    std::vector<int32_t> phones = topo.GetPhones();
    EXPECT_TRUE(IsSortedAndUniq(phones));
    for (size_t i = 0; i < phones.size(); ++i) {
      // every phone covers at least one emitting state
      EXPECT_GE(topo.MinLength(phones[i]), 1);
    }
  }
}

}  // namespace khg
