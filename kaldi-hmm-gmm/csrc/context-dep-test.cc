// kaldi-hmm-gmm/csrc/context-dep-test.cc
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2026  Xiaomi Corporation
//
// Ported from ../kaldi/src/tree/context-dep-test.cc (gtest-ified), restricted
// to the APIs present in this repo (context-dep.h): MonophoneContextDependency,
// Compute and GetPdfInfo, plus ContextDependency Write/Read.  kaldi's
// TestContextDep (GaussClusterable statistics) is a clusterable-classes test,
// not a context-dependency test, and kaldi's TestGenRandContextDependency uses
// GenRandContextDependency which is not in this repo; both are skipped.  The
// I/O coverage of TestGenRandContextDependency is re-captured here on a
// monophone tree.

#include "kaldi-hmm-gmm/csrc/context-dep.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

namespace {

// Uniform integer in [low, high] (kaldi RandInt semantics; test-local).
int32_t RandIntLocal(int32_t low, int32_t high) {
  return low + static_cast<int32_t>(RandUniform() * (high - low + 1));
}

// kaldi's TestMonophoneContextDependency phone set: up to 20 phones drawn
// uniformly from 1..30, each with 3 pdf-classes.  `phone2num_classes` is
// indexed by phone (0 for phones not in the set).
void GenRandPhones(std::vector<int32_t> *phones,
                   std::vector<int32_t> *phone2num_classes) {
  std::set<int32_t> phones_set;
  for (int32_t i = 0; i < 20; ++i) {
    phones_set.insert(RandIntLocal(1, 30));
  }
  phones->assign(phones_set.begin(), phones_set.end());
  phone2num_classes->assign(1 + phones->back(), 0);
  for (size_t i = 0; i < phones->size(); ++i) {
    (*phone2num_classes)[(*phones)[i]] = 3;
  }
}

}  // namespace

TEST(ContextDep, MonophoneCompute) {
  srand(1);
  std::vector<int32_t> phones;
  std::vector<int32_t> phone2num_classes;
  GenRandPhones(&phones, &phone2num_classes);

  ContextDependency *cd = MonophoneContextDependency(phones, phone2num_classes);
  EXPECT_EQ(cd->ContextWidth(), 1);
  EXPECT_EQ(cd->CentralPosition(), 0);
  // monophone tree: one pdf per (phone, pdf-class), nothing shared
  EXPECT_EQ(cd->NumPdfs(), static_cast<int32_t>(phones.size()) * 3);

  std::set<int32_t> all_pdfs;
  for (size_t i = 0; i < phones.size(); ++i) {
    for (int32_t pdf_class = 0; pdf_class < 3; ++pdf_class) {
      std::vector<int32_t> phoneseq(1, phones[i]);
      int32_t pdf_id = -1;
      EXPECT_TRUE(cd->Compute(phoneseq, pdf_class, &pdf_id));
      EXPECT_GE(pdf_id, 0);
      EXPECT_LT(pdf_id, cd->NumPdfs());
      all_pdfs.insert(pdf_id);
    }
  }
  EXPECT_EQ(all_pdfs.size(), phones.size() * 3u);

  // out-of-range pdf-classes and uncovered phones must fail
  {
    std::vector<int32_t> phoneseq(1, phones[0]);
    int32_t pdf_id = -1;
    EXPECT_FALSE(cd->Compute(phoneseq, 3, &pdf_id));
    EXPECT_FALSE(cd->Compute(phoneseq, -1, &pdf_id));
  }
  {
    // phone 0 is the BOS/EOS placeholder, never a real phone here
    std::vector<int32_t> phoneseq(1, 0);
    int32_t pdf_id = -1;
    EXPECT_FALSE(cd->Compute(phoneseq, 0, &pdf_id));
  }

  delete cd;
}

TEST(ContextDep, MonophoneGetPdfInfo) {
  srand(1);
  std::vector<int32_t> phones;
  std::vector<int32_t> phone2num_classes;
  GenRandPhones(&phones, &phone2num_classes);

  ContextDependency *cd = MonophoneContextDependency(phones, phone2num_classes);

  std::vector<std::vector<std::pair<int32_t, int32_t>>> pdf_info;
  cd->GetPdfInfo(phones, phone2num_classes, &pdf_info);
  ASSERT_EQ(pdf_info.size(), phones.size() * 3);
  for (size_t i = 0; i < pdf_info.size(); ++i) {
    // monophone: each pdf corresponds to exactly one (phone, pdf-class);
    // kaldi spot-checks one random pdf, we check them all.
    ASSERT_EQ(pdf_info[i].size(), 1u);
    int32_t phone = pdf_info[i][0].first;
    int32_t pdf_class = pdf_info[i][0].second;
    std::vector<int32_t> phoneseq(1, phone);
    int32_t pdf_id = -1;
    EXPECT_TRUE(cd->Compute(phoneseq, pdf_class, &pdf_id));
    EXPECT_EQ(pdf_id, static_cast<int32_t>(i));
  }

  delete cd;
}

TEST(ContextDep, WriteReadRoundTrip) {
  srand(1);
  std::vector<int32_t> phones;
  std::vector<int32_t> phone2num_classes;
  GenRandPhones(&phones, &phone2num_classes);

  ContextDependency *cd = MonophoneContextDependency(phones, phone2num_classes);

  for (int32_t mode = 0; mode < 2; ++mode) {
    bool binary = (mode == 1);
    std::ostringstream oss1;
    cd->Write(oss1, false);

    const char *filename = "tmp_context_dep_test";
    {
      kaldiio::Output ko(filename, binary);
      cd->Write(ko.Stream(), binary);
    }
    ContextDependency cd2;
    {
      bool binary_in = false;
      kaldiio::Input ki(filename, &binary_in);
      cd2.Read(ki.Stream(), binary_in);
    }
    std::remove(filename);

    std::ostringstream oss2;
    cd2.Write(oss2, false);
    EXPECT_EQ(oss1.str(), oss2.str());

    // the re-read tree must answer Compute identically
    EXPECT_EQ(cd2.NumPdfs(), cd->NumPdfs());
    for (size_t i = 0; i < phones.size(); ++i) {
      for (int32_t pdf_class = 0; pdf_class < 3; ++pdf_class) {
        std::vector<int32_t> phoneseq(1, phones[i]);
        int32_t pdf1 = -1;
        int32_t pdf2 = -1;
        EXPECT_TRUE(cd->Compute(phoneseq, pdf_class, &pdf1));
        EXPECT_TRUE(cd2.Compute(phoneseq, pdf_class, &pdf2));
        EXPECT_EQ(pdf1, pdf2);
      }
    }
  }

  delete cd;
}

}  // namespace khg
