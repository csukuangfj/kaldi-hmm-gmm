// kaldi-hmm-gmm/csrc/model-common-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Tests for ../kaldi/src/gmm/model-common.cc (kaldi has no unit tests for it).
// Covers StringToGmmFlags/GmmFlagsToString round-trips, AugmentGmmFlags, and
// GetSplitTargets sanity (total split targets, power rule, min-count).

#include "kaldi-hmm-gmm/csrc/model-common.h"

#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"

namespace khg {

TEST(ModelCommon, GmmFlagsToStringRoundTrip) {
  srand(1);
  // Every combination of the m/v/w/t bits must survive a flags -> string ->
  // flags round-trip.  Note the string itself is canonical ("mvwt" order), so
  // e.g. "vm" round-trips as flags but not as a string.
  for (int32_t f = 0; f <= kGmmAll; ++f) {
    GmmFlagsType flags = static_cast<GmmFlagsType>(f);
    std::string str = GmmFlagsToString(flags);
    GmmFlagsType flags2 = StringToGmmFlags(str);
    EXPECT_EQ(flags2, flags) << "str = " << str;
    // The string form must be a fixed point once canonicalized.
    EXPECT_EQ(GmmFlagsToString(flags2), str);
  }

  EXPECT_EQ(StringToGmmFlags(""), 0);
  EXPECT_EQ(StringToGmmFlags("m"), kGmmMeans);
  EXPECT_EQ(StringToGmmFlags("v"), kGmmVariances);
  EXPECT_EQ(StringToGmmFlags("w"), kGmmWeights);
  EXPECT_EQ(StringToGmmFlags("t"), kGmmTransitions);
  EXPECT_EQ(StringToGmmFlags("mvwt"), kGmmAll);
  // 'a' is an alias for all flags.
  EXPECT_EQ(StringToGmmFlags("a"), kGmmAll);
  EXPECT_EQ(StringToGmmFlags("mav"), kGmmAll);
  // Order in the input string does not matter; output is canonical.
  EXPECT_EQ(GmmFlagsToString(kGmmAll), "mvwt");
  EXPECT_EQ(GmmFlagsToString(StringToGmmFlags("vm")), "mv");
  GmmFlagsType mvt = kGmmMeans | kGmmVariances | kGmmTransitions;
  EXPECT_EQ(StringToGmmFlags("mvt"), mvt);
  EXPECT_EQ(GmmFlagsToString(mvt), "mvt");
}

TEST(ModelCommon, AugmentGmmFlags) {
  srand(1);
  // Variance accumulation implies mean accumulation implies weight
  // accumulation.
  EXPECT_EQ(AugmentGmmFlags(kGmmVariances),
            kGmmVariances | kGmmMeans | kGmmWeights);
  EXPECT_EQ(AugmentGmmFlags(kGmmMeans), kGmmMeans | kGmmWeights);
  EXPECT_EQ(AugmentGmmFlags(kGmmWeights), kGmmWeights);
  // Empty flags get kGmmWeights added regardless.
  EXPECT_EQ(AugmentGmmFlags(0), kGmmWeights);
  // Transitions alone do not imply mean/variance accumulation.
  EXPECT_EQ(AugmentGmmFlags(kGmmTransitions), kGmmTransitions | kGmmWeights);
  EXPECT_EQ(AugmentGmmFlags(kGmmAll), kGmmAll);
  // Augmenting is idempotent and only adds bits.
  for (int32_t f = 0; f <= kGmmAll; ++f) {
    GmmFlagsType flags = static_cast<GmmFlagsType>(f);
    GmmFlagsType augmented = AugmentGmmFlags(flags);
    EXPECT_EQ(AugmentGmmFlags(augmented), augmented);
    EXPECT_EQ(augmented & flags, flags);  // only adds, never removes
  }
}

TEST(ModelCommon, GetSplitTargets) {
  srand(1);
  // Uniform occupancies, tiny min-count: the power rule makes all pdfs equal,
  // so the targets are balanced and sum exactly to the requested count.
  {
    int32_t num_pdfs = 4;
    int32_t target_components = 8;
    FloatVector occs(num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      occs(i) = 100.0f;
    }
    std::vector<int32_t> targets;
    GetSplitTargets(occs, target_components, 0.2f, 0.01f, &targets);
    EXPECT_EQ(static_cast<int32_t>(targets.size()), num_pdfs);
    int32_t total = 0;
    for (int32_t i = 0; i < num_pdfs; ++i) {
      EXPECT_EQ(targets[i], 2);
      total += targets[i];
    }
    EXPECT_EQ(total, target_components);
  }

  // Random positive occupancies, tiny min-count: every split target is used,
  // so the total equals the requested number of components exactly and every
  // pdf keeps at least one component.
  {
    int32_t num_pdfs = 5;
    int32_t target_components = 3 * num_pdfs;
    FloatVector occs(num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      occs(i) = 100.0f * (RandUniform() + 0.5f);
    }
    std::vector<int32_t> targets;
    GetSplitTargets(occs, target_components, 0.2f, 0.01f, &targets);
    EXPECT_EQ(static_cast<int32_t>(targets.size()), num_pdfs);
    int32_t total = 0;
    for (int32_t i = 0; i < num_pdfs; ++i) {
      EXPECT_GE(targets[i], 1);
      total += targets[i];
    }
    EXPECT_EQ(total, target_components);
  }

  // min-count active: a pdf with occupancy occ cannot grow past the largest k
  // with k * min_count < occ (here occ = 10, min_count = 2 -> max 4).
  {
    int32_t num_pdfs = 3;
    FloatVector occs(num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      occs(i) = 10.0f;
    }
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 100, 0.2f, 2.0f, &targets);
    EXPECT_EQ(static_cast<int32_t>(targets.size()), num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      EXPECT_EQ(targets[i], 4);
    }
  }

  // A huge min-count forbids any split: all targets stay at 1.
  {
    int32_t num_pdfs = 3;
    FloatVector occs(num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      occs(i) = 1.0f;
    }
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 100, 0.2f, 100.0f, &targets);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      EXPECT_EQ(targets[i], 1);
    }
  }

  // The power rule concentrates extra components on high-occupancy pdfs:
  // with power = 1 and target = num_pdfs + 1 the largest pdf gets the split.
  {
    int32_t num_pdfs = 2;
    FloatVector occs(num_pdfs);
    occs(0) = 1.0f;
    occs(1) = 100.0f;
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 3, 1.0f, 0.01f, &targets);
    EXPECT_EQ(targets[0], 1);
    EXPECT_EQ(targets[1], 2);
  }

  // Requesting fewer components than there are pdfs cannot reduce the number
  // of components (each pdf starts at one).
  {
    int32_t num_pdfs = 4;
    FloatVector occs(num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      occs(i) = 50.0f * (i + 1);
    }
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 2, 0.2f, 0.01f, &targets);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      EXPECT_EQ(targets[i], 1);
    }
  }
}

}  // namespace khg

// ---------------------------------------------------------------------------
// GetSplitTargets edge cases (../kaldi/src/gmm/model-common.cc): zero
// occupancies and all-equal occupancies.
// ---------------------------------------------------------------------------

namespace khg {

TEST(ModelCommon, GetSplitTargetsEdgeCases) {
  srand(1);

  // Zero-occupancy entries never grow: all extra components go to the pdfs
  // with counts, and the total still hits the target.
  {
    int32_t num_pdfs = 3;
    FloatVector occs(num_pdfs);
    occs(0) = 0.0f;
    occs(1) = 50.0f;
    occs(2) = 100.0f;
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 6, 0.2f, 0.01f, &targets);
    EXPECT_EQ(static_cast<int32_t>(targets.size()), num_pdfs);
    EXPECT_EQ(targets[0], 1);
    EXPECT_EQ(targets[1], 2);
    EXPECT_EQ(targets[2], 3);
    int32_t total = 0;
    for (int32_t i = 0; i < num_pdfs; ++i) {
      total += targets[i];
    }
    EXPECT_EQ(total, 6);
  }

  // All occupancies zero: nothing can be split, every target stays at 1
  // (the implementation warns and stops).
  {
    int32_t num_pdfs = 3;
    FloatVector occs(num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      occs(i) = 0.0f;
    }
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 10, 0.2f, 0.01f, &targets);
    EXPECT_EQ(static_cast<int32_t>(targets.size()), num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      EXPECT_EQ(targets[i], 1);
    }
  }

  // All-equal occupancies with a target that is not a multiple of the pdf
  // count: the extra components are spread as evenly as possible.
  {
    int32_t num_pdfs = 3;
    FloatVector occs(num_pdfs);
    for (int32_t i = 0; i < num_pdfs; ++i) {
      occs(i) = 100.0f;
    }
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 7, 1.0f, 0.01f, &targets);
    EXPECT_EQ(static_cast<int32_t>(targets.size()), num_pdfs);
    int32_t total = 0;
    for (int32_t i = 0; i < num_pdfs; ++i) {
      // every pdf gets 2 or 3 components: 3 + 2 + 2 = 7, i.e. balanced
      EXPECT_GE(targets[i], 2);
      EXPECT_LE(targets[i], 3);
      total += targets[i];
    }
    EXPECT_EQ(total, 7);
  }

  // Zero-occupancy entry combined with a min-count that also blocks the
  // counted pdfs: all targets stay at 1 even though the target is large.
  {
    int32_t num_pdfs = 2;
    FloatVector occs(num_pdfs);
    occs(0) = 0.0f;
    occs(1) = 1.0f;
    std::vector<int32_t> targets;
    GetSplitTargets(occs, 10, 0.2f, 100.0f, &targets);
    EXPECT_EQ(targets[0], 1);
    EXPECT_EQ(targets[1], 1);
  }
}

}  // namespace khg
