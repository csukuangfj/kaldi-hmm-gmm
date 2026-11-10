// kaldi-hmm-gmm/csrc/event-map-test.cc
//
// Copyright 2009-2011  Microsoft Corporation;  Haihua Xu;  Yanmin Qian
//                2013  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2022  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/tree/event-map-test.cc
//

#include "kaldi-hmm-gmm/csrc/event-map.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

using KeyType = EventKeyType;
using ValueType = EventValueType;
using AnswerType = EventAnswerType;

TEST(EventMap, Case1) {
  ConstantEventMap *C0a = new ConstantEventMap(0);
  {
    int32_t num_leaves;
    std::vector<int32_t> parents;
    bool a = GetTreeStructure(*C0a, &num_leaves, &parents);
    KHG_ASSERT(a && parents.size() == 1 && parents[0] == 0);
  }

  ConstantEventMap *C1b = new ConstantEventMap(1);
  {
    int32_t num_leaves;
    std::vector<int32_t> parents;
    bool a = GetTreeStructure(*C1b, &num_leaves, &parents);
    KHG_ASSERT(!a);  // since C1b's leaves don't start from 0.
  }

  std::vector<EventMap *> tvec;
  tvec.push_back(C0a);
  tvec.push_back(C1b);

  // takes ownership of C0a, C1b
  TableEventMap *T1 = new TableEventMap(1, tvec);
  KHG_ASSERT(T1->MaxResult() == 1);

  {
    int32_t num_leaves;
    std::vector<int32_t> parents;
    bool a = GetTreeStructure(*T1, &num_leaves, &parents);
    KHG_ASSERT(a && parents.size() == 3 && parents[0] == 2 && parents[1] == 2 &&
               parents[2] == 2);
  }

  ConstantEventMap *C0c = new ConstantEventMap(0);
  ConstantEventMap *C1d = new ConstantEventMap(1);

  std::map<ValueType, EventMap *> tmap;
  tmap[0] = C0c;
  tmap[1] = C1d;

  // takes ownership of pointers C0c and C1d.
  TableEventMap *T2 = new TableEventMap(1, tmap);

  std::vector<ValueType> vec;
  vec.push_back(4);
  vec.push_back(5);

  ConstantEventMap *D1 = new ConstantEventMap(10);  // owned by D3 below
  ConstantEventMap *D2 = new ConstantEventMap(15);  // owned by D3 below
  SplitEventMap *D3 = new SplitEventMap(1, vec, D1, D2);

  // Test different initializer  for TableEventMap where input maps ints to
  // ints.
  for (size_t i = 0; i < 100; i++) {
    size_t nElems = rand() % 10;  // num of value->answer pairs. // NOLINT
    std::map<ValueType, AnswerType> init_map;
    for (size_t i = 0; i < nElems; i++) {
      init_map[rand() % 10] = rand() % 5;  // NOLINT
    }
    EventKeyType key = rand() % 10;  // NOLINT
    TableEventMap T3(key, init_map);
    for (size_t i = 0; i < 10; i++) {
      EventType vec;
      vec.push_back(std::make_pair(key, (ValueType)i));
      AnswerType ans;
      // T3.Map(vec, &ans);
      if (init_map.count(i) == 0) {
        KHG_ASSERT(!T3.Map(vec, &ans));  // false
      } else {
        bool b = T3.Map(vec, &ans);
        KHG_ASSERT(b);
        KHG_ASSERT(ans == init_map[i]);  // true
      }
    }
  }

  delete T1;
  delete T2;
  delete D3;
}

}  // namespace khg

// ---------------------------------------------------------------------------
// Ported from ../kaldi/src/tree/event-map-test.cc (gtest-ified):
// TestEventTypeIo, TestEventMapIo, TestEventMapPrune, TestEventMapMapValues.
// ---------------------------------------------------------------------------

namespace khg {

namespace {

const int32_t kMaxVal = 20;

// Knuth's Poisson sampler (kaldi's RandPoisson is not part of this repo;
// test-local).  Only used to pick random table sizes.
int32_t RandPoissonLocal(double lambda) {
  double L = std::exp(-lambda);
  int32_t k = 0;
  double p = 1.0;
  do {
    ++k;
    p *= RandUniform();
  } while (p > L);
  return k - 1;
}

// kaldi's RandomEventMap: build a random EventMap over the given keys.
EventMap *RandomEventMap(const std::vector<EventKeyType> &keys) {
  // Do not mess with the probabilities inside this routine or there
  // is a danger this function will blow up.
  int32_t max_val = kMaxVal;
  KHG_ASSERT(!keys.empty());
  float f = RandUniform();
  if (f < 0.333) {  // w.p. 0.333, return ConstantEventMap.
    return new ConstantEventMap(Rand() % max_val);
  } else if (f < 0.666) {  // w.p. 0.333, return TableEventMap.
    float nonnull_prob = 0.3;  // prob of a non-NULL pointer.
    float expected_table_size = 3.0;
    int32_t table_size = RandPoissonLocal(expected_table_size);
    // fertility from this branch is 0.333 * 3.0 * 0.2333 = 0.3.
    EventKeyType key = keys[Rand() % keys.size()];
    std::vector<EventMap *> table(table_size);
    for (int32_t t = 0; t < table_size; ++t) {
      if (RandUniform() < nonnull_prob) {
        table[t] = RandomEventMap(keys);
      } else {
        table[t] = nullptr;
      }
    }
    return new TableEventMap(key, table);
  } else {  // w.p. 0.333, return SplitEventMap.
    // Fertility of this stage is 0.333 * 2 = 0.666.
    EventKeyType key = keys[Rand() % keys.size()];
    std::set<EventValueType> yes_set;
    for (int32_t i = 0; i < 5; ++i) {
      yes_set.insert(Rand() % max_val);
    }
    std::vector<EventValueType> yes_vec(yes_set.begin(), yes_set.end());
    EventMap *yes = RandomEventMap(keys);
    EventMap *no = RandomEventMap(keys);
    return new SplitEventMap(key, yes_vec, yes, no);
  }
  // total fertility is 0.3 + 0.666 = 0.9666, hence this will terminate with
  // finite memory (w.p.1)
}

}  // namespace

// kaldi's TestEventTypeIo: WriteEventType/ReadEventType round-trip through a
// temp file (text and binary).
TEST(EventMap, EventTypeIo) {
  srand(1);
  for (int32_t i = 0; i < 2; ++i) {
    bool binary = (i == 1);
    for (int32_t p = 0; p < 20; ++p) {
      EventType event_vec;
      int32_t size = Rand() % 20;
      event_vec.resize(size);
      for (int32_t j = 0; j < size; ++j) {
        event_vec[j].first = Rand() % 10 + (j > 0 ? event_vec[j - 1].first : 0);
        event_vec[j].second = Rand() % 20;
      }

      const char *filename = "tmp_event_type_io";
      {
        kaldiio::Output ko(filename, binary);
        std::ostream &outfile = ko.Stream();
        WriteEventType(outfile, binary, event_vec);
      }
      {
        bool binary_in = false;
        kaldiio::Input ki(filename, &binary_in);
        EXPECT_EQ(binary_in, binary);
        std::istream &infile = ki.Stream();
        EventType evec2;
        // Make it nonempty to check ReadEventType clears it.
        evec2.push_back(std::pair<EventKeyType, EventValueType>(1, 1));
        ReadEventType(infile, binary_in, &evec2);
        EXPECT_TRUE(evec2 == event_vec);
      }
      std::remove(filename);
    }
  }
}

// kaldi's TestEventMapIo: write a random EventMap to a temp file, read it back
// and check the re-serialized form is identical (text and binary).
TEST(EventMap, EventMapIo) {
  srand(1);
  for (int32_t i = 0; i < 2; ++i) {
    bool binary = (i == 1);
    for (int32_t p = 0; p < 20; ++p) {
      int32_t max_key = 10;
      int32_t num_keys = 1 + (Rand() % (max_key - 1));
      std::set<EventKeyType> key_set;
      // - 5 to allow negative keys.  These are allowed.
      while (static_cast<int32_t>(key_set.size()) < num_keys) {
        key_set.insert((Rand() % (2 * max_key)) - 5);
      }
      std::vector<EventKeyType> key_vec(key_set.begin(), key_set.end());
      EventMap *rand_map = RandomEventMap(key_vec);

      const char *filename = "tmp_event_map_io";
      {
        kaldiio::Output ko(filename, binary);
        EventMap::Write(ko.Stream(), binary, rand_map);
      }
      EventMap *read_map = nullptr;
      {
        bool binary_in = false;
        kaldiio::Input ki(filename, &binary_in);
        EXPECT_EQ(binary_in, binary);
        read_map = EventMap::Read(ki.Stream(), binary_in);
      }

      std::ostringstream str_out;
      std::ostringstream str2_out;
      EventMap::Write(str_out, binary, rand_map);
      EventMap::Write(str2_out, binary, read_map);
      // Writing the map, reading it in and writing it again must give the
      // same string form.
      EXPECT_EQ(str_out.str(), str2_out.str());

      delete read_map;
      delete rand_map;
      std::remove(filename);
    }
  }
}

// kaldi's TestEventMapPrune: leaves mapped to -1 may (but need not) be pruned
// away; other answers must survive with the mapped value.
TEST(EventMap, Prune) {
  srand(1);
  const EventAnswerType no_ans = -10;
  for (int32_t rep = 0; rep < 10; ++rep) {
    std::vector<EventKeyType> keys;
    keys.push_back(1);  // these keys are
    keys.push_back(2);  // hardwired into the code below, do not change
    EventMap *em = RandomEventMap(keys);
    EventType empty_event;
    std::vector<EventAnswerType> all_answers;
    em->MultiMap(empty_event, &all_answers);
    SortAndUniq(&all_answers);
    std::vector<EventMap *> new_leaves;
    std::vector<EventAnswerType> mapping;
    for (size_t i = 0; i < all_answers.size(); ++i) {
      EventAnswerType ans = all_answers[i];
      KHG_ASSERT(ans >= 0);
      new_leaves.resize(ans + 1, nullptr);
      mapping.resize(ans + 1, no_ans);
      EventAnswerType map_to;
      if (Rand() % 2 == 0) {
        map_to = -1;
      } else {
        map_to = Rand() % 20;
      }
      new_leaves[ans] = new ConstantEventMap(map_to);
      mapping[ans] = map_to;
    }
    EventMap *mapped_em = em->Copy(new_leaves);
    EventMap *pruned_em = mapped_em->Prune();
    for (size_t i = 0; i < new_leaves.size(); ++i) {
      delete new_leaves[i];
    }
    for (int32_t i = 0; i < 10; ++i) {
      EventType event;
      for (int32_t key = 1; key <= 2; ++key) {
        if (Rand() % 2 == 0) {
          EventValueType value = Rand() % 20;
          event.push_back(std::make_pair(key, value));
        }
      }
      EventAnswerType answer = 0;
      EventAnswerType answer2 = 0;
      if (em->Map(event, &answer)) {
        bool ret;
        if (pruned_em == nullptr) {
          ret = false;
        } else {
          ret = pruned_em->Map(event, &answer2);
        }
        KHG_ASSERT(answer >= 0);
        EventAnswerType mapped_ans = mapping[answer];
        KHG_ASSERT(mapped_ans != no_ans);
        if (mapped_ans == -1) {
          // It is allowed (but not required) that this answer was pruned away.
          if (ret) {
            KHG_LOG << "Answer was not pruned away [but this is not required]";
          }
        } else {
          EXPECT_TRUE(ret);
          EXPECT_EQ(answer2, mapped_ans);
        }
      }
    }
    delete em;
    delete mapped_em;
    delete pruned_em;
  }
}

// kaldi's TestEventMapMapValues: mapping the values of some keys must not
// change the answers of the event map.
TEST(EventMap, MapValues) {
  srand(1);
  for (int32_t rep = 0; rep < 10; ++rep) {
    std::vector<EventKeyType> keys;
    keys.push_back(1);  // these keys are
    keys.push_back(2);  // hardwired into the code below, do not change
    EventMap *em = RandomEventMap(keys);

    std::unordered_set<EventKeyType> mapped_keys;
    std::unordered_map<EventValueType, EventValueType> value_map;
    if (Rand() % 2 == 0) mapped_keys.insert(1);
    if (Rand() % 2 == 0) mapped_keys.insert(2);

    EventValueType v_offset = Rand() % kMaxVal;
    for (EventValueType v = 0; v < kMaxVal; ++v) {
      value_map[v] = (v + v_offset) % kMaxVal;
    }

    EventMap *mapped_em = em->MapValues(mapped_keys, value_map);

    for (int32_t i = 0; i < 10; ++i) {
      EventType event;
      EventType mapped_event;
      for (int32_t key = 1; key <= 2; ++key) {
        if (Rand() % 2 == 0) {
          EventValueType value = Rand() % kMaxVal;
          event.push_back(std::make_pair(key, value));
          EventValueType mapped_value;
          if (mapped_keys.count(key) == 0) {
            mapped_value = value;
          } else {
            mapped_value = value_map[value];
          }
          mapped_event.push_back(std::make_pair(key, mapped_value));
        }
      }
      EventAnswerType answer = 0;
      EventAnswerType answer2 = 0;
      if (em->Map(event, &answer)) {
        bool ret = mapped_em->Map(mapped_event, &answer2);
        EXPECT_TRUE(ret);
        EXPECT_EQ(answer, answer2);
      }
    }
    delete em;
    delete mapped_em;
  }
}

}  // namespace khg

// ---------------------------------------------------------------------------
// kaldi's EventMap::MultiMap semantics (kaldi/src/tree/event-map.h): a
// partially specified event maps to the set of answers it might reach; when
// the split key is absent from the event, all branches contribute.  The
// output vector is appended to and is not guaranteed unique.
// ---------------------------------------------------------------------------

namespace khg {

TEST(EventMap, MultiMap) {
  srand(1);

  // TableEventMap over one key with several values: MultiMap without the
  // key returns all leaves.
  std::map<ValueType, AnswerType> table_in;
  table_in[0] = 0;
  table_in[1] = 10;
  table_in[2] = 20;
  TableEventMap table(1, table_in);  // one ConstantEventMap per value

  {
    EventType empty_event;
    std::vector<EventAnswerType> ans;
    table.MultiMap(empty_event, &ans);
    SortAndUniq(&ans);
    std::vector<EventAnswerType> expected;
    expected.push_back(0);
    expected.push_back(10);
    expected.push_back(20);
    EXPECT_TRUE(ans == expected);
  }

  // specifying the key restricts the result to that one leaf
  {
    EventType event;
    event.push_back(MakeEventPair(1, 1));
    std::vector<EventAnswerType> ans;
    table.MultiMap(event, &ans);
    std::vector<EventAnswerType> expected;
    expected.push_back(10);
    EXPECT_TRUE(ans == expected);
  }

  // a key the table does not split on is simply absent: all leaves again
  {
    EventType event;
    event.push_back(MakeEventPair(2, 0));
    std::vector<EventAnswerType> ans;
    table.MultiMap(event, &ans);
    SortAndUniq(&ans);
    std::vector<EventAnswerType> expected;
    expected.push_back(0);
    expected.push_back(10);
    expected.push_back(20);
    EXPECT_TRUE(ans == expected);
  }

  // the table's key is present but the value is out of range: no answers
  {
    EventType event;
    event.push_back(MakeEventPair(1, 7));
    std::vector<EventAnswerType> ans;
    table.MultiMap(event, &ans);
    EXPECT_TRUE(ans.empty());
  }

  // answers are appended to what the vector already holds and need not be
  // unique (kaldi documents: sort & uniq afterwards)
  {
    std::map<ValueType, AnswerType> dup_in;
    dup_in[0] = 5;
    dup_in[1] = 5;
    TableEventMap dup_table(3, dup_in);
    EventType empty_event;
    std::vector<EventAnswerType> ans;
    ans.push_back(99);
    dup_table.MultiMap(empty_event, &ans);
    EXPECT_EQ(static_cast<int32_t>(ans.size()), 3);
    EXPECT_EQ(ans[0], 99);
  }

  // SplitEventMap: an absent key sends both branches, a present key picks
  // exactly one.
  {
    std::vector<ValueType> yes_set;
    yes_set.push_back(1);
    EventMap *yes = new ConstantEventMap(7);
    EventMap *no = new ConstantEventMap(8);
    SplitEventMap split(2, yes_set, yes, no);  // takes ownership

    EventType empty_event;
    std::vector<EventAnswerType> ans;
    split.MultiMap(empty_event, &ans);
    SortAndUniq(&ans);
    EXPECT_EQ(static_cast<int32_t>(ans.size()), 2);
    EXPECT_EQ(ans[0], 7);
    EXPECT_EQ(ans[1], 8);

    EventType yes_event;
    yes_event.push_back(MakeEventPair(2, 1));
    std::vector<EventAnswerType> ans_yes;
    split.MultiMap(yes_event, &ans_yes);
    EXPECT_EQ(static_cast<int32_t>(ans_yes.size()), 1);
    EXPECT_EQ(ans_yes[0], 7);

    EventType no_event;
    no_event.push_back(MakeEventPair(2, 4));
    std::vector<EventAnswerType> ans_no;
    split.MultiMap(no_event, &ans_no);
    EXPECT_EQ(static_cast<int32_t>(ans_no.size()), 1);
    EXPECT_EQ(ans_no[0], 8);
  }
}

}  // namespace khg
