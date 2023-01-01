// kaldi-hmm-gmm/csrc/event-map-test.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/event-map.h"

#include "gtest/gtest.h"

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
