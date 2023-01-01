// kaldi-hmm-gmm/csrc/event-map.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/event-map.h"

#include <string>

#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi_native_io/csrc/io-funcs.h"
#include "kaldi_native_io/csrc/kaldi-utils.h"

namespace khg {

void WriteEventType(std::ostream &os, bool binary, const EventType &evec) {
  kaldiio::WriteToken(os, binary, "EV");
  uint32_t size = evec.size();
  kaldiio::WriteBasicType(os, binary, size);
  for (size_t i = 0; i < size; ++i) {
    kaldiio::WriteBasicType(os, binary, evec[i].first);
    kaldiio::WriteBasicType(os, binary, evec[i].second);
  }

  if (!binary) {
    os << '\n';
  }
}

void ReadEventType(std::istream &is, bool binary, EventType *evec) {
  KHG_ASSERT(evec != NULL);
  kaldiio::ExpectToken(is, binary, "EV");

  uint32_t size;
  kaldiio::ReadBasicType(is, binary, &size);
  evec->resize(size);

  for (size_t i = 0; i < size; i++) {
    kaldiio::ReadBasicType(is, binary, &((*evec)[i].first));
    kaldiio::ReadBasicType(is, binary, &((*evec)[i].second));
  }
}

std::string EventTypeToString(const EventType &evec) {
  std::ostringstream os;
  auto iter = evec.begin(), end = evec.end();
  std::string sep = "";
  for (; iter != end; ++iter) {
    os << sep << iter->first << ":" << iter->second;
    sep = " ";
  }
  return os.str();
}

size_t EventMapVectorHash::operator()(const EventType &vec) const {
  EventType::const_iterator iter = vec.begin(), end = vec.end();
  size_t ans = 0;
  constexpr size_t kPrime1 = 47087, kPrime2 = 1321;
  for (; iter != end; ++iter) {
#ifdef DEBUG  // Check names are distinct and increasing.
    EventType::const_iterator iter2 = iter;
    iter2++;
    if (iter2 != end) {
      KHG_ASSERT(iter->first < iter2->first);
    }
#endif
    ans += iter->first + kPrime1 * iter->second;
    ans *= kPrime2;
  }
  return ans;
}

// static member of EventMap.
void EventMap::Check(
    const std::vector<std::pair<EventKeyType, EventValueType>> &event) {
  // will crash if not sorted or has duplicates
  size_t sz = event.size();
  for (size_t i = 0; i + 1 < sz; ++i)
    KHG_ASSERT(event[i].first < event[i + 1].first);
}

// static member of EventMap.
bool EventMap::Lookup(const EventType &event, EventKeyType key,
                      EventValueType *ans) {
  // this assumes that the "event" array is sorted (e.g. on the KeyType value;
  // just doing std::sort will do this) and has no duplicate values with the
  // same key.  call Check() to verify this.
#ifdef DEBUG
  Check(event);
#endif
  auto begin = event.begin(), end = event.end();
  decltype(begin) middle;
  // "middle" is used as a temporary variable in the algorithm.
  // begin and sz store the current region where the first instance of
  // "value" might appear.
  // This is like this stl algorithm "lower_bound".
  size_t sz = end - begin, half;
  while (sz > 0) {
    half = sz >> 1;
    middle = begin + half;  // "end" here is now really the middle.
    if (middle->first < key) {
      begin = middle;
      ++begin;
      sz = sz - half - 1;
    } else {
      sz = half;
    }
  }
  if (begin != end && begin->first == key) {
    *ans = begin->second;
    return true;
  } else {
    return false;
  }
}

void EventMap::Write(std::ostream &os, bool binary, EventMap *emap) {
  if (emap == NULL) {
    kaldiio::WriteToken(os, binary, "NULL");
  } else {
    emap->Write(os, binary);
  }
}

EventMap *EventMap::Read(std::istream &is, bool binary) {
  char c = kaldiio::Peek(is, binary);
  if (c == 'N') {
    kaldiio::ExpectToken(is, binary, "NULL");
    return nullptr;
  } else if (c == 'C') {
    return ConstantEventMap::Read(is, binary);
  } else if (c == 'T') {
    return TableEventMap::Read(is, binary);
  } else if (c == 'S') {
    return SplitEventMap::Read(is, binary);
  } else {
    KHG_ERR << "EventMap::read, was not expecting character "
            << kaldiio::CharToString(c) << ", at file position " << is.tellg();
    return nullptr;  // suppress warning.
  }
}

void ConstantEventMap::Write(std::ostream &os, bool binary) {
  kaldiio::WriteToken(os, binary, "CE");
  kaldiio::WriteBasicType(os, binary, answer_);
  if (os.fail()) {
    KHG_ERR << "ConstantEventMap::Write(), could not write to stream.";
  }
}

// static member function.
ConstantEventMap *ConstantEventMap::Read(std::istream &is, bool binary) {
  kaldiio::ExpectToken(is, binary, "CE");
  EventAnswerType answer;
  kaldiio::ReadBasicType(is, binary, &answer);
  return new ConstantEventMap(answer);
}

EventMap *TableEventMap::Prune() const {
  std::vector<EventMap *> table;
  table.reserve(table_.size());
  EventValueType size = table_.size();
  for (EventKeyType value = 0; value < size; ++value) {
    if (table_[value] != nullptr) {
      EventMap *pruned_map = table_[value]->Prune();
      if (pruned_map != nullptr) {
        table.resize(value + 1, nullptr);
        table[value] = pruned_map;
      }
    }
  }
  if (table.empty()) {
    return nullptr;
  } else {
    return new TableEventMap(key_, table);
  }
}

EventMap *TableEventMap::MapValues(
    const std::unordered_set<EventKeyType> &keys_to_map,
    const std::unordered_map<EventValueType, EventValueType> &value_map) const {
  std::vector<EventMap *> table;
  table.reserve(table_.size());
  EventValueType size = table_.size();

  for (EventValueType value = 0; value < size; ++value) {
    if (table_[value] != nullptr) {
      EventMap *this_map = table_[value]->MapValues(keys_to_map, value_map);
      EventValueType mapped_value;

      if (keys_to_map.count(key_) == 0) {
        mapped_value = value;
      } else {
        auto iter = value_map.find(value);
        if (iter == value_map.end()) {
          KHG_ERR << "Could not map value " << value << " for key " << key_;
        }
        mapped_value = iter->second;
      }
      KHG_ASSERT(mapped_value >= 0);

      if (static_cast<EventValueType>(table.size()) <= mapped_value) {
        table.resize(mapped_value + 1, nullptr);
      }

      if (table[mapped_value] != nullptr) {
        KHG_ERR << "Multiple values map to the same point: this code cannot "
                << "handle this case.";
      }
      table[mapped_value] = this_map;
    }
  }
  return new TableEventMap(key_, table);
}

void TableEventMap::Write(std::ostream &os, bool binary) {
  kaldiio::WriteToken(os, binary, "TE");
  kaldiio::WriteBasicType(os, binary, key_);
  uint32_t size = table_.size();

  kaldiio::WriteBasicType(os, binary, size);
  kaldiio::WriteToken(os, binary, "(");

  for (size_t t = 0; t < size; ++t) {
    // This Write function works for NULL pointers.
    EventMap::Write(os, binary, table_[t]);
  }

  kaldiio::WriteToken(os, binary, ")");

  if (!binary) {
    os << '\n';
  }

  if (os.fail()) {
    KHG_ERR << "TableEventMap::Write(), could not write to stream.";
  }
}

// static member function.
TableEventMap *TableEventMap::Read(std::istream &is, bool binary) {
  kaldiio::ExpectToken(is, binary, "TE");

  EventKeyType key;
  kaldiio::ReadBasicType(is, binary, &key);

  uint32_t size;
  kaldiio::ReadBasicType(is, binary, &size);

  std::vector<EventMap *> table(size);
  kaldiio::ExpectToken(is, binary, "(");
  for (size_t t = 0; t < size; t++) {
    // This Read function works for NULL pointers.
    table[t] = EventMap::Read(is, binary);
  }
  kaldiio::ExpectToken(is, binary, ")");
  return new TableEventMap(key, table);
}

TableEventMap::TableEventMap(EventKeyType key,
                             const std::map<EventValueType, EventMap *> &map_in)
    : key_(key) {
  if (map_in.size() == 0) {
    return;  // empty table.
  } else {
    EventValueType highest_val = map_in.rbegin()->first;
    table_.resize(highest_val + 1, nullptr);
    auto iter = map_in.begin(), end = map_in.end();

    for (; iter != end; ++iter) {
      KHG_ASSERT(iter->first >= 0 && iter->first <= highest_val);

      table_[iter->first] = iter->second;
    }
  }
}

TableEventMap::TableEventMap(
    EventKeyType key, const std::map<EventValueType, EventAnswerType> &map_in)
    : key_(key) {
  if (map_in.size() == 0) {
    return;  // empty table.
  } else {
    EventValueType highest_val = map_in.rbegin()->first;

    table_.resize(highest_val + 1, nullptr);
    auto iter = map_in.begin(), end = map_in.end();

    for (; iter != end; ++iter) {
      KHG_ASSERT(iter->first >= 0 && iter->first <= highest_val);
      table_[iter->first] = new ConstantEventMap(iter->second);
    }
  }
}

EventMap *SplitEventMap::Prune() const {
  EventMap *yes = yes_->Prune(), *no = no_->Prune();
  if (yes == nullptr && no == nullptr) {
    return nullptr;
  } else if (yes == nullptr) {
    return no;
  } else if (no == nullptr) {
    return yes;
  } else {
    return new SplitEventMap(key_, yes_set_, yes, no);
  }
}

EventMap *SplitEventMap::MapValues(
    const std::unordered_set<EventKeyType> &keys_to_map,
    const std::unordered_map<EventValueType, EventValueType> &value_map) const {
  EventMap *yes = yes_->MapValues(keys_to_map, value_map),
           *no = no_->MapValues(keys_to_map, value_map);

  if (keys_to_map.count(key_) == 0) {
    return new SplitEventMap(key_, yes_set_, yes, no);
  } else {
    std::vector<EventValueType> yes_set;
    for (auto iter = yes_set_.begin(); iter != yes_set_.end(); ++iter) {
      EventValueType value = *iter;
      auto map_iter = value_map.find(value);
      if (map_iter == value_map.end()) {
        KHG_ERR << "Value " << value << ", for key " << key_
                << ", cannot be mapped.";
      }

      EventValueType mapped_value = map_iter->second;
      yes_set.push_back(mapped_value);
    }
    SortAndUniq(&yes_set);
    return new SplitEventMap(key_, yes_set, yes, no);
  }
}

void SplitEventMap::Write(std::ostream &os, bool binary) {
  kaldiio::WriteToken(os, binary, "SE");
  kaldiio::WriteBasicType(os, binary, key_);
  // WriteIntegerVector(os, binary, yes_set_);
  yes_set_.Write(os, binary);
  KHG_ASSERT(yes_ != NULL && no_ != NULL);
  kaldiio::WriteToken(os, binary, "{");
  yes_->Write(os, binary);
  no_->Write(os, binary);
  kaldiio::WriteToken(os, binary, "}");

  if (!binary) {
    os << '\n';
  }

  if (os.fail()) {
    KHG_ERR << "SplitEventMap::Write(), could not write to stream.";
  }
}

// static member function.
SplitEventMap *SplitEventMap::Read(std::istream &is, bool binary) {
  kaldiio::ExpectToken(is, binary, "SE");
  EventKeyType key;
  kaldiio::ReadBasicType(is, binary, &key);
  // std::vector<EventValueType> yes_set;
  // ReadIntegerVector(is, binary, &yes_set);
  ConstIntegerSet<EventValueType> yes_set;
  yes_set.Read(is, binary);
  kaldiio::ExpectToken(is, binary, "{");
  EventMap *yes = EventMap::Read(is, binary);
  EventMap *no = EventMap::Read(is, binary);
  kaldiio::ExpectToken(is, binary, "}");
  // yes and no should be non-NULL because NULL values are not valid for
  // SplitEventMap; the constructor checks this.  Therefore this is an unlikely
  // error.
  if (yes == nullptr || no == nullptr) {
    KHG_ERR << "SplitEventMap::Read, NULL pointers.";
  }

  return new SplitEventMap(key, yes_set, yes, no);
}

// This function is only used inside this .cc file so make it static.
static bool IsLeafNode(const EventMap *e) {
  std::vector<EventMap *> children;
  e->GetChildren(&children);
  return children.empty();
}

// This helper function called from GetTreeStructure outputs the tree structure
// of the EventMap in a more convenient form.  At input, the objects pointed to
// by last three pointers should be empty.  The function will return false if
// the EventMap "map" doesn't have the required structure (see the comments in
// the header for GetTreeStructure).  If it returns true, then at output,
// "nonleaf_nodes" will be a vector of pointers to the EventMap* values
// corresponding to nonleaf nodes, in an order where the root node comes first
// and child nodes are after their parents; "nonleaf_parents" will be a map
// from each nonleaf node to its parent, and the root node points to itself;
// and "leaf_parents" will be a map from the numeric id of each leaf node
// (corresponding to the value returned by the EventMap) to its parent node;
// leaf_parents will contain no NULL pointers, otherwise we would have returned
// false as the EventMap would not have had the required structure.

static bool GetTreeStructureInternal(
    const EventMap &map, std::vector<const EventMap *> *nonleaf_nodes,
    std::map<const EventMap *, const EventMap *> *nonleaf_parents,
    std::vector<const EventMap *> *leaf_parents) {
  std::vector<const EventMap *> queue;  // parents to be processed.

  const EventMap *top_node = &map;

  queue.push_back(top_node);
  nonleaf_nodes->push_back(top_node);
  (*nonleaf_parents)[top_node] = top_node;

  while (!queue.empty()) {
    const EventMap *parent = queue.back();
    queue.pop_back();
    std::vector<EventMap *> children;
    parent->GetChildren(&children);
    KHG_ASSERT(!children.empty());
    for (size_t i = 0; i < children.size(); i++) {
      EventMap *child = children[i];
      if (IsLeafNode(child)) {
        int32_t leaf;
        if (!child->Map(EventType(), &leaf) || leaf < 0) return false;
        if (static_cast<int32_t>(leaf_parents->size()) <= leaf)
          leaf_parents->resize(leaf + 1, NULL);
        if ((*leaf_parents)[leaf] != NULL) {
          KHG_WARN << "Repeated leaf! Did you suppress leaf clustering when "
                      "building the tree?";
          return false;  // repeated leaf.
        }
        (*leaf_parents)[leaf] = parent;
      } else {
        nonleaf_nodes->push_back(child);
        (*nonleaf_parents)[child] = parent;
        queue.push_back(child);
      }
    }
  }

  for (size_t i = 0; i < leaf_parents->size(); i++)
    if ((*leaf_parents)[i] == NULL) {
      KHG_WARN << "non-consecutively numbered leaves";
      return false;
    }
  // non-consecutively numbered leaves.

  KHG_ASSERT(!leaf_parents->empty());  // or no leaves.

  return true;
}

// See the header for a description of what this function does.
bool GetTreeStructure(const EventMap &map, int32_t *num_leaves,
                      std::vector<int32_t> *parents) {
  KHG_ASSERT(num_leaves != NULL && parents != NULL);

  if (IsLeafNode(&map)) {  // handle degenerate case where root is a leaf.
    int32_t leaf;
    if (!map.Map(EventType(), &leaf) || leaf != 0) return false;
    *num_leaves = 1;
    parents->resize(1);
    (*parents)[0] = 0;
    return true;
  }

  // This vector gives the address of nonleaf nodes in the tree,
  // in a numbering where 0 is the root and children always come
  // after parents.
  std::vector<const EventMap *> nonleaf_nodes;

  // Map from each nonleaf node to its parent node
  // (or to itself for the root node).
  std::map<const EventMap *, const EventMap *> nonleaf_parents;

  // Map from leaf nodes to their parent nodes.
  std::vector<const EventMap *> leaf_parents;

  if (!GetTreeStructureInternal(map, &nonleaf_nodes, &nonleaf_parents,
                                &leaf_parents))
    return false;

  *num_leaves = leaf_parents.size();
  int32_t num_nodes = leaf_parents.size() + nonleaf_nodes.size();

  std::map<const EventMap *, int32_t> nonleaf_indices;

  // number the nonleaf indices so they come after the leaf
  // indices and the root is last.
  for (size_t i = 0; i < nonleaf_nodes.size(); i++)
    nonleaf_indices[nonleaf_nodes[i]] = num_nodes - i - 1;

  parents->resize(num_nodes);
  for (size_t i = 0; i < leaf_parents.size(); i++) {
    KHG_ASSERT(nonleaf_indices.count(leaf_parents[i]) != 0);
    (*parents)[i] = nonleaf_indices[leaf_parents[i]];
  }
  for (size_t i = 0; i < nonleaf_nodes.size(); i++) {
    KHG_ASSERT(nonleaf_indices.count(nonleaf_nodes[i]) != 0);
    KHG_ASSERT(nonleaf_parents.count(nonleaf_nodes[i]) != 0);
    KHG_ASSERT(nonleaf_indices.count(nonleaf_parents[nonleaf_nodes[i]]) != 0);
    int32_t index = nonleaf_indices[nonleaf_nodes[i]],
            parent_index = nonleaf_indices[nonleaf_parents[nonleaf_nodes[i]]];
    KHG_ASSERT(index > 0 && parent_index >= index);
    (*parents)[index] = parent_index;
  }
  for (int32_t i = 0; i < num_nodes; i++)
    KHG_ASSERT((*parents)[i] > i || (i + 1 == num_nodes && (*parents)[i] == i));
  return true;
}

}  // namespace khg
