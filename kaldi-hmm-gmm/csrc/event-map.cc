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
  std::stringstream ss;
  EventType::const_iterator iter = evec.begin(), end = evec.end();
  std::string sep = "";
  for (; iter != end; ++iter) {
    ss << sep << iter->first << ":" << iter->second;
    sep = " ";
  }
  return ss.str();
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
  std::vector<std::pair<EventKeyType, EventValueType>>::const_iterator
      begin = event.begin(),
      end = event.end(),
      middle;  // "middle" is used as a temporary variable in the algorithm.
  // begin and sz store the current region where the first instance of
  // "value" might appear.
  // This is like this stl algorithm "lower_bound".
  size_t sz = end - begin, half;
  while (sz > 0) {
    half = sz >> 1;
    middle = begin + half;  // "end" here is now reallly the middle.
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
    // return SplitEventMap::Read(is, binary);
  } else {
    KHG_ERR << "EventMap::read, was not expecting character "
            << kaldiio::CharToString(c) << ", at file position " << is.tellg();
    return nullptr;  // suppress warning.
  }
  return nullptr;  // TODO(fangjun): remove me
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

}  // namespace khg
