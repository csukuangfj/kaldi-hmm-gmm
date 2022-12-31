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
    return NULL;
  } else if (c == 'C') {
    // return ConstantEventMap::Read(is, binary);
  } else if (c == 'T') {
    // return TableEventMap::Read(is, binary);
  } else if (c == 'S') {
    // return SplitEventMap::Read(is, binary);
  } else {
    KHG_ERR << "EventMap::read, was not expecting character "
            << kaldiio::CharToString(c) << ", at file position " << is.tellg();
    return nullptr;  // suppress warning.
  }
  return nullptr;  // TODO(fangjun): remove me
}

}  // namespace khg
