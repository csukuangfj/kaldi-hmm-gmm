// kaldi-hmm-gmm/csrc/event-map.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
#define KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
#include <cstdint>
#include <utility>
#include <vector>

// this file is copied and modified from
// kaldi/src/tree/event-map.h

namespace khg {

// Note RE negative values: some of this code will not work if things of type
// EventValueType are negative.  In particular, TableEventMap can't be used if
// things of EventValueType are negative, and additionally TableEventMap won't
// be efficient if things of EventValueType take on extremely large values.  The
// EventKeyType can be negative though.

/// Things of type EventKeyType can take any value.  The code does not assume
/// they are contiguous. So values like -1, 1000000 and the like are acceptable.
using EventKeyType = int32_t;

/// Given current code, things of type EventValueType should generally be
/// nonnegative and in a reasonably small range (e.g. not one million), as we
/// sometimes construct vectors of the size: [largest value we saw for this
/// key].  This deficiency may be fixed in future [would require modifying
/// TableEventMap]
using EventValueType = int32_t;

/// As far as the event-map code itself is concerned, things of type
/// EventAnswerType may take any value except kNoAnswer (== -1).  However, some
/// specific uses of EventMap (e.g. in build-tree-utils.h) assume these
/// quantities are nonnegative.
using EventAnswerType = int32_t;

// It is required to be sorted and have unique keys-- i.e. functions assume this
// when called with this type.
using EventType = std::vector<std::pair<EventKeyType, EventValueType>>;

inline std::pair<EventKeyType, EventValueType> MakeEventPair(EventKeyType k,
                                                             EventValueType v) {
  return std::pair<EventKeyType, EventValueType>(k, v);
}

void WriteEventType(std::ostream &os, bool binary, const EventType &vec);
void ReadEventType(std::istream &is, bool binary, EventType *vec);

std::string EventTypeToString(
    const EventType &evec);  // so we can print events out in error messages.

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
