// kaldi-hmm-gmm/csrc/event-map.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
#define KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/log.h"

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

// so we can print events out in error messages.
std::string EventTypeToString(const EventType &evec);

// Hashing object for EventMapVector.  Works for both pointers and references.
struct EventMapVectorHash {
  // Not used in event-map.{h, cc}
  size_t operator()(const EventType &vec) const;
  size_t operator()(const EventType *ptr) const { return (*this)(*ptr); }
};
struct EventMapVectorEqual {  // Equality object for EventType pointers-- test
                              // equality of underlying vector.
  // Not used in event-map.{h, cc}
  size_t operator()(const EventType *p1, const EventType *p2) const {
    return (*p1 == *p2);
  }
};

/// A class that is capable of representing a generic mapping from
/// EventType (which is a vector of (key, value) pairs) to
/// EventAnswerType which is just an integer.  See \ref tree_internals
/// for overview.
class EventMap {
 public:
  virtual ~EventMap() = default;

  // will crash if not sorted and unique on key.
  static void Check(const EventType &event);

  static bool Lookup(const EventType &event, EventKeyType key,
                     EventValueType *ans);

  // Maps events to the answer type. input must be sorted.
  virtual bool Map(const EventType &event, EventAnswerType *ans) const = 0;

  // MultiMap maps a partially specified set of events to the set of answers it
  // might map to.  It appends these to "ans".  "ans" is
  // **not guaranteed unique at output** if the
  // tree contains duplicate answers at leaves -- you should sort & uniq
  // afterwards. e.g.: SortAndUniq(ans).
  virtual void MultiMap(const EventType &event,
                        std::vector<EventAnswerType> *ans) const = 0;

  // GetChildren() returns the EventMaps that are immediate children of this
  // EventMap (if they exist), by putting them in *out.  Useful for
  // determining the structure of the event map.
  virtual void GetChildren(std::vector<EventMap *> *out) const = 0;

  // This Copy() does a deep copy of the event map.
  // If new_leaves is nonempty when it reaches a leaf with value l s.t.
  // new_leaves[l] != NULL, it replaces it with a copy of that EventMap.  This
  // makes it possible to extend and modify It's the way we do splits of trees,
  // and clustering of trees.  Think about this carefully, because the EventMap
  // structure does not support modification of an existing tree.  Do not be
  // tempted to do this differently, because other kinds of mechanisms would get
  // very messy and unextensible. Copy() is the only mechanism to modify a tree.
  // It's similar to a kind of function composition. Copy() does not take
  // ownership of the pointers in new_leaves (it uses the Copy() function of
  // those EventMaps).
  virtual EventMap *Copy(const std::vector<EventMap *> &new_leaves) const = 0;

  EventMap *Copy() const {
    std::vector<EventMap *> new_leaves;
    return Copy(new_leaves);
  }

  // The function MapValues() is intended to be used to map phone-sets between
  // different integer representations.  For all the keys in the set
  // "keys_to_map", it will map the corresponding values using the map
  // "value_map".  Note: these values are the values in the key->value pairs of
  // the EventMap, which really correspond to phones in the usual case; they are
  // not the "answers" of the EventMap which correspond to clustered states.  In
  // case multiple values are mapped to the same value, it will try to deal with
  // it gracefully where it can, but will crash if, for example, this would
  // cause problems with the TableEventMap.  It will also crash if any values
  // used for keys in "keys_to_map" are not mapped by "value_map".  This
  // function is not currently used.
  virtual EventMap *MapValues(
      const std::unordered_set<EventKeyType> &keys_to_map,
      const std::unordered_map<EventValueType, EventValueType> &value_map)
      const = 0;

  // The function Prune() is like Copy(), except it removes parts of the tree
  // that return only -1 (it will return NULL if this EventMap returns only -1).
  // This is a mechanism to remove parts of the tree-- you would first use the
  // Copy() function with a vector of EventMap*, and for the parts you don't
  // want, you'd put a ConstantEventMap with -1; you'd then call
  // Prune() on the result.  This function is not currently used.
  virtual EventMap *Prune() const = 0;
  // child classes may override this for efficiency; here is basic
  // version.
  virtual EventAnswerType MaxResult() const {
    // returns -1 if nothing found.
    std::vector<EventAnswerType> tmp;
    EventType empty_event;
    MultiMap(empty_event, &tmp);
    if (tmp.empty()) {
      KHG_WARN << "EventMap::MaxResult(), empty result";
      return std::numeric_limits<EventAnswerType>::min();
    } else {
      return *std::max_element(tmp.begin(), tmp.end());
    }
  }

  /// Write to stream.
  virtual void Write(std::ostream &os, bool binary) = 0;

  /// a Write function that takes care of NULL pointers.
  static void Write(std::ostream &os, bool binary, EventMap *emap);
  /// a Read function that reads an arbitrary EventMap; also
  /// works for NULL pointers.
  static EventMap *Read(std::istream &is, bool binary);
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
