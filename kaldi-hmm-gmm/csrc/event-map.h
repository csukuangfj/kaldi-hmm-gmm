// kaldi-hmm-gmm/csrc/event-map.h
//
// Copyright 2009-2011  Microsoft Corporation;  Haihua Xu
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
#define KALDI_HMM_GMM_CSRC_EVENT_MAP_H_

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/const-integer-set.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"

/**
 * \file event-map.h
 * \brief Generic mapping from integer events to integer answers (decision
 *        trees).
 *
 * This file is copied/modified from kaldi/src/tree/event-map.h.
 *
 * An \c EventMap represents a mapping from an \c EventType -- a sorted vector
 * of (key, value) integer pairs, i.e. one "event" -- to an integer answer
 * (\c EventAnswerType), e.g. a pdf-id.  It is the data structure underlying
 * Kaldi's decision trees: the leaves hold answers and the internal nodes
 * (\c SplitEventMap, \c TableEventMap) ask questions about individual event
 * keys.  See kaldi's \c tree_internals documentation for an overview.
 *
 * The file provides:
 * - The event types: \c EventKeyType, \c EventValueType, \c EventAnswerType,
 *   \c EventType, and small helpers (\c MakeEventPair, \c WriteEventType,
 *   \c ReadEventType, \c EventTypeToString).
 * - The abstract base class \c EventMap and its three node kinds:
 *   \c ConstantEventMap (leaf), \c TableEventMap (multi-way split on a key),
 *   \c SplitEventMap (binary yes/no split on a key).
 * - \c GetTreeStructure, which extracts the tree topology of an \c EventMap.
 *
 * Building trees from statistics is done in build-tree-utils.h /
 * cluster-utils.h; using them for context dependency is done in
 * context-dep.h.
 */

namespace khg {

/**
 * \brief The key of an event; may take any integer value.
 *
 * Things of type EventKeyType can take any value.  The code does not assume
 * they are contiguous. So values like -1, 1000000 and the like are acceptable.
 */
using EventKeyType = int32_t;

/**
 * \brief The value of an event for a given key; should be nonnegative and in
 *        a reasonably small range.
 *
 * Given current code, things of type EventValueType should generally be
 * nonnegative and in a reasonably small range (e.g. not one million), as we
 * sometimes construct vectors of the size: [largest value we saw for this
 * key].  This deficiency may be fixed in future [would require modifying
 * TableEventMap].
 *
 * Note RE negative values: some of this code will not work if things of type
 * EventValueType are negative.  In particular, TableEventMap can't be used if
 * things of EventValueType are negative, and additionally TableEventMap won't
 * be efficient if things of EventValueType take on extremely large values.
 * The EventKeyType can be negative though.
 */
using EventValueType = int32_t;

/**
 * \brief The answer (output) value of an EventMap, e.g. a pdf-id.
 *
 * As far as the event-map code itself is concerned, things of type
 * EventAnswerType may take any value except kNoAnswer (== -1).  However, some
 * specific uses of EventMap (e.g. in build-tree-utils.h) assume these
 * quantities are nonnegative.
 */
using EventAnswerType = int32_t;

/// One event: a list of (key, value) pairs.  It is required to be sorted and
/// have unique keys-- i.e. functions assume this when called with this type.
using EventType = std::vector<std::pair<EventKeyType, EventValueType>>;

/**
 * \brief Convenience helper that constructs a (key, value) pair.
 *
 * \param k  The event key.
 * \param v  The event value.
 * \return The pair <tt>(k, v)</tt>.
 */
inline std::pair<EventKeyType, EventValueType> MakeEventPair(EventKeyType k,
                                                             EventValueType v) {
  return std::pair<EventKeyType, EventValueType>(k, v);
}

/**
 * \brief Writes an \c EventType to a stream.
 *
 * \param os      The stream to write to.
 * \param binary  Whether to write in binary (\c true) or text (\c false) form.
 * \param vec     The event to write.
 */
void WriteEventType(std::ostream &os, bool binary, const EventType &vec);

/**
 * \brief Reads an \c EventType from a stream written by \c WriteEventType.
 *
 * \param is      The stream to read from.
 * \param binary  Whether the stream is in binary (\c true) or text
 *                (\c false) form.
 * \param vec     [out] The event that is read.
 */
void ReadEventType(std::istream &is, bool binary, EventType *vec);

/// Returns a string representation of an event, so we can print events out in
/// error messages.
std::string EventTypeToString(const EventType &evec);

/// Hashing object for EventMapVector.  Works for both pointers and references.
struct EventMapVectorHash {
  // Not used in event-map.{h, cc}
  size_t operator()(const EventType &vec) const;
  size_t operator()(const EventType *ptr) const { return (*this)(*ptr); }
};
/// Equality object for EventType pointers-- test equality of underlying
/// vector.
struct EventMapVectorEqual {
  // Not used in event-map.{h, cc}
  size_t operator()(const EventType *p1, const EventType *p2) const {
    return (*p1 == *p2);
  }
};

/**
 * \brief A class that is capable of representing a generic mapping from
 *        EventType (which is a vector of (key, value) pairs) to
 *        EventAnswerType which is just an integer.
 *
 * See kaldi's \c tree_internals documentation for an overview.
 */
class EventMap {
 public:
  virtual ~EventMap() = default;

  /**
   * \brief Checks that an event is sorted and has unique keys.
   *
   * Will crash if not sorted and unique on key.
   *
   * \param event  The event to check.
   */
  static void Check(const EventType &event);

  /**
   * \brief Looks up the value of one key within an event.
   *
   * \param event  The event to look in; must be sorted and unique on key.
   * \param key    The key to look up.
   * \param ans    [out] The value corresponding to \c key, if present.
   * \return  \c true if the key was found in the event.
   */
  static bool Lookup(const EventType &event, EventKeyType key,
                     EventValueType *ans);

  /**
   * \brief Maps an event to the answer type.
   *
   * Maps events to the answer type; input must be sorted.
   *
   * \param event  The (fully or partially specified) event; must be sorted.
   * \param ans    [out] The answer this event maps to.
   * \return  \c true on success; \c false if the event does not determine an
   *          answer.
   */
  virtual bool Map(const EventType &event, EventAnswerType *ans) const = 0;

  /**
   * \brief Maps a partially specified event to the set of answers it might
   *        map to.
   *
   * MultiMap maps a partially specified set of events to the set of answers it
   * might map to.  It appends these to "ans".  "ans" is
   * **not guaranteed unique at output** if the
   * tree contains duplicate answers at leaves -- you should sort & uniq
   * afterwards. e.g.: SortAndUniq(ans).
   *
   * \param event  The (possibly partially specified) event; must be sorted.
   * \param ans    [out] The possible answers are appended to this vector (not
   *               guaranteed unique at output).
   */
  virtual void MultiMap(const EventType &event,
                        std::vector<EventAnswerType> *ans) const = 0;

  /**
   * \brief Returns the immediate children of this EventMap (if they exist).
   *
   * GetChildren() returns the EventMaps that are immediate children of this
   * EventMap (if they exist), by putting them in *out.  Useful for
   * determining the structure of the event map.
   *
   * The returned pointer in out are borrowed.
   *
   * \param out  [out] Cleared, then filled with the borrowed pointers to the
   *            immediate children.
   */
  virtual void GetChildren(std::vector<EventMap *> *out) const = 0;

  /**
   * \brief Returns a deep copy of this EventMap, optionally replacing leaves.
   *
   * This Copy() does a deep copy of the event map.
   * If new_leaves is nonempty when it reaches a leaf with value l s.t.
   * new_leaves[l] != NULL, it replaces it with a copy of that EventMap.  This
   * makes it possible to extend and modify It's the way we do splits of trees,
   * and clustering of trees.  Think about this carefully, because the EventMap
   * structure does not support modification of an existing tree.  Do not be
   * tempted to do this differently, because other kinds of mechanisms would get
   * very messy and unextensible. Copy() is the only mechanism to modify a tree.
   * It's similar to a kind of function composition. Copy() does not take
   * ownership of the pointers in new_leaves (it uses the Copy() function of
   * those EventMaps).
   *
   * \param new_leaves  If nonempty, leaf value \c l is replaced by a copy of
   *                    \c new_leaves[l] where that pointer is non-NULL.  Not
   *                    owned by the result.
   * \return  A newly allocated deep copy; the caller owns it.
   */
  virtual EventMap *Copy(const std::vector<EventMap *> &new_leaves) const = 0;

  /**
   * \brief Returns a deep copy of this EventMap with its leaves unchanged.
   *
   * \return  A newly allocated deep copy; the caller owns it.
   */
  EventMap *Copy() const {
    std::vector<EventMap *> new_leaves;
    return Copy(new_leaves);
  }

  /**
   * \brief Maps the event values (not the answers) of selected keys.
   *
   * The function MapValues() is intended to be used to map phone-sets between
   * different integer representations.  For all the keys in the set
   * "keys_to_map", it will map the corresponding values using the map
   * "value_map".  Note: these values are the values in the key->value pairs of
   * the EventMap, which really correspond to phones in the usual case; they are
   * not the "answers" of the EventMap which correspond to clustered states.  In
   * case multiple values are mapped to the same value, it will try to deal with
   * it gracefully where it can, but will crash if, for example, this would
   * cause problems with the TableEventMap.  It will also crash if any values
   * used for keys in "keys_to_map" are not mapped by "value_map".  This
   * function is not currently used.
   *
   * \param keys_to_map  The set of keys whose values should be remapped.
   * \param value_map    The map from old values to new values.
   * \return  A newly allocated remapped EventMap; the caller owns it.
   */
  virtual EventMap *MapValues(
      const std::unordered_set<EventKeyType> &keys_to_map,
      const std::unordered_map<EventValueType, EventValueType> &value_map)
      const = 0;

  /**
   * \brief Returns a copy of this EventMap with the parts answering only -1
   *        removed.
   *
   * The function Prune() is like Copy(), except it removes parts of the tree
   * that return only -1 (it will return NULL if this EventMap returns only -1).
   * This is a mechanism to remove parts of the tree-- you would first use the
   * Copy() function with a vector of EventMap*, and for the parts you don't
   * want, you'd put a ConstantEventMap with -1; you'd then call
   * Prune() on the result.  This function is not currently used.
   *
   * \return  A newly allocated pruned EventMap, or NULL if this EventMap
   *          returns only -1; the caller owns it.
   */
  virtual EventMap *Prune() const = 0;

  /**
   * \brief Returns the largest answer value occurring in this EventMap.
   *
   * Child classes may override this for efficiency; here is the basic version.
   *
   * \return  The largest answer value, or a minimal (large-negative) value if
   *          nothing was found (returns -1 if nothing found).
   */
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

  /**
   * \brief Writes this EventMap to a stream.
   *
   * \param os      The stream to write to.
   * \param binary  Whether to write in binary (\c true) or text (\c false)
   *                form.
   */
  virtual void Write(std::ostream &os, bool binary) = 0;

  /**
   * \brief Writes an EventMap to a stream; also works for NULL pointers.
   *
   * A Write function that takes care of NULL pointers.
   *
   * \param os      The stream to write to.
   * \param binary  Whether to write in binary (\c true) or text (\c false)
   *                form.
   * \param emap    The EventMap to write; may be NULL.
   */
  static void Write(std::ostream &os, bool binary, EventMap *emap);

  /**
   * \brief Reads an arbitrary EventMap from a stream; also works for NULL
   *        pointers.
   *
   * \param is      The stream to read from.
   * \param binary  Whether the stream is in binary (\c true) or text
   *                (\c false) form.
   * \return  The EventMap that was read (may be NULL); the caller owns it.
   */
  static EventMap *Read(std::istream &is, bool binary);
};

/**
 * \brief A leaf of an EventMap: returns a constant answer for any event.
 */
class ConstantEventMap : public EventMap {
 public:
  ConstantEventMap(const ConstantEventMap &) = delete;
  ConstantEventMap &operator=(const ConstantEventMap &) = delete;

  bool Map(const EventType &event, EventAnswerType *ans) const override {
    *ans = answer_;
    return true;
  }

  void MultiMap(const EventType &,
                std::vector<EventAnswerType> *ans) const override {
    ans->push_back(answer_);
  }

  void GetChildren(std::vector<EventMap *> *out) const override {
    out->clear();
  }

  EventMap *Copy(const std::vector<EventMap *> &new_leaves) const override {
    if (answer_ < 0 || answer_ >= (EventAnswerType)new_leaves.size() ||
        new_leaves[answer_] == nullptr) {
      return new ConstantEventMap(answer_);
    } else {
      return new_leaves[answer_]->Copy();
    }
  }

  EventMap *MapValues(const std::unordered_set<EventKeyType> &keys_to_map,
                      const std::unordered_map<EventValueType, EventValueType>
                          &value_map) const override {
    return new ConstantEventMap(answer_);
  }

  EventMap *Prune() const override {
    return (answer_ == -1 ? nullptr : new ConstantEventMap(answer_));
  }

  /**
   * \brief Constructor.
   *
   * \param answer  The constant answer this leaf returns for any event.
   */
  explicit ConstantEventMap(EventAnswerType answer) : answer_(answer) {}

  void Write(std::ostream &os, bool binary) override;
  static ConstantEventMap *Read(std::istream &is, bool binary);

 private:
  EventAnswerType answer_;
};

/**
 * \brief An EventMap node that splits on the value of a single key by looking
 *        up a table of children.
 *
 * The event is queried for \c key_; the value found is used as an index into
 * the table of child EventMaps, which are recursed into.  EventValueType
 * values must be nonnegative and not too large for this node to be usable.
 */
class TableEventMap : public EventMap {
 public:
  TableEventMap(const TableEventMap &) = delete;
  TableEventMap &operator=(const TableEventMap &) = delete;

  bool Map(const EventType &event, EventAnswerType *ans) const override {
    EventValueType tmp;

    *ans = -1;  // means no answer

    if (Lookup(event, key_, &tmp) && tmp >= 0 &&
        tmp < (EventValueType)table_.size() && table_[tmp] != nullptr) {
      return table_[tmp]->Map(event, ans);
    }
    return false;
  }

  void GetChildren(std::vector<EventMap *> *out) const override {
    out->clear();
    for (size_t i = 0; i < table_.size(); ++i) {
      if (table_[i] != nullptr) {
        out->push_back(table_[i]);
      }
    }
  }

  void MultiMap(const EventType &event,
                std::vector<EventAnswerType> *ans) const override {
    EventValueType tmp;
    if (Lookup(event, key_, &tmp)) {
      if (tmp >= 0 && tmp < (EventValueType)table_.size() &&
          table_[tmp] != nullptr)
        return table_[tmp]->MultiMap(event, ans);
      // else no answers.
    } else {  // all answers are possible if no such key.
      for (size_t i = 0; i < table_.size(); ++i) {
        if (table_[i] != nullptr) {
          // append.
          table_[i]->MultiMap(event, ans);
        }
      }
    }
  }

  EventMap *Prune() const override;

  EventMap *MapValues(const std::unordered_set<EventKeyType> &keys_to_map,
                      const std::unordered_map<EventValueType, EventValueType>
                          &value_map) const override;

  /**
   * \brief Constructor from a vector of children indexed by value.
   *
   * Takes ownership of pointers.
   *
   * \param key    The event key this node splits on.
   * \param table  The child EventMaps, indexed by the event value for
   *               \c key; entries may be NULL (no answers for that value).
   *               Ownership is transferred to this object.
   */
  TableEventMap(EventKeyType key, const std::vector<EventMap *> &table)
      : key_(key), table_(table) {}

  /**
   * \brief Constructor from a map of value to child.
   *
   * Takes ownership of pointers.
   *
   * \param key     The event key this node splits on.
   * \param map_in  Maps event values to child EventMaps; ownership is
   *                transferred to this object.
   */
  TableEventMap(EventKeyType key,
                const std::map<EventValueType, EventMap *> &map_in);

  /**
   * \brief Constructor from a map of value to constant answer.
   *
   * This initializer creates a ConstantEventMap for each value in the map.
   *
   * \param key     The event key this node splits on.
   * \param map_in  Maps event values to constant answers.
   */
  TableEventMap(EventKeyType key,
                const std::map<EventValueType, EventAnswerType> &map_in);

  void Write(std::ostream &os, bool binary) override;
  static TableEventMap *Read(std::istream &is, bool binary);

  EventMap *Copy(const std::vector<EventMap *> &new_leaves) const override {
    std::vector<EventMap *> new_table_(table_.size(), nullptr);
    for (size_t i = 0; i < table_.size(); i++) {
      if (table_[i]) {
        new_table_[i] = table_[i]->Copy(new_leaves);
      }
    }

    return new TableEventMap(key_, new_table_);
  }
  virtual ~TableEventMap() { DeletePointers(&table_); }

 private:
  EventKeyType key_;
  std::vector<EventMap *> table_;
};

/**
 * \brief A decision tree [non-leaf] node: a binary yes/no question about one
 *        event key.
 *
 * The node asks whether the value of \c key_ in the event belongs to
 * \c yes_set_; it recurses into the "yes" child or the "no" child
 * accordingly.  This is the classic decision-tree question node.
 */
class SplitEventMap : public EventMap {
 public:
  bool Map(const EventType &event, EventAnswerType *ans) const override {
    EventValueType value;
    if (Lookup(event, key_, &value)) {
      // if (std::binary_search(yes_set_.begin(), yes_set_.end(), value)) {
      if (yes_set_.count(value)) {
        return yes_->Map(event, ans);
      }
      return no_->Map(event, ans);
    }
    return false;
  }

  void MultiMap(const EventType &event,
                std::vector<EventAnswerType> *ans) const override {
    EventValueType tmp;
    if (Lookup(event, key_, &tmp)) {
      if (std::binary_search(yes_set_.begin(), yes_set_.end(), tmp)) {
        yes_->MultiMap(event, ans);
      } else {
        no_->MultiMap(event, ans);
      }
    } else {  // both yes and no contribute.
      yes_->MultiMap(event, ans);
      no_->MultiMap(event, ans);
    }
  }

  void GetChildren(std::vector<EventMap *> *out) const override {
    out->clear();
    out->push_back(yes_);
    out->push_back(no_);
  }

  EventMap *Copy(const std::vector<EventMap *> &new_leaves) const override {
    return new SplitEventMap(key_, yes_set_, yes_->Copy(new_leaves),
                             no_->Copy(new_leaves));
  }

  void Write(std::ostream &os, bool binary) override;
  static SplitEventMap *Read(std::istream &is, bool binary);

  EventMap *Prune() const override;

  EventMap *MapValues(const std::unordered_set<EventKeyType> &keys_to_map,
                      const std::unordered_map<EventValueType, EventValueType>
                          &value_map) const override;

  virtual ~SplitEventMap() { Destroy(); }

  /**
   * \brief Constructor taking ownership of the "yes" and "no" arguments.
   *
   * \param key      The event key this node splits on.
   * \param yes_set  The set of values of \c key for which we take the "yes"
   *                 branch; must be sorted and unique.
   * \param yes      The child for values in \c yes_set; ownership is
   *                 transferred to this object.
   * \param no       The child for values not in \c yes_set; ownership is
   *                 transferred to this object.
   */
  SplitEventMap(EventKeyType key, const std::vector<EventValueType> &yes_set,
                EventMap *yes, EventMap *no)
      : key_(key), yes_set_(yes_set), yes_(yes), no_(no) {
    KHG_PARANOID_ASSERT(IsSorted(yes_set));
    KHG_ASSERT(yes_ != NULL && no_ != NULL);
  }

 private:
  /// This constructor used in the Copy() function.
  SplitEventMap(EventKeyType key,
                const ConstIntegerSet<EventValueType> &yes_set, EventMap *yes,
                EventMap *no)
      : key_(key), yes_set_(yes_set), yes_(yes), no_(no) {
    KHG_ASSERT(yes_ != NULL && no_ != NULL);
  }
  void Destroy() {
    delete yes_;
    delete no_;
  }
  EventKeyType key_;
  //  std::vector<EventValueType> yes_set_;
  ConstIntegerSet<EventValueType> yes_set_;  // more efficient Map function.
  EventMap *yes_;                            // owned here.
  EventMap *no_;                             // owned here.
  SplitEventMap &operator=(const SplitEventMap &other);  // Disallow.
};

/**
 * \brief Extracts the tree structure of an EventMap into a parent array.
 *
 * This function gets the tree structure of the EventMap "map" in a convenient
 * form. If "map" corresponds to a tree structure (not necessarily binary) with
 * leaves uniquely numbered from 0 to num_leaves-1, then the function will
 * return true, output "num_leaves", and set "parent" to a vector of size equal
 * to the number of nodes in the tree (nonleaf and leaf), where each index
 * corresponds to a node and the leaf indices correspond to the values returned
 * by the EventMap from that leaf; for an index i, parent[i] equals the parent
 * of that node in the tree structure, where parent[i] > i, except for the last
 * (root) node where parent[i] == i. If the EventMap does not have this
 * structure (e.g. if multiple different leaf nodes share the same number), then
 * it will return false.
 *
 * \param map       [in] The EventMap to inspect.
 * \param num_leaves  [out] The number of leaves (only set on success).
 * \param parents   [out] For each node, the index of its parent (see above);
 *                  only set on success.
 * \return  \c true if the EventMap has the required tree structure.
 */
bool GetTreeStructure(const EventMap &map, int32_t *num_leaves,
                      std::vector<int32_t> *parents);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_EVENT_MAP_H_
