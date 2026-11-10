// kaldi-hmm-gmm/csrc/hmm-topology.h

#include <cstdint>
// Compat typedefs formerly provided transitively by old openfst headers.
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_HMM_TOPOLOGY_H_
#define KALDI_HMM_GMM_CSRC_HMM_TOPOLOGY_H_

/**
 * \file hmm-topology.h
 * \brief Per-phone HMM topology: states, pdf-classes, transitions and final
 *  probabilities.
 *
 * This file is copied/modified from kaldi/src/hmm/hmm-topology.h.
 *
 * This module provides the HmmTopology class, which stores one topology entry
 * (a vector of HmmState) for each phone: the states of the phone's HMM, the
 * pdf-class attached to each state (a pdf-class is later mapped to an actual
 * pdf-id by the phonetic decision tree), the transitions out of each state
 * with their default (pre-training) transition probabilities, and the
 * final-probability of the last state.  The constant kNoPdf marks
 * nonemitting states, which have no pdf attached.
 *
 * The API consists of:
 *
 * - The HmmState structure (a single HMM state: pdf-classes, transitions).
 * - Read()/Write() of the kaldi text (or binary) format; the topology is
 *   always stored without any symbol table.
 * - Queries used by tree building and by the TransitionModel:
 *   TopologyForPhone(), NumPdfClasses(), GetPhoneToNumPdfClasses(),
 *   MinLength(), GetPhones(), IsHmm().
 *
 * Usage notes: every phone in the system must be covered by exactly one
 * &lt;TopologyEntry&gt; block; the first state of an entry is its start state and
 * its last state must be nonemitting with no output transitions (its exit
 * probability is one).  IsHmm() distinguishes "normal" topologies (same
 * pdf-class for self-loops and forward transitions) from the more general
 * kind used e.g. by chain models.
 *
 * See also kaldi/src/hmm/hmm-topology.h.
 */

#include <assert.h>

#include <cstdint>
#include <istream>
#include <utility>
#include <vector>

namespace khg {

/// kNoPdf is used where pdf_class or pdf would be used, to indicate,
/// none is there.  Mainly useful in skippable models, but also used
/// for end states.
///
/// A caveat with nonemitting states is that their out-transitions
/// are not trainable, due to technical issues with the way
/// we decided to accumulate the stats.  Any transitions arising from (*)
/// HMM states with "kNoPdf" as the label are second-class transitions.
/// They do not have "transition-states" or "transition-ids" associated
/// with them.  They are used to create the FST version of the
/// HMMs, where they lead to epsilon arcs.
///
/// (*) "arising from" is a bit of a technical term here, due to the way
/// (if reorder == true), we put the transition-id associated with the
/// outward arcs of the state, on the input transition to the state.
///
/// A constant used in the HmmTopology class as the pdf-class
/// kNoPdf, which is used when a HMM-state is nonemitting (has no associated
/// PDF).
static const int32_t kNoPdf = -1;

/**
 * HmmTopology stores the HMM topology of the phones: one topology entry (a
 * vector of HmmState) for each phone covered by the system.
 *
 * The following would be the text form for the "normal" HMM topology.
 * Note that the first state is the start state, and the final state,
 * which must have no output transitions and must be nonemitting, has
 * an exit probability of one (no other state can have nonzero exit
 * probability; you can treat the transition probability to the final
 * state as an exit probability).
 * Note also that it's valid to omit the &lt;PdfClass&gt; entry of the &lt;State&gt;,
 * which will mean we won't have a pdf on that state [non-emitting state].
 * This is equivalent to setting the &lt;PdfClass&gt; to -1.  We do this normally
 * just for the final state.  The Topology object can have multiple
 * &lt;TopologyEntry&gt; blocks.  This is useful if there are multiple types of
 * topology in the system.
 *
 * \code
 * <Topology>
 * <TopologyEntry>
 * <ForPhones> 1 2 3 4 5 6 7 8 </ForPhones>
 * <State> 0 <PdfClass> 0
 * <Transition> 0 0.5
 * <Transition> 1 0.5
 * </State>
 * <State> 1 <PdfClass> 1
 * <Transition> 1 0.5
 * <Transition> 2 0.5
 * </State>
 * <State> 2 <PdfClass> 2
 * <Transition> 2 0.5
 * <Transition> 3 0.5
 * </State>
 * <State> 3
 * </State>
 * </TopologyEntry>
 * </Topology>
 * \endcode
 *
 * Note: the default assignment operator and copy constructor are allowed.
 */
class HmmTopology {
 public:
  /// A structure defined inside HmmTopology to represent a HMM state.
  struct HmmState {
    /// The forward pdf-class, typically 0, 1 or 2 (the same as
    /// the HMM-state index), but may be different to enable us to hardwire
    /// sharing of state, and may be equal to \c kNoPdf == -1 in order to
    /// specify nonemitting states (unusual).
    int32_t forward_pdf_class;

    /// The self-loop pdf-class, similar to the
    /// forward pdf-class.  They will either both be \c kNoPdf, or neither
    /// be \c kNoPdf.
    int32_t self_loop_pdf_class;

    /// A list of transitions, indexed by what we call a 'transition-index'.
    /// The first member of each pair is the index of the next HmmState, and
    /// the second is the default transition probability (before training).
    std::vector<std::pair<int32_t, float>> transitions;

    /// Constructor that sets both pdf-classes to \p pdf_class.
    explicit HmmState(int32_t pdf_class) {
      forward_pdf_class = pdf_class;
      self_loop_pdf_class = pdf_class;
    }

    /// Constructor that sets the two pdf-classes and the transition list.
    HmmState(int32_t forward_pdf_class, int32_t self_loop_pdf_class,
             const std::vector<std::pair<int32_t, float>> &transitions)
        : forward_pdf_class(forward_pdf_class),
          self_loop_pdf_class(self_loop_pdf_class),
          transitions(transitions) {}

    /// Constructor that sets the two pdf-classes; asserts that they are
    /// either both \c kNoPdf or both not \c kNoPdf.
    HmmState(int32_t forward_pdf_class, int32_t self_loop_pdf_class) {
      assert((forward_pdf_class != kNoPdf && self_loop_pdf_class != kNoPdf) ||
             (forward_pdf_class == kNoPdf && self_loop_pdf_class == kNoPdf));
      this->forward_pdf_class = forward_pdf_class;
      this->self_loop_pdf_class = self_loop_pdf_class;
    }

    /// Equality: both pdf-classes and the transition list are equal.
    bool operator==(const HmmState &other) const {
      return (forward_pdf_class == other.forward_pdf_class &&
              self_loop_pdf_class == other.self_loop_pdf_class &&
              transitions == other.transitions);
    }

    /// Default constructor: a nonemitting state (both pdf-classes are
    /// \c kNoPdf), with no transitions.
    HmmState() : forward_pdf_class(-1), self_loop_pdf_class(-1) {}
  };

  /// TopologyEntry is a typedef that represents the topology of
  /// a single (prototype) state.
  using TopologyEntry = std::vector<HmmState>;

  /// Default constructor: an empty topology; fill it via Read() or the
  /// constructor below.
  HmmTopology() = default;

  /// Constructor that fills in the members directly (the members are
  /// documented below).
  /// \param phones    Sorted, uniq list of phones covered by the topology.
  /// \param phone2idx Map from phone to index into \p entries (or -1 if not
  ///                  present).
  /// \param entries   The topology entries, one per covered phone.
  HmmTopology(const std::vector<int32_t> &phones,
              const std::vector<int32_t> &phone2idx,
              const std::vector<TopologyEntry> &entries)
      : phones_(phones), phone2idx_(phone2idx), entries_(entries) {}

  /// Read the topology from a stream, in the format shown in the class
  /// documentation (or its binary equivalent).  Note: no symbol table is
  /// involved; phones are raw integer ids.
  /// \param is      Stream to read from.
  /// \param binary  If true, read in binary mode; otherwise in text mode.
  void Read(std::istream &is, bool binary);

  /// Write the topology to a stream, in the format shown in the class
  /// documentation (or its binary equivalent).  Note: no symbol table is
  /// involved; phones are raw integer ids.
  /// \param os      Stream to write to.
  /// \param binary  If true, write in binary mode; otherwise in text mode.
  void Write(std::ostream &os, bool binary) const;

  /// Returns true if this HmmTopology is really 'hmm-like', i.e. the
  /// pdf-class on the self-loops and forward transitions of all states are
  /// identical.  [note: in HMMs, the densities are associated with the
  /// states.]  We have extended this to support 'non-hmm-like' topologies
  /// (where those pdf-classes are different), in order to make for more
  /// compact decoding graphs in our so-called 'chain models' (AKA
  /// lattice-free MMI), where we use 1-state topologies that have different
  /// pdf-classes for the self-loop and the forward transition.  Note that we
  /// always use the 'reorder=true' option so the 'forward transition' actually
  /// comes before the self-loop.
  /// \return True if the topology is 'hmm-like'.
  bool IsHmm() const;

  /// Checks that the object is valid, and throws an exception otherwise.
  void Check();

  /// Returns a reference to a sorted, unique list of phones covered by
  /// the topology (these phones will be positive integers, and usually
  /// contiguous and starting from one but the toolkit doesn't assume
  /// they are contiguous).
  /// \return The sorted, uniq list of phones.
  const std::vector<int32_t> &GetPhones() const { return phones_; }

  /// Returns a map from phones to indexes into the entries vector (or -1
  /// for phones not present).
  /// \return The phone to entry-index map.
  const std::vector<int32_t> &GetPhone2Idx() const { return phone2idx_; }

  /// Returns the topology entries, indexed as indicated by GetPhone2Idx().
  /// \return The topology entries.
  const std::vector<TopologyEntry> &GetEntries() const { return entries_; }

  /// Returns the topology entry (i.e. vector of HmmState) for this phone;
  /// will throw exception if phone not covered by the topology.
  /// \param phone  Phone index (one-based).
  /// \return The TopologyEntry of the phone.
  const TopologyEntry &TopologyForPhone(int32_t phone) const;

  /// Returns the number of pdf-classes for this phone;
  /// throws exception if phone not covered by this topology.
  /// \param phone  Phone index (one-based).
  /// \return The number of distinct pdf-classes used by the phone's states.
  int32_t NumPdfClasses(int32_t phone) const;

  /// Outputs a vector of int32, indexed by phone, that gives the
  /// number of pdf-classes for the phones; this is
  /// used by tree-building code such as BuildTree().
  /// \param phone2num_pdf_classes  [out] Resized to (highest phone + 1);
  ///        element \c p holds the number of pdf-classes of phone \c p (0 for
  ///        phones not covered).
  void GetPhoneToNumPdfClasses(
      std::vector<int32_t> *phone2num_pdf_classes) const;

  /// Returns the minimum number of frames it takes to traverse this model
  /// for this phone: e.g. 3 for the normal HMM topology.
  /// \param phone  Phone index (one-based).
  /// \return The minimum model length in frames.
  int32_t MinLength(int32_t phone) const;

 private:
  /// list of all phones we have topology for.
  /// Sorted, uniq.  no epsilon (zero) phone.
  std::vector<int32_t> phones_;

  /// map from phones to indexes into the
  /// entries vector (or -1 for not present).
  std::vector<int32_t> phone2idx_;

  /// The topology entries, one per covered phone, indexed as given by
  /// phone2idx_.
  std::vector<TopologyEntry> entries_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_HMM_TOPOLOGY_H_
