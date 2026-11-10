// kaldi-hmm-gmm/csrc/transition-model.h
//
// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Guoguo Chen)
// Copyright (c)  2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
#define KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_

/**
 * \file transition-model.h
 * \brief Transition probabilities and integer-id mappings for HMM-GMM models.
 *
 * This file is copied/modified from kaldi/src/hmm/transition-model.h.
 *
 * The main class, TransitionModel, is the repository for the HMM transition
 * probabilities of an acoustic model and the place where the toolkit's integer
 * id spaces (phone, HMM-state, pdf-id, transition-state, transition-index,
 * transition-id) are defined and mapped onto each other.  The transition
 * probabilities are associated with (phone, HMM-state, pdf-id) triples that are
 * resolved through the phonetic decision tree; the pdf-classes of the HMM
 * topology are replaced by actual pdf-ids via the ContextDependencyInterface.
 *
 * The API falls into the following groups:
 *
 * - Construction and I/O: initialization from a
 *   \c ContextDependencyInterface and a \c HmmTopology (e.g. at the start of
 *   training), and kaldi-compatible Read()/Write() (no symbol table; the
 *   topology is always stored without symbols).
 * - Integer mappings between the id spaces (see the TransitionModel class
 *   docs): TupleToTransitionState(), PairToTransitionId(),
 *   TransitionIdToTransitionState(), TransitionStateToPhone(),
 *   TransitionStateToHmmState(), TransitionStateToForwardPdf(),
 *   TransitionStateToSelfLoopPdf(), ...
 * - Transition probabilities: GetTransitionProb(), GetTransitionLogProb(),
 *   GetTransitionLogProbIgnoringSelfLoops(), GetNonSelfLoopLogProb(),
 *   SelfLoopOf().  Transition costs used inside graphs are \f$-\log p\f$,
 *   possibly scaled.
 * - Training: InitStats()/Accumulate() gather counts indexed by
 *   transition-id; MleUpdate() performs maximum-likelihood re-estimation of
 *   the transition probabilities (configured by
 *   MleTransitionUpdateConfig).
 *
 * TransitionModel implements the \c kaldi_decoder::TransitionInformation
 * interface from the kaldi-decoder project; that is how the decoder stack
 * learns the transition-id-to-pdf mapping and per-transition-id properties
 * without depending on this class directly.
 *
 * Typical usage: construct the object from the same tree and topology that
 * were used to build the model; persist it in a kaldi-style \c final.mdl
 * together with an \c AmDiagGmm and a \c ContextDependency; use it to map
 * alignments (transition-id sequences) to pdfs and phones, and to look up
 * transition probabilities during graph construction and decoding.
 *
 * See also kaldi/src/hmm/transition-model.h.
 */

#include <algorithm>
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-decoder/csrc/transition-information.h"
#include "kaldi_native_io/csrc/kaldi-vector.h"

namespace khg {

/// Configuration for the maximum-likelihood transition-probability update
/// performed by TransitionModel::MleUpdate().
///
/// Note: this was previously called \c TransitionUpdateConfig.
struct MleTransitionUpdateConfig {
  /// Floor for transition probabilities.
  float floor;

  /// Minimum count required to update transitions from a state.
  float mincount;

  /// If true, share all transition parameters that have the same pdf.
  bool share_for_pdfs;
  MleTransitionUpdateConfig(float floor = 0.01, float mincount = 5.0,
                            bool share_for_pdfs = false)
      : floor(floor), mincount(mincount), share_for_pdfs(share_for_pdfs) {}
};

/**
 * The class TransitionModel is a repository for the transition probabilities.
 * It also handles certain integer mappings.  It implements the
 * \c kaldi_decoder::TransitionInformation interface (from the kaldi-decoder
 * project), so decoders and lattice tools can use it through that interface.
 *
 * The basic model is as follows.  Each phone has a HMM topology defined in
 * hmm-topology.h.  Each HMM-state of each of these phones has a number of
 * transitions (and final-probs) out of it.  Each HMM-state defined in the
 * HmmTopology class has an associated "pdf_class".  This gets replaced with
 * an actual pdf-id via the tree.  The transition model associates the
 * transition probs with the (phone, HMM-state, pdf-id).  We associate with
 * each such triple a transition-state.  Each transition-state has a number of
 * associated probabilities to estimate; this depends on the number of
 * transitions/final-probs in the topology for that (phone, HMM-state).  Each
 * probability has an associated transition-index.  We associate with each
 * (transition-state, transition-index) a unique transition-id.  Each
 * individual probability estimated by the transition-model is associated
 * with a transition-id.
 *
 * List of the various types of quantity referred to here and what they mean:
 * - \c phone: a phone index (1, 2, 3 ...)
 * - \c HMM-state: a number (0, 1, 2...) that indexes TopologyEntry (see
 *   hmm-topology.h)
 * - \c pdf-id: a number output by the Compute function of ContextDependency
 *   (it indexes pdf's, either forward or self-loop).  Zero-based.
 * - \c transition-state: the states for which we estimate transition
 *   probabilities for transitions out of them.  In some topologies, will map
 *   one-to-one with pdf-ids.  One-based, since it appears on FSTs.
 * - \c transition-index: identifier of a transition (or final-prob) in the
 *   HMM.  Indexes the "transitions" vector in HmmTopology::HmmState.  [if it
 *   is out of range, equal to transitions.size(), it refers to the
 *   final-prob.]  Zero-based.
 * - \c transition-id: identifier of a unique parameter of the
 *   TransitionModel.  Associated with a (transition-state, transition-index)
 *   pair.  One-based, since it appears on FSTs.
 *
 * List of the possible mappings TransitionModel can do:
 * - (phone, HMM-state, forward-pdf-id, self-loop-pdf-id) -> transition-state
 * - (transition-state, transition-index) -> transition-id
 *
 * Reverse mappings:
 * - transition-id -> transition-state
 * - transition-id -> transition-index
 * - transition-state -> phone
 * - transition-state -> HMM-state
 * - transition-state -> forward-pdf-id
 * - transition-state -> self-loop-pdf-id
 *
 * The main things the TransitionModel object can do are:
 * - Get initialized (need ContextDependency and HmmTopology objects).
 * - Read/write.
 * - Update [given a vector of counts indexed by transition-id].
 * - Do the various integer mappings mentioned above.
 * - Get the probability (or log-probability) associated with a particular
 *   transition-id.
 */
class TransitionModel : public kaldi_decoder::TransitionInformation {
 public:
  /// Initialize the object [e.g. at the start of training].
  /// The class keeps a copy of the HmmTopology object, but not
  /// the ContextDependency object.
  ///
  /// \param ctx_dep  The phonetic decision tree; used to map pdf-classes to
  ///                 pdf-ids.  Not stored in this object.
  /// \param hmm_topo The HMM topology; a copy is stored in this object.
  TransitionModel(const ContextDependencyInterface &ctx_dep,
                  const HmmTopology &hmm_topo);

  /// Constructor that takes no arguments: typically used prior to calling
  /// Read.
  TransitionModel() : num_pdfs_(0) {}

  /// Read the object from a stream.
  /// Note: no symbol table: topo object always read/written w/o symbols.
  /// \param is      Stream to read from.
  /// \param binary  If true, read in binary mode; otherwise in text mode.
  void Read(std::istream &is, bool binary);

  /// Write the object to a stream.
  /// Note: no symbol table: topo object always read/written w/o symbols.
  /// \param os      Stream to write to.
  /// \param binary  If true, write in binary mode; otherwise in text mode.
  void Write(std::ostream &os, bool binary) const;

  /// Return reference to the HMM-topology object.
  /// \return The HmmTopology object stored in this class.
  const HmmTopology &GetTopo() const { return topo_; }

  /// Returns a sorted, unique list of phones.
  /// \return The sorted, uniq list of phones covered by the topology.
  const std::vector<int32_t> &GetPhones() const { return topo_.GetPhones(); }

  /// Returns the probability (not the log-probability) of the transition
  /// associated with \p trans_id.
  /// \param trans_id  Transition-id (one-based).
  /// \return The transition probability \f$p\f$.
  float GetTransitionProb(int32_t trans_id) const;

  /// Returns the array that backs TransitionIdToPdf(): element \c t is the
  /// pdf-id of transition-id \c t (transition-ids are one-based, so element 0
  /// is unused).  Part of the \c kaldi_decoder::TransitionInformation
  /// interface.
  /// \return The transition-id to pdf-id mapping array.
  const std::vector<int32_t> &TransitionIdToPdfArray() const override;

  /// Return true if \p trans_id corresponds to a self-loop.
  /// \param trans_id  Transition-id (one-based).
  /// \return True if the transition is a self-loop.
  bool IsSelfLoop(int32_t trans_id) const override;

  /// Return true if the two transition-ids belong to the same
  /// transition-state, i.e. they are transitions out of the same HMM state of
  /// the same phone (in this implementation they are equivalent iff
  /// TransitionIdToTransitionState() gives the same value for both; in
  /// particular a transition-id is equivalent to itself).  Part of the
  /// \c kaldi_decoder::TransitionInformation interface.
  /// \param trans_id1  First transition-id (one-based).
  /// \param trans_id2  Second transition-id (one-based).
  /// \return True if the two transition-ids belong to the same
  ///         transition-state.
  bool TransitionIdsEquivalent(int32_t trans_id1,
                               int32_t trans_id2) const override;

  /// Return true if the hmm state corresponding to the given transition ID
  /// is the 0th state of the topology (i.e. this transition-id starts a new
  /// phone in an alignment).
  /// \param trans_id  Transition-id (one-based).
  /// \return True if the transition-id is in the 0th HMM state of its phone.
  bool TransitionIdIsStartOfPhone(int32_t trans_id) const override;

  /// Return the phone corresponding to the given transition id.
  /// \param trans_id  Transition-id (one-based).
  /// \return The phone index (one-based).
  int32_t TransitionIdToPhone(int32_t trans_id) const override;

  /// Returns true if \p trans_id goes to the final state (which is bound to
  /// be nonemitting).
  /// \param trans_id  Transition-id (one-based).
  /// \return True if the transition goes to the final (nonemitting) state.
  bool IsFinal(int32_t trans_id) const override;

  /// NumPdfs() actually returns the highest-numbered pdf we ever saw, plus
  /// one.  In normal cases this should equal the number of pdfs in the
  /// system, but if you initialized this object with fewer than all the
  /// phones, and it happens that an unseen phone has the highest-numbered
  /// pdf, this might be different.
  /// \return One plus the highest-numbered pdf seen from the tree.
  int32_t NumPdfs() const override { return num_pdfs_; }

  /// Maps a (phone, HMM-state, pdf, self-loop-pdf) tuple to its
  /// transition-state.
  /// Will crash if no transition state matches the given four tuple.
  /// \param phone        Phone index (one-based).
  /// \param hmm_state    HMM-state index within the phone's topology entry
  ///                     (zero-based).
  /// \param pdf          Forward pdf-id (zero-based; a pdf-id, not a
  ///                     pdf-class).
  /// \param self_loop_pdf Self-loop pdf-id (zero-based; a pdf-id, not a
  ///                     pdf-class).
  /// \return The transition-state (one-based).
  int32_t TupleToTransitionState(int32_t phone, int32_t hmm_state, int32_t pdf,
                                 int32_t self_loop_pdf) const;

  /// Return the transition id given the transition state and the transition
  /// index.
  /// \param trans_state  Transition-state (one-based).
  /// \param trans_index  Transition-index (zero-based offset into the
  ///                     transitions out of the state; equal to the number of
  ///                     transitions refers to the final-prob).
  /// \return The transition-id (one-based).
  int32_t PairToTransitionId(int32_t trans_state, int32_t trans_index) const;

  /// Returns the log-probability of the transition associated with
  /// \p trans_id, i.e. \f$\log p\f$.
  /// \param trans_id  Transition-id (one-based).
  /// \return The log-probability of the transition.
  float GetTransitionLogProb(int32_t trans_id) const;

  /// The following functions are more specialized functions for getting
  /// transition probabilities, that are provided for convenience.
  ///
  /// Returns the log-probability of a particular non-self-loop transition
  /// after subtracting the probability mass of the self-loop and
  /// renormalizing; will crash if called on a self-loop.  Specifically:
  /// for non-self-loops it returns the log of (that prob divided by (1 minus
  /// self-loop-prob-for-that-state)), i.e.
  /// \f$\log\bigl(p / (1 - p_{\mathrm{sl}})\bigr)\f$.
  /// \param trans_id  Transition-id (one-based) of a non-self-loop transition.
  /// \return The renormalized log-probability.
  float GetTransitionLogProbIgnoringSelfLoops(int32_t trans_id) const;

  /// Returns the log-prob of the non-self-loop probability mass for this
  /// transition state, i.e. \f$\log(1 - p_{\mathrm{sl}})\f$ (you can get the
  /// self-loop prob, if a self-loop exists, by calling
  /// GetTransitionLogProb(SelfLoopOf(trans_state))).
  /// \param trans_state  Transition-state (one-based).
  /// \return The log of the non-self-loop probability mass.
  float GetNonSelfLoopLogProb(int32_t trans_state) const;

  /// Returns the self-loop transition-id, or zero if this state doesn't have
  /// a self-loop.
  /// \param trans_state  Transition-state (one-based).
  /// \return The transition-id of the self-loop, or 0 if none exists.
  int32_t SelfLoopOf(int32_t trans_state) const;

  /// Returns the transition-state that the given transition-id belongs to.
  /// \param trans_id  Transition-id (one-based).
  /// \return The transition-state (one-based).
  int32_t TransitionIdToTransitionState(int32_t trans_id) const;

  /// Initializes the stats vector used by Accumulate()/MleUpdate(): sets it
  /// to zero with length NumTransitionIds() + 1.  Transition-ids start from
  /// 1, so stats[0] is never used.
  /// \param stats  [out] 1-D double vector to initialize.
  void InitStats(DoubleVector *stats) const {
    // transition id starts from 1
    // stats[0] is never used
    *stats = DoubleVector::Zero(NumTransitionIds() + 1);
  }

  /// Accumulates a count/weight for one transition-id into the stats vector
  /// (as initialized by InitStats()).
  /// \param prob      Count or weight to add.
  /// \param trans_id  Transition-id (one-based).
  /// \param stats     [in,out] 1-D double vector of shape
  ///                  (num_transition_ids + 1,).
  void Accumulate(float prob, int32_t trans_id, DoubleVector *stats) const {
    KHG_ASSERT(trans_id <= NumTransitionIds());

    (*stats)[trans_id] += prob;
    // This is trivial and doesn't require class members, but leaves us more
    // open to design changes than doing it manually.
  }

  /// Does Maximum Likelihood estimation.  The stats are counts/weights,
  /// indexed by transition-id.  This was previously called Update().
  ///
  /// \param stats          1-D double vector of shape
  ///                       (num_transition_ids + 1,), as filled in by
  ///                       InitStats()/Accumulate().
  /// \param cfg            Update configuration (probability floor, minimum
  ///                       count, optional sharing for equal pdfs).
  /// \param objf_impr_out  [out] If non-NULL, the total objective-function
  ///                       improvement of the update is written here.
  /// \param count_out      [out] If non-NULL, the total count (sum of the
  ///                       stats) is written here.
  void MleUpdate(const DoubleVector &stats,
                 const MleTransitionUpdateConfig &cfg, float *objf_impr_out,
                 float *count_out);

  /// Returns the total number of transition-states (note, these are
  /// one-based).
  /// \return The number of transition-states.
  int32_t NumTransitionStates() const { return tuples_.size(); }

  /// Returns the phone corresponding to the given transition-state.
  /// \param trans_state  Transition-state (one-based).
  /// \return The phone index (one-based).
  int32_t TransitionStateToPhone(int32_t trans_state) const;

  /// Returns the HMM-state index (within the phone's topology entry)
  /// corresponding to the given transition-state.
  /// Note: this is public in kaldi/src/hmm/transition-model.h (used by e.g.
  /// SplitToPhones); keep the same access level here.
  /// \param trans_state  Transition-state (one-based).
  /// \return The HMM-state (zero-based) within the phone's TopologyEntry.
  int32_t TransitionStateToHmmState(int32_t trans_state) const;

  /// Returns the self-loop pdf-id corresponding to the given
  /// transition-state.
  /// \param trans_state  Transition-state (one-based).
  /// \return The self-loop pdf-id (zero-based).
  int32_t TransitionStateToSelfLoopPdf(int32_t trans_state) const;

  /// Returns the forward pdf-id corresponding to the given transition-state.
  /// \param trans_state  Transition-state (one-based).
  /// \return The forward pdf-id (zero-based).
  int32_t TransitionStateToForwardPdf(int32_t trans_state) const;

 private:
  /// Called from constructor.  initializes tuples_.
  void ComputeTuples(const ContextDependencyInterface &ctx_dep);

  /// Returns true if the topology is really 'hmm-like'; see
  /// HmmTopology::IsHmm().
  bool IsHmm() const;

  /// Computes tuples_ for the 'hmm-like' case, where the forward and
  /// self-loop pdf of each state coincide.
  void ComputeTuplesIsHmm(const ContextDependencyInterface &ctx_dep);

  /// Computes tuples_ for the 'non-hmm-like' case, where the forward and
  /// self-loop pdf of a state may differ.
  void ComputeTuplesNotHmm(const ContextDependencyInterface &ctx_dep);

  /// Called from constructor and Read function: computes state2id_ and
  /// id2state_.
  void ComputeDerived();

  /// Called from constructor.
  void InitializeProbs();

  /// computes quantities derived from log-probs (currently just
  /// non_self_loop_log_probs_; called whenever log-probs change.
  void ComputeDerivedOfProbs();

  /// Checks that the object is internally consistent; crashes otherwise.
  void Check() const;

  /// Returns the number of transition-indices for a particular
  /// transition-state.  Note: "Indices" is the plural of "index".  Index is
  /// not the same as "id", here.  A transition-index is a zero-based offset
  /// into the transitions out of a particular transition state.
  /// \param trans_state  Transition-state (one-based).
  /// \return The number of transitions (including the final-prob) out of the
  ///         state.
  int32_t NumTransitionIndices(int32_t trans_state) const;

  /// Returns the transition-index of the given transition-id, i.e. its
  /// offset within the transitions out of its transition-state.
  /// \param trans_id  Transition-id (one-based).
  int32_t TransitionIdToTransitionIndex(int32_t trans_id) const;

  /// Returns the HMM-state (zero-based, within the phone's topology entry)
  /// of the given transition-id.
  /// \param trans_id  Transition-id (one-based).
  int32_t TransitionIdToHmmState(int32_t trans_id) const;

  /// As MleUpdate(), but pools the counts of all transition parameters that
  /// share the same pdf before updating (used when
  /// MleTransitionUpdateConfig::share_for_pdfs is true).
  void MleUpdateShared(const DoubleVector &stats,
                       const MleTransitionUpdateConfig &cfg,
                       float *objf_impr_out, float *count_out);

 public:
  /// Identifies a transition-state: the (phone, HMM-state, forward-pdf,
  /// self-loop-pdf) quadruple that the transition-state stands for.  The
  /// tuples are kept sorted, which maps quadruples to transition-states.
  struct Tuple {
    /// Phone index (one-based).
    int32_t phone;
    /// HMM-state index within the phone's topology entry (zero-based).
    int32_t hmm_state;
    /// Forward pdf-id (it is pdf id, not pdf class).
    int32_t forward_pdf;
    /// Self-loop pdf-id (it is pdf id, not pdf class).
    int32_t self_loop_pdf;
    Tuple() = default;

    /// Constructor filling in all four members.
    Tuple(int32_t phone, int32_t hmm_state, int32_t forward_pdf,
          int32_t self_loop_pdf)
        : phone(phone),
          hmm_state(hmm_state),
          forward_pdf(forward_pdf),
          self_loop_pdf(self_loop_pdf) {}

    /// Ordering on (phone, hmm_state, forward_pdf, self_loop_pdf), used to
    /// keep the tuple list sorted so that the reverse mapping
    /// tuple -> transition-state is possible.
    bool operator<(const Tuple &other) const {
      if (phone < other.phone) {
        return true;
      } else if (phone > other.phone) {
        return false;
      } else if (hmm_state < other.hmm_state) {
        return true;
      } else if (hmm_state > other.hmm_state) {
        return false;
      } else if (forward_pdf < other.forward_pdf) {
        return true;
      } else if (forward_pdf > other.forward_pdf) {
        return false;
      } else {
        return (self_loop_pdf < other.self_loop_pdf);
      }
    }
    /// Equality: all four members are equal.
    bool operator==(const Tuple &other) const {
      return (phone == other.phone && hmm_state == other.hmm_state &&
              forward_pdf == other.forward_pdf &&
              self_loop_pdf == other.self_loop_pdf);
    }
  };

  /// Assembles a TransitionModel directly from its components, without
  /// consulting a tree or re-deriving the mappings.  The vectors have the
  /// same meaning as the correspondingly named private members (see their
  /// docs): \p state2id is indexed by transition-state (1..num-states+1),
  /// \p id2state and \p id2pdf_id are indexed by transition-id (one-based),
  /// and the transition-id-indexed entries of the probability vectors are
  /// likewise one-based.
  /// \param tuples     The tuple list, indexed by transition-state minus one.
  /// \param topo       The HMM topology.
  /// \param state2id   First transition-id of each transition-state.
  /// \param id2state   Transition-id to transition-state mapping.
  /// \param id2pdf_id  Transition-id to pdf-id mapping.
  /// \param num_pdfs   One plus the highest-numbered pdf.
  /// \param log_probs  Log-probabilities indexed by transition-id.
  /// \param non_self_loop_log_probs  Log of (1 - self-loop-prob) per
  ///                   transition-state.
  TransitionModel(const std::vector<Tuple> &tuples, const HmmTopology &topo,
                  const std::vector<int32_t> state2id,
                  const std::vector<int32_t> id2state,
                  const std::vector<int32_t> id2pdf_id, int32_t num_pdfs,
                  const std::vector<float> log_probs,
                  const std::vector<float> non_self_loop_log_probs)
      : tuples_(tuples),
        topo_(topo),
        state2id_(state2id),
        id2state_(id2state),
        id2pdf_id_(id2pdf_id),
        num_pdfs_(num_pdfs) {
    log_probs_.Resize(log_probs.size());
    std::copy(log_probs.begin(), log_probs.end(), log_probs_.Data());

    non_self_loop_log_probs_.Resize(non_self_loop_log_probs.size());
    std::copy(non_self_loop_log_probs.begin(), non_self_loop_log_probs.end(),
              non_self_loop_log_probs_.Data());
  }
  /// Returns the tuple list (indexed by transition-state minus one).
  /// \return The tuples_ member.
  const std::vector<Tuple> &GetTuples() const { return tuples_; }
  /// Returns the state2id_ mapping (first transition-id per transition-state).
  /// \return The state2id_ member.
  const std::vector<int32_t> &GetState2Id() const { return state2id_; }
  /// Returns the id2state_ mapping (transition-id to transition-state).
  /// \return The id2state_ member.
  const std::vector<int32_t> &GetId2State() const { return id2state_; }
  /// Returns the id2pdf_id_ mapping (transition-id to pdf-id).
  /// \return The id2pdf_id_ member.
  const std::vector<int32_t> &GetId2PdfId() const { return id2pdf_id_; }
  /// Returns the log-probabilities indexed by transition-id.
  /// \return The log_probs_ member.
  const kaldiio::Vector<float> &GetLogProbs() const { return log_probs_; }
  /// Returns, for each transition-state, the log of (1 - self-loop-prob).
  /// \return The non_self_loop_log_probs_ member.
  const kaldiio::Vector<float> &GetNonSelfLoopLogProbs() const {
    return non_self_loop_log_probs_;
  }

 private:
  /// Tuples indexed by transition state minus one;
  /// the tuples are in sorted order which allows us to do the reverse mapping
  /// from tuple to transition state.
  /// Index of tuple_s plus 1 is called transition state;
  /// transition state starts from 1,
  /// transition id also starts from 1,
  /// pdf id starts from 0.
  std::vector<Tuple> tuples_;

  /// Copy of the HMM topology this model was built from.
  HmmTopology topo_;

  /// Gives the first transition_id of each transition-state; indexed by
  /// the transition-state.  Array indexed 1..num-transition-states+1 (the last
  /// one is needed so we can know the num-transitions of the last
  /// transition-state.
  std::vector<int32_t> state2id_;

  /// For each transition-id, the corresponding transition
  /// state (indexed by transition-id).
  /// Valid transition id starts from 1
  std::vector<int32_t> id2state_;

  /// transition id to pdf id.
  /// valid transition id starts from 1
  std::vector<int32_t> id2pdf_id_;

  /// This is actually one plus the highest-numbered pdf we ever got back from
  /// the tree (but the tree numbers pdfs contiguously from zero so this is the
  /// number of pdfs).
  int32_t num_pdfs_;

  /// For each transition-id, the corresponding log-prob.  Indexed by
  /// transition-id.
  kaldiio::Vector<float> log_probs_;

  /// For each transition-state, the log of (1 - self-loop-prob).  Indexed by
  /// transition-state.
  kaldiio::Vector<float> non_self_loop_log_probs_;
};

/// Works out which pdfs might correspond to the given phones.  Will return
/// true if these pdfs correspond *just* to these phones, false if these pdfs
/// are also used by other phones.
/// \param trans_model [in] Transition-model used to work out this information
/// \param phones [in] A sorted, uniq vector that represents a set of phones
/// \param pdfs [out] Will be set to a sorted, uniq list of pdf-ids that
///        correspond to one of this set of phones.
/// \return Returns true if all of the pdfs output to "pdfs" correspond to
///         phones from this set (false if they may be shared with phones
///         outside this set).
bool GetPdfsForPhones(const TransitionModel &trans_model,
                      const std::vector<int32_t> &phones,
                      std::vector<int32_t> *pdfs);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
