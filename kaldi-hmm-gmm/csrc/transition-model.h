// kaldi-hmm-gmm/csrc/transition-model.h
//
// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Guoguo Chen)
// Copyright (c)  2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
#define KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_

// this if is copied and modified from
// kaldi/src/hmm/transition-model.h
//
#include <algorithm>
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/transition-information.h"
#include "kaldi_native_io/csrc/kaldi-vector.h"

namespace khg {

// The class TransitionModel is a repository for the transition probabilities.
// It also handles certain integer mappings.
// The basic model is as follows.  Each phone has a HMM topology defined in
// hmm-topology.h.  Each HMM-state of each of these phones has a number of
// transitions (and final-probs) out of it.  Each HMM-state defined in the
// HmmTopology class has an associated "pdf_class".  This gets replaced with
// an actual pdf-id via the tree.  The transition model associates the
// transition probs with the (phone, HMM-state, pdf-id).  We associate with
// each such triple a transition-state.  Each
// transition-state has a number of associated probabilities to estimate;
// this depends on the number of transitions/final-probs in the topology for
// that (phone, HMM-state).  Each probability has an associated
// transition-index. We associate with each (transition-state, transition-index)
// a unique transition-id. Each individual probability estimated by the
// transition-model is associated with a transition-id.
//
// List of the various types of quantity referred to here and what they mean:
//           phone:  a phone index (1, 2, 3 ...)
//       HMM-state:  a number (0, 1, 2...) that indexes TopologyEntry (see
//       hmm-topology.h)
//          pdf-id:  a number output by the Compute function of
//          ContextDependency (it
//                   indexes pdf's, either forward or self-loop).  Zero-based.
// transition-state:  the states for which we estimate transition probabilities
// for transitions
//                    out of them.  In some topologies, will map one-to-one with
//                    pdf-ids. One-based, since it appears on FSTs.
// transition-index:  identifier of a transition (or final-prob) in the HMM.
// Indexes the
//                    "transitions" vector in HmmTopology::HmmState.  [if it is
//                    out of range, equal to transitions.size(), it refers to
//                    the final-prob.] Zero-based.
//   transition-id:   identifier of a unique parameter of the TransitionModel.
//                    Associated with a (transition-state, transition-index)
//                    pair. One-based, since it appears on FSTs.
//
// List of the possible mappings TransitionModel can do:
//   (phone, HMM-state, forward-pdf-id, self-loop-pdf-id) -> transition-state
//                   (transition-state, transition-index) -> transition-id
//  Reverse mappings:
//                        transition-id -> transition-state
//                        transition-id -> transition-index
//                     transition-state -> phone
//                     transition-state -> HMM-state
//                     transition-state -> forward-pdf-id
//                     transition-state -> self-loop-pdf-id
//
// The main things the TransitionModel object can do are:
//    Get initialized (need ContextDependency and HmmTopology objects).
//    Read/write.
//    Update [given a vector of counts indexed by transition-id].
//    Do the various integer mappings mentioned above.
//    Get the probability (or log-probability) associated with a particular
//    transition-id.

// Note: this was previously called TransitionUpdateConfig.
struct MleTransitionUpdateConfig {
  // floor for transition probabilities
  float floor;

  // Minimum count required to update transitions from a state
  float mincount;

  bool share_for_pdfs;  // If true, share all transition parameters that have
                        // the same pdf.
  MleTransitionUpdateConfig(float floor = 0.01, float mincount = 5.0,
                            bool share_for_pdfs = false)
      : floor(floor), mincount(mincount), share_for_pdfs(share_for_pdfs) {}
};

class TransitionModel : public TransitionInformation {
 public:
  /// Initialize the object [e.g. at the start of training].
  /// The class keeps a copy of the HmmTopology object, but not
  /// the ContextDependency object.
  TransitionModel(const ContextDependencyInterface &ctx_dep,
                  const HmmTopology &hmm_topo);

  /// Constructor that takes no arguments: typically used prior to calling Read.
  TransitionModel() : num_pdfs_(0) {}

  // note, no symbol table: topo object always read/written w/o symbols.
  void Read(std::istream &is, bool binary);

  void Write(std::ostream &os, bool binary) const;

  /// return reference to HMM-topology object.
  const HmmTopology &GetTopo() const { return topo_; }

  /// Returns a sorted, unique list of phones.
  const std::vector<int32_t> &GetPhones() const { return topo_.GetPhones(); }

  // Transition-parameter-getting functions:
  float GetTransitionProb(int32_t trans_id) const;

  const std::vector<int32_t> &TransitionIdToPdfArray() const override;

  // return true if this trans_id corresponds to a self-loop.
  bool IsSelfLoop(int32_t trans_id) const override;

  bool TransitionIdsEquivalent(int32_t trans_id1,
                               int32_t trans_id2) const override;

  // return true if the hmm state corresponding to the given transition ID
  // is the 0th state of the topology
  bool TransitionIdIsStartOfPhone(int32_t trans_id) const override;

  // return the phone corresponding to the given transition id
  int32_t TransitionIdToPhone(int32_t trans_id) const override;

  bool IsFinal(int32_t trans_id)
      const override;  // returns true if this trans_id goes to the final state
                       // (which is bound to be nonemitting).

  // NumPdfs() actually returns the highest-numbered pdf we ever saw, plus one.
  // In normal cases this should equal the number of pdfs in the system, but if
  // you initialized this object with fewer than all the phones, and it happens
  // that an unseen phone has the highest-numbered pdf, this might be different.
  int32_t NumPdfs() const override { return num_pdfs_; }

  // will crash if no transition state matches the given four tuple
  int32_t TupleToTransitionState(int32_t phone, int32_t hmm_state, int32_t pdf,
                                 int32_t self_loop_pdf) const;

  // Return the transition id given the transition state and the transition
  // index.
  int32_t PairToTransitionId(int32_t trans_state, int32_t trans_index) const;

  float GetTransitionLogProb(int32_t trans_id) const;

  // The following functions are more specialized functions for getting
  // transition probabilities, that are provided for convenience.

  /// Returns the log-probability of a particular non-self-loop transition
  /// after subtracting the probability mass of the self-loop and renormalizing;
  /// will crash if called on a self-loop.  Specifically:
  /// for non-self-loops it returns the log of (that prob divided by (1 minus
  /// self-loop-prob-for-that-state)).
  float GetTransitionLogProbIgnoringSelfLoops(int32_t trans_id) const;

  /// Returns the log-prob of the non-self-loop probability
  /// mass for this transition state. (you can get the self-loop prob, if a
  /// self-loop exists, by calling
  /// GetTransitionLogProb(SelfLoopOf(trans_state)).
  float GetNonSelfLoopLogProb(int32_t trans_state) const;

  // returns the self-loop transition-id, or zero if
  // this state doesn't have a self-loop.
  int32_t SelfLoopOf(int32_t trans_state) const;

  int32_t TransitionIdToTransitionState(int32_t trans_id) const;

  void InitStats(DoubleVector *stats) const {
    // transition id starts from 1
    // stats[0] is never used
    *stats = DoubleVector::Zero(NumTransitionIds() + 1);
  }

  // @param stats 1-d double tensor
  void Accumulate(float prob, int32_t trans_id, DoubleVector *stats) const {
    KHG_ASSERT(trans_id <= NumTransitionIds());

    (*stats)[trans_id] += prob;
    // This is trivial and doesn't require class members, but leaves us more
    // open to design changes than doing it manually.
  }

  /// Does Maximum Likelihood estimation.  The stats are counts/weights, indexed
  /// by transition-id.  This was previously called Update().
  ///
  /// @param stats 1-D double tensor of shape (num_transiation_ids + 1,)
  void MleUpdate(const DoubleVector &stats,
                 const MleTransitionUpdateConfig &cfg, float *objf_impr_out,
                 float *count_out);

  /// Returns the total number of transition-states (note, these are one-based).
  int32_t NumTransitionStates() const { return tuples_.size(); }

  int32_t TransitionStateToPhone(int32_t trans_state) const;

  int32_t TransitionStateToSelfLoopPdf(int32_t trans_state) const;

  int32_t TransitionStateToForwardPdf(int32_t trans_state) const;

 private:
  // called from constructor.  initializes tuples_.
  void ComputeTuples(const ContextDependencyInterface &ctx_dep);
  bool IsHmm() const;

  void ComputeTuplesIsHmm(const ContextDependencyInterface &ctx_dep);
  void ComputeTuplesNotHmm(const ContextDependencyInterface &ctx_dep);

  void ComputeDerived();  // called from constructor and Read function: computes
                          // state2id_ and id2state_.

  void InitializeProbs();  // called from constructor.

  // computes quantities derived from log-probs
  // (currently just non_self_loop_log_probs_; called whenever log-probs change.
  void ComputeDerivedOfProbs();

  void Check() const;

  /// Returns the number of transition-indices for a particular
  /// transition-state. Note: "Indices" is the plural of "index".   Index is not
  /// the same as "id", here.  A transition-index is a zero-based offset into
  /// the transitions out of a particular transition state.
  int32_t NumTransitionIndices(int32_t trans_state) const;

  int32_t TransitionStateToHmmState(int32_t trans_state) const;

  int32_t TransitionIdToTransitionIndex(int32_t trans_id) const;

  int32_t TransitionIdToHmmState(int32_t trans_id) const;

  void MleUpdateShared(const DoubleVector &stats,
                       const MleTransitionUpdateConfig &cfg,
                       float *objf_impr_out, float *count_out);

 public:
  struct Tuple {
    int32_t phone;
    int32_t hmm_state;
    int32_t forward_pdf;    // it is pdf id, not pdf class
    int32_t self_loop_pdf;  // it is pdf id, not pdf class
    Tuple() = default;

    Tuple(int32_t phone, int32_t hmm_state, int32_t forward_pdf,
          int32_t self_loop_pdf)
        : phone(phone),
          hmm_state(hmm_state),
          forward_pdf(forward_pdf),
          self_loop_pdf(self_loop_pdf) {}

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
    bool operator==(const Tuple &other) const {
      return (phone == other.phone && hmm_state == other.hmm_state &&
              forward_pdf == other.forward_pdf &&
              self_loop_pdf == other.self_loop_pdf);
    }
  };
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
  const std::vector<Tuple> &GetTuples() const { return tuples_; }
  const std::vector<int32_t> &GetState2Id() const { return state2id_; }
  const std::vector<int32_t> &GetId2State() const { return id2state_; }
  const std::vector<int32_t> &GetId2PdfId() const { return id2pdf_id_; }
  const kaldiio::Vector<float> &GetLogProbs() const { return log_probs_; }
  const kaldiio::Vector<float> &GetNonSelfLoopLogProbs() const {
    return non_self_loop_log_probs_;
  }

 private:
  /// Tuples indexed by transition state minus one;
  /// the tuples are in sorted order which allows us to do the reverse mapping
  /// from tuple to transition state
  std::vector<Tuple> tuples_;
  // Index of tuple_s plus 1 is called transition state
  // transition state starts from 1
  // transition id also starts from 1
  // pdf id starts from 0

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

/// Works out which pdfs might correspond to the given phones.  Will return true
/// if these pdfs correspond *just* to these phones, false if these pdfs are
/// also used by other phones.
/// @param trans_model [in] Transition-model used to work out this information
/// @param phones [in] A sorted, uniq vector that represents a set of phones
/// @param pdfs [out] Will be set to a sorted, uniq list of pdf-ids that
/// correspond to one of this set of phones.
/// @return  Returns true if all of the pdfs output to "pdfs" correspond to
/// phones from this set (false if they may be shared with phones outside this
/// set).
bool GetPdfsForPhones(const TransitionModel &trans_model,
                      const std::vector<int32_t> &phones,
                      std::vector<int32_t> *pdfs);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
