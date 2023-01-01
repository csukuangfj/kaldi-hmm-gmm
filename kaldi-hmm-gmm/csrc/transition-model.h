// kaldi-hmm-gmm/csrc/transition-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
#define KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_

// this if is copied and modified from
// kaldi/src/hmm/transition-model.h
//
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/transition-information.h"
#include "kaldi_native_io/csrc/kaldi-vector.h"

namespace khg {

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

  const std::vector<int32_t> &TransitionIdToPdfArray() const override;

  // return true if this trans_id corresponds to a self-loop.
  bool IsSelfLoop(int32_t trans_id) const override;

  bool TransitionIdsEquivalent(int32_t trans_id1,
                               int32_t trans_id2) const override;

  bool TransitionIdIsStartOfPhone(int32_t trans_id) const override;

  int32_t TransitionIdToPhone(int32_t trans_id) const override;

  bool IsFinal(int32_t trans_id)
      const override;  // returns true if this trans_id goes to the final state
                       // (which is bound to be nonemitting).

  // NumPdfs() actually returns the highest-numbered pdf we ever saw, plus one.
  // In normal cases this should equal the number of pdfs in the system, but if
  // you initialized this object with fewer than all the phones, and it happens
  // that an unseen phone has the highest-numbered pdf, this might be different.
  int32_t NumPdfs() const override { return num_pdfs_; }

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

  // returns the self-loop transition-id, or zero if
  // this state doesn't have a self-loop.
  int32_t SelfLoopOf(int32_t trans_state) const;

  int32_t PairToTransitionId(int32_t trans_state, int32_t trans_index) const;

  /// Returns the total number of transition-states (note, these are one-based).
  int32_t NumTransitionStates() const { return tuples_.size(); }

  float GetTransitionLogProb(int32_t trans_id) const;

  void Check() const;

  int32_t TransitionIdToTransitionState(int32_t trans_id) const;

  /// Returns the number of transition-indices for a particular
  /// transition-state. Note: "Indices" is the plural of "index".   Index is not
  /// the same as "id", here.  A transition-index is a zero-based offset into
  /// the transitions out of a particular transition state.
  int32_t NumTransitionIndices(int32_t trans_state) const;

  int32_t TupleToTransitionState(int32_t phone, int32_t hmm_state, int32_t pdf,
                                 int32_t self_loop_pdf) const;

  int32_t TransitionStateToSelfLoopPdf(int32_t trans_state) const;

  int32_t TransitionStateToForwardPdf(int32_t trans_state) const;

  int32_t TransitionStateToHmmState(int32_t trans_state) const;

  int32_t TransitionStateToPhone(int32_t trans_state) const;

  int32_t TransitionIdToTransitionIndex(int32_t trans_id) const;

  int32_t TransitionIdToHmmState(int32_t trans_id) const;

 private:
  struct Tuple {
    int32_t phone;
    int32_t hmm_state;
    int32_t forward_pdf;
    int32_t self_loop_pdf;
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

  /// Tuples indexed by transition state minus one;
  /// the tuples are in sorted order which allows us to do the reverse mapping
  /// from tuple to transition state
  std::vector<Tuple> tuples_;

  HmmTopology topo_;

  /// Gives the first transition_id of each transition-state; indexed by
  /// the transition-state.  Array indexed 1..num-transition-states+1 (the last
  /// one is needed so we can know the num-transitions of the last
  /// transition-state.
  std::vector<int32_t> state2id_;

  /// For each transition-id, the corresponding transition
  /// state (indexed by transition-id).
  std::vector<int32_t> id2state_;

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

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
