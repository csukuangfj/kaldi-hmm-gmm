// kaldi-hmm-gmm/csrc/transition-model.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
#define KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_

// this if is copied and modified from
// kaldi/src/hmm/transition-model.h
//
#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/transition-information.h"

namespace khg {

class TransitionModel : public TransitionInformation {
 public:
  /// Initialize the object [e.g. at the start of training].
  /// The class keeps a copy of the HmmTopology object, but not
  /// the ContextDependency object.
  TransitionModel(const ContextDependencyInterface &ctx_dep,
                  const HmmTopology &hmm_topo);

 private:
  // called from constructor.  initializes tuples_.
  void ComputeTuples(const ContextDependencyInterface &ctx_dep);
  bool IsHmm() const;

  void ComputeTuplesIsHmm(const ContextDependencyInterface &ctx_dep);
  void ComputeTuplesNotHmm(const ContextDependencyInterface &ctx_dep);

  void ComputeDerived();  // called from constructor and Read function: computes
                          // state2id_ and id2state_.

  // return true if this trans_id corresponds to a self-loop.
  bool IsSelfLoop(int32_t trans_id) const;

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
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TRANSITION_MODEL_H_
