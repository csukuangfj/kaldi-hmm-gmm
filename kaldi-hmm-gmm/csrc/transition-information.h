// kaldi-hmm-gmm/csrc/transition-information.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_TRANSITION_INFORMATION_H_
#define KALDI_HMM_GMM_CSRC_TRANSITION_INFORMATION_H_

// this if is copied and modified from
// kaldi/src/itf/transition-information.h

#include <vector>

namespace khg {

/**
 * Class that abstracts out TransitionModel's methods originally used
 * in the lat/ directory. By instantiating a subclass of this abstract
 * class other than TransitionModel, you can use kaldi's lattice tools
 * without a dependency on the hmm/ and tree/ directories. For
 * example, you can consider creating a subclass that implements these
 * via extracting information from Eesen's CTC T.fst object.
 *
 * TransitionId values must be contiguous, and starting from 1, rather
 * than 0, since 0 corresponds to epsilon in OpenFST.
 */
class TransitionInformation {
 public:
  virtual ~TransitionInformation() = default;
  /**
   * Returns true if trans_id1 and trans_id2 can correspond to the
   * same phone when trans_id1 immediately precedes trans_id2 (i.e.,
   * trans_id1 occurs at timestep t, and trans_id2 ocurs at timestep
   * 2) (possibly with epsilons between trans_id1 and trans_id2) OR
   * trans_id1 occurs before trans_id2, with some number of
   * trans_id_{k} values, all of which fulfill
   * TransitionIdsEquivalent(trans_id1, trans_id_{k})
   *
   * If trans_id1 == trans_id2, it must be the case that
   * TransitionIdsEquivalent(trans_id1, trans_id2) == true
   */
  virtual bool TransitionIdsEquivalent(int32_t trans_id1,
                                       int32_t trans_id2) const = 0;
  /**
   * Returns true if this trans_id corresponds to the start of a
   * phone.
   */
  virtual bool TransitionIdIsStartOfPhone(int32_t trans_id) const = 0;
  /**
   * Phone is a historical term, and should be understood in a wider
   * sense that also includes graphemes, word pieces, etc.: any
   * minimal entity in your problem domain which is represented by a
   * sequence of transitions with a PDF assigned to each of them by
   * the model. In this sense, Token is a better word. Since
   * TransitionInformation was added to subsume TransitionModel, we
   * did not want to change the call site of every
   * TransitionModel::TransitionIdToPhone to
   * TransitionInformation::TransitionIdToToken.
   */
  virtual int32_t TransitionIdToPhone(int32_t trans_id) const = 0;
  /**
   * Returns true if the destination of any edge with this trans_id
   * as its ilabel is a final state (or if a final state is
   * epsilon-reachable from its destination state).
   */
  virtual bool IsFinal(int32_t trans_id) const = 0;
  /**
   * Returns true if *all* of the FST edge labeled by this trans_id
   * have the same start and end states.
   */
  virtual bool IsSelfLoop(int32_t trans_id) const = 0;
  int32_t TransitionIdToPdf(int32_t trans_id) const {
    return TransitionIdToPdfArray()[trans_id];
  }
  /**
   * Returns the contiguous array that backs calls to
   * TransitionIdToPdf().
   *
   * Ideally, this would return a std::span, but it doesn't because
   * kaldi doesn't support C++20 at the time this interface was
   * written.
   */
  virtual const std::vector<int32_t> &TransitionIdToPdfArray() const = 0;
  int32_t NumTransitionIds() const {
    return TransitionIdToPdfArray().size() - 1;
  }
  /**
   * Return the number of distinct outputs from
   * TransitionIdToPdf(). Another way to look at this is as the number
   * of outputs over which your acoustic model does a softmax.
   */
  virtual int32_t NumPdfs() const = 0;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TRANSITION_INFORMATION_H_
