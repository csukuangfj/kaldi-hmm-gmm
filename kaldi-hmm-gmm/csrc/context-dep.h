// kaldi-hmm-gmm/csrc/context-dep.h
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CONTEXT_DEP_H_
#define KALDI_HMM_GMM_CSRC_CONTEXT_DEP_H_

// This file is copied/modified from kaldi/src/tree/context-dep.h

#include <unordered_set>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/event-map.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"

/**
 * \file context-dep.h
 * \brief Decision-tree based implementation of context dependency.
 *
 * This file is copied/modified from kaldi/src/tree/context-dep.h.
 *
 * It declares \c ContextDependency, the concrete, decision-tree based
 * implementation of \c ContextDependencyInterface (see context-dep-itf.h),
 * and the factory functions \c MonophoneContextDependency and
 * \c MonophoneContextDependencyShared that build trivial one-phone trees.
 * The decision-tree nodes themselves are \c EventMap objects (event-map.h);
 * this file only encodes the phone context as an event and queries the map.
 * Training of the tree is deliberately kept out of this file (see
 * build-tree-utils.h and cluster-utils.h for the tree-building side).
 */

namespace khg {

/**
 * \brief The "name" to which we assign the pdf-class.
 *
 * The pdf-class generally corresponds to position in the HMM, zero-based;
 * must not be used for any other event.  I.e. the value corresponding to
 * this key is the pdf-class (see hmm-topology.h for explanation of what this
 * is).
 */
static const EventKeyType kPdfClass = -1;

/**
 * \brief ContextDependency is quite a generic decision tree.
 *
 * It does not actually do very much-- all the magic is in the EventMap object.
 * All this class does is to encode the phone context as a sequence of events,
 * and pass this to the EventMap object to turn into what it will interpret as a
 * vector of pdfs.
 *
 * Different versions of the ContextDependency class that are written in the
 * future may have slightly different interfaces and pass more stuff in as
 * events, to the EventMap object.
 *
 * In order to separate the process of training decision trees from the process
 * of actually using them, we do not put any training code into the
 * ContextDependency class.
 */
class ContextDependency : public ContextDependencyInterface {
 public:
  ContextDependency(const ContextDependency &) = delete;
  ContextDependency &operator=(const ContextDependency &) = delete;

  int32_t ContextWidth() const override { return N_; }
  int32_t CentralPosition() const override { return P_; }

  /**
   * \brief Maps a phone context and pdf-class to a pdf-id.
   *
   * Returns success or failure; outputs pdf to pdf_id.  For positions that
   * were outside the sequence (due to end effects), put zero.  Naturally
   * \c phoneseq[CentralPosition()] must be nonzero.
   *
   * \param phoneseq   The context window of phones, of length N = \c N_; use
   *                   zero at positions outside the sequence (end effects).
   * \param pdf_class  The pdf-class (position within the HMM, zero-based).
   * \param pdf_id     [out] On success, the pdf-id.
   * \return  Returns success or failure; outputs the pdf-id.
   */
  bool Compute(const std::vector<int32_t> &phoneseq, int32_t pdf_class,
               int32_t *pdf_id) const override;

  /**
   * \brief Returns the number of acoustic pdfs (numbered 0..NumPdfs()-1).
   *
   * \return The number of pdfs; 0 if the to-pdf map is empty.
   */
  int32_t NumPdfs() const override {
    // this routine could be simplified to return to_pdf_->MaxResult()+1.  we're
    // a bit more paranoid than that.
    if (!to_pdf_) return 0;

    EventAnswerType max_result = to_pdf_->MaxResult();
    if (max_result < 0) {
      return 0;
    } else {
      return (int32_t)max_result + 1;
    }
  }

  /// \brief Returns a new object which is a deep copy of the current one.
  ///
  /// \return A newly allocated copy (including a copy of the to-pdf map);
  ///         the caller owns it.
  ContextDependencyInterface *Copy() const override {
    return new ContextDependency(N_, P_, to_pdf_->Copy());
  }

  /**
   * \brief Writes the context-dependency object to a stream.
   *
   * \param os      The stream to write to.
   * \param binary  Whether to write in binary (\c true) or text (\c false)
   *                form.
   */
  void Write(std::ostream &os, bool binary) const;

  /**
   * \brief Reads a context-dependency object from a stream; throws on error.
   *
   * \param is      The stream to read from.
   * \param binary  Whether the stream is in binary (\c true) or text
   *                (\c false) form.
   */
  void Read(std::istream &is, bool binary);

  /// \brief Constructor with no arguments; will normally be called prior to
  ///        Read().
  ContextDependency() : N_(0), P_(0), to_pdf_(nullptr) {}

  /**
   * \brief Constructor taking ownership of the supplied to-pdf map.
   *
   * \param N      The context width (number of phones considered).
   * \param P      The central position within the context window (0-based).
   * \param to_pdf The to-pdf \c EventMap; ownership is transferred to this
   *               object.
   */
  ContextDependency(int32_t N, int32_t P, EventMap *to_pdf)
      : N_(N), P_(P), to_pdf_(to_pdf) {}

  ~ContextDependency() { delete to_pdf_; }

  /// \brief Returns the underlying to-pdf \c EventMap (not owned by caller).
  const EventMap &ToPdfMap() const { return *to_pdf_; }

  /**
   * \brief Returns, for each pdf-id, the (phone, pdf-class) pairs it can
   *        correspond to.
   *
   * GetPdfInfo returns a vector indexed by pdf-id, saying for each pdf which
   * pairs of (phone, pdf-class) it can correspond to.  (Usually just one).
   * c.f. hmm/hmm-topology.h for meaning of pdf-class.
   * This is the old, simpler interface of GetPdfInfo(), and that this one can
   * only be called if the HmmTopology object's IsHmm() function call returns
   * true.
   *
   * \param phones            [in] List of phones.
   * \param num_pdf_classes   [in] Indexed by phone.
   * \param pdf_info          [out] Indexed by pdf-id; each element is the
   *                          list of (phone, pdf_class) pairs the pdf can
   *                          correspond to.
   */
  void GetPdfInfo(
      const std::vector<int32_t> &phones,           // list of phones
      const std::vector<int32_t> &num_pdf_classes,  // indexed by phone,
      std::vector<std::vector<std::pair<int32_t, int32_t>>> *pdf_info)
      const override;

  /**
   * \brief Returns, for each (phone, pdf-class pair), the possible
   *        (forward-transition-pdf, self-loop-pdf) id pairs.
   *
   * This function outputs information about what possible pdf-ids can
   * be generated for HMM-states; it covers the general case where
   * the self-loop pdf-class may be different from the forward-transition
   * pdf-class, so we are asking not about the set of possible pdf-ids
   * for a given (phone, pdf-class), but the set of possible ordered pairs
   * (forward-transition-pdf, self-loop-pdf) for a given (phone,
   * forward-transition-pdf-class, self-loop-pdf-class).
   *
   * \param phones           [in] A list of integer ids of phones.
   * \param pdf_class_pairs  [in] Indexed by phone; a list of pairs
   *                         (forward-transition-pdf-class,
   *                         self-loop-pdf-class) that we can have for that
   *                         phone.
   * \param pdf_info         [out] Indexed first by phone and then by the
   *                         same index that indexes each element of
   *                         \c pdf_class_pairs, and tells us for each pair in
   *                         \c pdf_class_pairs, what is the list of possible
   *                         (forward-transition-pdf-id, self-loop-pdf-id)
   *                         that we can have.
   *
   * This is less efficient than the other version of GetPdfInfo().
   * Note: if there is no self-loop, the corresponding entry (.second) in
   * \c pdf_class_pairs and the output \c pdf_info would be -1.
   */
  void GetPdfInfo(
      const std::vector<int32_t> &phones,
      const std::vector<std::vector<std::pair<int32_t, int32_t>>>
          &pdf_class_pairs,
      std::vector<std::vector<std::vector<std::pair<int32_t, int32_t>>>>
          *pdf_info) const override;

 private:
  int32_t N_;  //
  int32_t P_;
  EventMap *to_pdf_;  // owned here.

  /**
   * \brief Helper for GetPdfInfo(): enumerates allowed pdf-id pairs for one
   *        context.
   *
   * 'context' is the context-window of phones, of length N, with -1 for those
   * positions where phones that are currently unknown, treated as wildcards;
   * at least the central phone [position P] must be a real phone, i.e. not -1.
   * This function inserts any allowed pairs (forward_pdf, self_loop_pdf) to
   * the set "pairs".
   *
   * \param phones             [in] List of phones.
   * \param self_loop_pdf_class  [in] The pdf-class of the self-loop transition.
   * \param forward_pdf_class  [in] The pdf-class of the forward transition.
   * \param context            [in] The context window of phones (see above).
   * \param pairs              [out] Receives the allowed
   *                           (forward_pdf, self_loop_pdf) pairs.
   */
  void EnumeratePairs(const std::vector<int32_t> &phones,
                      int32_t self_loop_pdf_class, int32_t forward_pdf_class,
                      const std::vector<int32_t> &context,
                      std::unordered_set<std::pair<int32_t, int32_t>,
                                         PairHasher<int32_t>> *pairs) const;
};

/**
 * \brief Returns a new ContextDependency object for a monophone system.
 *
 * MonophoneContextDependency() returns a new ContextDependency object that
 * corresponds to a monophone system.
 * The map phone2num_pdf_classes maps from the phone id to the number of
 * pdf-classes we have for that phone (e.g. 3, so the pdf-classes would be
 * 0, 1, 2).
 *
 * \param phones               The list of phone ids to build the tree for.
 * \param phone2num_pdf_classes  Maps from the phone id to the number of
 *                             pdf-classes we have for that phone (e.g. 3, so
 *                             the pdf-classes would be 0, 1, 2).
 * \return  A newly allocated monophone \c ContextDependency; the caller owns
 *          it.
 */
ContextDependency *MonophoneContextDependency(
    const std::vector<int32_t> &phones,
    const std::vector<int32_t> &phone2num_pdf_classes);

/**
 * \brief As \c MonophoneContextDependency, but lets classes of phones share
 *        pdfs.
 *
 * MonophoneContextDependencyShared is as MonophoneContextDependency but lets
 * you define classes of phones which share pdfs (e.g. different stress-markers
 * of a single phone.)  Each element of phone_classes is a set of phones that
 * are in that class.
 *
 * \param phone_classes        Each element is a set of phones that are in that
 *                             class; phones within a class share pdfs.
 * \param phone2num_pdf_classes  Maps from the phone id to the number of
 *                             pdf-classes we have for that phone (e.g. 3, so
 *                             the pdf-classes would be 0, 1, 2).
 * \return  A newly allocated monophone \c ContextDependency with shared roots;
 *          the caller owns it.
 */
ContextDependency *MonophoneContextDependencyShared(
    const std::vector<std::vector<int32_t>> &phone_classes,
    const std::vector<int32_t> &phone2num_pdf_classes);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CONTEXT_DEP_H_
