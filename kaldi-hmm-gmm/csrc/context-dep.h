// kaldi-hmm-gmm/csrc/context-dep.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CONTEXT_DEP_H_
#define KALDI_HMM_GMM_CSRC_CONTEXT_DEP_H_

// this file is copied and modified from
// kaldi/src/tree/context-dep.h

#include <unordered_set>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep-itf.h"
#include "kaldi-hmm-gmm/csrc/event-map.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"

namespace khg {

static const EventKeyType kPdfClass = -1;
// The "name" to which we assign the
// pdf-class (generally corresponds to position in the HMM, zero-based);
// must not be used for any other event.  I.e. the value corresponding to
// this key is the pdf-class (see hmm-topology.h for explanation of what this
// is).

/* ContextDependency is quite a generic decision tree.

   It does not actually do very much-- all the magic is in the EventMap object.
   All this class does is to encode the phone context as a sequence of events,
   and pass this to the EventMap object to turn into what it will interpret as a
   vector of pdfs.

   Different versions of the ContextDependency class that are written in the
   future may have slightly different interfaces and pass more stuff in as
   events, to the EventMap object.

   In order to separate the process of training decision trees from the process
   of actually using them, we do not put any training code into the
   ContextDependency class.
 */
class ContextDependency : public ContextDependencyInterface {
 public:
  ContextDependency(const ContextDependency &) = delete;
  ContextDependency &operator=(const ContextDependency &) = delete;

  int32_t ContextWidth() const override { return N_; }
  int32_t CentralPosition() const override { return P_; }

  /// returns success or failure; outputs pdf to pdf_id For positions that were
  /// outside the sequence (due to end effects), put zero.  Naturally
  /// phoneseq[CentralPosition()] must be nonzero.
  bool Compute(const std::vector<int32_t> &phoneseq, int32_t pdf_class,
               int32_t *pdf_id) const override;

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

  ContextDependencyInterface *Copy() const override {
    return new ContextDependency(N_, P_, to_pdf_->Copy());
  }

  void Write(std::ostream &os, bool binary) const;

  /// Read context-dependency object from disk; throws on error
  void Read(std::istream &is, bool binary);

  // Constructor with no arguments; will normally be called
  // prior to Read()
  ContextDependency() : N_(0), P_(0), to_pdf_(nullptr) {}

  // Constructor takes ownership of pointers.
  ContextDependency(int32_t N, int32_t P, EventMap *to_pdf)
      : N_(N), P_(P), to_pdf_(to_pdf) {}

  ~ContextDependency() { delete to_pdf_; }

  const EventMap &ToPdfMap() const { return *to_pdf_; }

  /// GetPdfInfo returns a vector indexed by pdf-id, saying for each pdf which
  /// pairs of (phone, pdf-class) it can correspond to.  (Usually just one).
  /// c.f. hmm/hmm-topology.h for meaning of pdf-class.
  /// This is the old, simpler interface of GetPdfInfo(), and that this one can
  /// only be called if the HmmTopology object's IsHmm() function call returns
  /// true.
  void GetPdfInfo(
      const std::vector<int32_t> &phones,           // list of phones
      const std::vector<int32_t> &num_pdf_classes,  // indexed by phone,
      std::vector<std::vector<std::pair<int32_t, int32_t>>> *pdf_info)
      const override;

  /// This function outputs information about what possible pdf-ids can
  /// be generated for HMM-states; it covers the general case where
  /// the self-loop pdf-class may be different from the forward-transition
  /// pdf-class, so we are asking not about the set of possible pdf-ids
  /// for a given (phone, pdf-class), but the set of possible ordered pairs
  /// (forward-transition-pdf, self-loop-pdf) for a given (phone,
  /// forward-transition-pdf-class, self-loop-pdf-class).
  /// Note: 'phones' is a list of integer ids of phones, and
  /// 'pdf-class-pairs', indexed by phone, is a list of pairs
  /// (forward-transition-pdf-class, self-loop-pdf-class) that we can have for
  /// that phone.
  /// The output 'pdf_info' is indexed first by phone and then by the
  /// same index that indexes each element of 'pdf_class_pairs',
  /// and tells us for each pair in 'pdf_class_pairs', what is the
  /// list of possible (forward-transition-pdf-id, self-loop-pdf-id) that
  /// we can have.
  /// This is less efficient than the other version of GetPdfInfo().
  /// Note: if there is no self-loop, the corresponding entry (.second) in
  /// pdf_class_pairs and the output pdf_info would be -1.
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

  // 'context' is the context-window of phones, of
  // length N, with -1 for those positions where phones
  // that are currently unknown, treated as wildcards; at least
  // the central phone [position P] must be a real phone, i.e.
  // not -1.
  // This function inserts any allowed pairs (forward_pdf, self_loop_pdf)
  // to the set "pairs".
  void EnumeratePairs(const std::vector<int32_t> &phones,
                      int32_t self_loop_pdf_class, int32_t forward_pdf_class,
                      const std::vector<int32_t> &context,
                      std::unordered_set<std::pair<int32_t, int32_t>,
                                         PairHasher<int32_t>> *pairs) const;
};

// MonophoneContextDependency() returns a new ContextDependency object that
// corresponds to a monophone system.
// The map phone2num_pdf_classes maps from the phone id to the number of
// pdf-classes we have for that phone (e.g. 3, so the pdf-classes would be
// 0, 1, 2).

ContextDependency *MonophoneContextDependency(
    const std::vector<int32_t> &phones,
    const std::vector<int32_t> &phone2num_pdf_classes);

// MonophoneContextDependencyShared is as MonophoneContextDependency but lets
// you define classes of phones which share pdfs (e.g. different stress-markers
// of a single phone.)  Each element of phone_classes is a set of phones that
// are in that class.
ContextDependency *MonophoneContextDependencyShared(
    const std::vector<std::vector<int32_t>> &phone_classes,
    const std::vector<int32_t> &phone2num_pdf_classes);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CONTEXT_DEP_H_
