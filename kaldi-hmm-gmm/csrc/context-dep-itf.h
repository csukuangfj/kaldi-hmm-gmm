// kaldi-hmm-gmm/csrc/context-dep-itf.h
//
// Copyright (c)  2022  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CONTEXT_DEP_ITF_H_
#define KALDI_HMM_GMM_CSRC_CONTEXT_DEP_ITF_H_
#include <cstdint>
#include <utility>
#include <vector>

// this file is copied and modified from
// kaldi/src/itf/context-dep-itf.h

namespace khg {

/// context-dep-itf.h provides a link between
/// the tree-building code in ../tree/, and the FST code in ../fstext/
/// (particularly, ../fstext/context-dep.h).  It is an abstract
/// interface that describes an object that can map from a
/// phone-in-context to a sequence of integer leaf-ids.
class ContextDependencyInterface {
 public:
  virtual ~ContextDependencyInterface() = default;
  ContextDependencyInterface() = default;
  ContextDependencyInterface(const ContextDependencyInterface &) = delete;
  ContextDependencyInterface &operator=(const ContextDependencyInterface &) =
      delete;

  /// ContextWidth() returns the value N (e.g. 3 for triphone models) that says
  /// how many phones are considered for computing context.
  virtual int ContextWidth() const = 0;

  /// Central position P of the phone context, in 0-based numbering, e.g. P = 1
  /// for typical triphone system.  We have to see if we can do without this
  /// function.
  virtual int CentralPosition() const = 0;

  /// The "new" Compute interface.  For typical topologies,
  /// pdf_class would be 0, 1, 2.
  /// Returns success or failure; outputs the pdf-id.
  ///
  /// "Compute" is the main function of this interface, that takes a
  /// sequence of N phones (and it must be N phones), possibly
  /// including epsilons (symbol id zero) but only at positions other
  /// than P [these represent unknown phone context due to end or
  /// begin of sequence].  We do not insist that Compute must always
  /// output (into stateseq) a nonempty sequence of states, but we
  /// anticipate that stateseq will always be nonempty at output in
  /// typical use cases.  "Compute" returns false if expansion somehow
  /// failed.  Normally the calling code should raise an exception if
  /// this happens.  We can define a different interface later in
  /// order to handle other kinds of information-- the underlying
  /// data-structures from event-map.h are very flexible.
  virtual bool Compute(const std::vector<int32_t> &phoneseq, int32_t pdf_class,
                       int32_t *pdf_id) const = 0;

  /// GetPdfInfo returns a vector indexed by pdf-id, saying for each pdf which
  /// pairs of (phone, pdf-class) it can correspond to.  (Usually just one).
  /// c.f. hmm/hmm-topology.h for meaning of pdf-class.
  /// This is the old, simpler interface of GetPdfInfo(), and that this one can
  /// only be called if the HmmTopology object's IsHmm() function call returns
  /// true.
  ///
  /// pdf_info is indexed by pdf_id
  /// pdf_info[i_th_pdf_id][k] is a pair: (phone_id, pdf_class)
  virtual void GetPdfInfo(
      const std::vector<int32_t> &phones,           // list of phones
      const std::vector<int32_t> &num_pdf_classes,  // indexed by phone,
      std::vector<std::vector<std::pair<int32_t, int32_t>>> *pdf_info)
      const = 0;

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
  virtual void GetPdfInfo(
      const std::vector<int32_t> &phones,
      const std::vector<std::vector<std::pair<int32_t, int32_t>>>
          &pdf_class_pairs,
      std::vector<std::vector<std::vector<std::pair<int32_t, int32_t>>>>
          *pdf_info) const = 0;

  /// NumPdfs() returns the number of acoustic pdfs (they are numbered 0..
  /// NumPdfs()-1).
  virtual int32_t NumPdfs() const = 0;

  /// Returns pointer to new object which is copy of current one.
  virtual ContextDependencyInterface *Copy() const = 0;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CONTEXT_DEP_ITF_H_
