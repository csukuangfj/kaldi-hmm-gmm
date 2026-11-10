// kaldi-hmm-gmm/csrc/am-diag-gmm.h
//
// Copyright 2009-2012  Saarland University (Author:  Arnab Ghoshal)
//                      Johns Hopkins University (Author: Daniel Povey)
//                      Karel Vesely
//                2023  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/gmm/am-diag-gmm.h
//
#ifndef KALDI_HMM_GMM_CSRC_AM_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_AM_DIAG_GMM_H_

/**
 * \file am-diag-gmm.h
 * \brief Acoustic model of diagonal-covariance GMMs (\c AmDiagGmm).
 *
 * This file is copied/modified from ../kaldi/src/gmm/am-diag-gmm.h.
 *
 * It provides \c khg::AmDiagGmm, the acoustic model used for GMM training:
 * a collection of \c DiagGmm emission densities (see diag-gmm.h), one per
 * pdf, indexed by pdf-id.  The pdf-ids are the ids output by the Compute()
 * function of the phonetic decision tree (\c ContextDependency) and returned
 * by TransitionModel::TransitionIdToPdf(); they are zero-based and run from
 * 0 to NumPdfs()-1, exactly as in kaldi (kaldi's \c ali-to-pdf writes
 * zero-based pdf-ids).  Note that FST- and decoder-facing interfaces
 * sometimes number states from one for OpenFst compatibility; subtract one
 * before calling the methods here.
 *
 * The API falls into the following groups:
 *
 * - Construction: Init() fills the model with copies of one "prototype" GMM
 *   (one per pdf); AddPdf() appends a GMM one pdf at a time; the destructor
 *   frees the owned GMMs (copy construction and assignment are disabled).
 * - Model manipulation: SplitPdf() splits the components of a single pdf;
 *   SplitByCount()/MergeByCount() grow/shrink all pdfs towards a total
 *   Gaussian target according to a power-of-occupancy rule; CopyFromAmDiagGmm()
 *   deep-copies another model; ComputeGconsts() refreshes the cached
 *   log-normalizers of all pdfs.
 * - Scoring: LogLikelihood() evaluates one pdf on one frame.
 * - Parameter access: GetPdf() returns a pdf (mutable or const), the
 *   Get/SetGaussian* methods address individual Gaussian components.
 * - I/O: Read()/Write() use kaldi's on-disk format (the implementation is
 *   ported from ../kaldi/src/gmm/am-diag-gmm.cc and the file format is
 *   identical), so models are interchangeable with kaldi; a kaldi
 *   \c final.mdl stores this object together with a TransitionModel and a
 *   ContextDependency.
 *
 * Typical usage: Init() the model from a prototype (e.g. a single-Gaussian
 *   model) with one pdf per tree leaf, score frames with LogLikelihood(),
 *   accumulate statistics with AccumAmDiagGmm (mle-am-diag-gmm.h), and
 *   re-estimate with MleAmDiagGmmUpdate().
 */

#include <vector>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"

namespace khg {

/**
 * \brief Acoustic model: a collection of diagonal-covariance GMMs, one per
 *  pdf, indexed by pdf-id.
 *
 * Each pdf is a \c DiagGmm owned by this object; the pdf-ids are zero-based
 * indices into the internal list (see the file docs).  The model owns its
 * GMMs and frees them in the destructor; copying is disabled.
 */
class AmDiagGmm {
 public:
  /// Empty constructor.
  AmDiagGmm() = default;

  /// Destructor; frees the owned GMMs.
  ~AmDiagGmm();

  /// Copy construction is not allowed.
  AmDiagGmm(const AmDiagGmm &) = delete;

  /// Assignment is not allowed.
  AmDiagGmm &operator=(const AmDiagGmm &) = delete;

  /// \return The dimensionality of the feature vectors, taken from the first
  ///         pdf (0 if the model is empty).
  int32_t Dim() const {
    return (densities_.size() > 0) ? densities_[0]->Dim() : 0;
  }

  /// \return The number of pdfs (emission densities) in the model.
  int32_t NumPdfs() const { return densities_.size(); }

  /// \return The total number of Gaussian components over all pdfs.
  int32_t NumGauss() const;

  /// \param pdf_index  Zero-based pdf-id.
  /// \return The number of mixture components of the given pdf.
  int32_t NumGaussInPdf(int32_t pdf_index) const;

  /// Initializes with a single "prototype" GMM.
  /// \param proto  The prototype GMM; every pdf starts as a copy of it.
  /// \param num_pdfs  Number of pdfs to create (typically the number of tree
  ///                 leaves, i.e. TransitionModel::NumPdfs()).
  void Init(const DiagGmm &proto, int32_t num_pdfs);

  /// Adds a GMM to the model, and increments the total number of PDFs.
  /// \param gmm  The GMM to append as the next pdf; it is copied.  Its
  ///             dimension must match the existing pdfs (except for the
  ///             first one).
  void AddPdf(const DiagGmm &gmm);

  /// Copies the parameters from another model. Allocates necessary memory.
  /// \param other  The model to copy from.
  void CopyFromAmDiagGmm(const AmDiagGmm &other);

  /// Splits the components of a single pdf towards a target count.
  /// \param idx  Zero-based pdf-id of the pdf to split.
  /// \param target_components  Desired number of components of this pdf
  ///                           after splitting.
  /// \param perturb_factor  Scale of the random perturbation used to
  ///                        differentiate the split halves.
  void SplitPdf(int32_t idx, int32_t target_components, float perturb_factor);

  /// In SplitByCount we use the "target_components" and "power"
  /// to work out targets for each state (according to power-of-occupancy rule),
  /// and any state less than its target gets mixed up.  If some states
  /// were over their target, this may take the \c Gauss over the target.
  /// we enforce a min-count on Gaussians while splitting (don't split
  /// if it would take it below min-count).
  ///
  /// @param state_occs A 1-D float tensor of shape (num_pdfs,)
  /// @param target_components  Expected sum of number of gaussian in all pdfs
  /// @param perturb_factor to use when splitting a GMM
  /// @param power  It is used to compute state_occs.pow(power)
  /// @param min_count If the average of occupancy of gaussians in a pdf is less
  ///                  than this number, then we won't split this pdf any more
  void SplitByCount(const FloatVector &state_occs, int32_t target_components,
                    float perturb_factor, float power, float min_count);

  /// In MergeByCount we use the "target_components" and "power"
  /// to work out targets for each state (according to power-of-occupancy rule),
  /// and any state over its target gets mixed down.  If some states
  /// were under their target, this may take the \c Gauss below the target.
  /// @param state_occs  1-D float tensor of shape (num_pdfs,)
  /// @param target_components  Expected sum of number of gaussians in all pdfs
  /// @param power  It is used to compute state_occs.pow(power)
  /// @param min_count  The per-pdf target is capped so that each Gaussian
  ///                  retains at least this average occupancy (a LARGER
  ///                  value forces MORE merging, never less).
  void MergeByCount(const FloatVector &state_occs,
                    int32_t target_components, float power, float min_count);

  /// Sets the gconsts for all the PDFs. Returns the total number of Gaussians
  /// over all PDFs that are "invalid" e.g. due to zero weights or variances.
  /// \return The number of invalid Gaussians over all pdfs.
  int32_t ComputeGconsts() const;

  /// Computes the log-likelihood of a frame under one pdf of the model.
  /// @param pdf_index  Zero-based pdf-id.
  /// @param data 1-D float tensor of shape (dim,)
  /// @return Return the total loglike of the specified pdf
  float LogLikelihood(int32_t pdf_index, const FloatVector &data) const;

  /// Mutable accessor for a pdf.
  /// \param pdf_index  Zero-based pdf-id.
  /// \return Reference to the \c DiagGmm of the given pdf.
  DiagGmm &GetPdf(int32_t pdf_index);

  /// Const accessor for a pdf.
  /// \param pdf_index  Zero-based pdf-id.
  /// \return Const reference to the \c DiagGmm of the given pdf.
  const DiagGmm &GetPdf(int32_t pdf_index) const;

  /// Accessor for the mean of a single Gaussian component of a pdf.
  /// @param pdf_index  Zero-based pdf-id.
  /// @param gauss  Index of the mixture component within the pdf.
  /// @return Return a 1-D float tensor of shape (dim,)
  FloatVector GetGaussianMean(int32_t pdf_index, int32_t gauss) const;

  /// Accessor for the variance of a single Gaussian component of a pdf.
  /// @param pdf_index  Zero-based pdf-id.
  /// @param gauss  Index of the mixture component within the pdf.
  /// @return Return a 1-D float tensor of shape (dim,)
  FloatVector GetGaussianVariance(int32_t pdf_index, int32_t gauss) const;

  /// Mutators
  /// Sets the mean of a single Gaussian component of a pdf.
  /// \param pdf_index  Zero-based pdf-id.
  /// \param gauss_index  Index of the mixture component within the pdf.
  /// \param in  1-D float tensor of shape (dim,) with the new mean.
  void SetGaussianMean(int32_t pdf_index, int32_t gauss_index,
                       const FloatVector &in);

  /// Reads/writes in kaldi's format (ported from
  /// ../kaldi/src/gmm/am-diag-gmm.cc; file format is identical).
  /// \param is  The stream to read from / write to.
  /// \param binary  true for binary mode, false for text mode.
  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary) const;

 private:
  /// The GMMs, one per pdf, indexed by pdf-id.
  std::vector<DiagGmm *> densities_;

  /// Removes a single pdf and frees its GMM (used when shrinking the model).
  /// \param pdf_index  Zero-based pdf-id of the pdf to remove.
  void RemovePdf(int32_t pdf_index);
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_AM_DIAG_GMM_H_
