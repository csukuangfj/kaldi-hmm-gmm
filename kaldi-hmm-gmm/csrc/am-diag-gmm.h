// kaldi-hmm-gmm/csrc/am-diag-gmm.h
//
// Copyright 2009-2012  Saarland University (Author:  Arnab Ghoshal)
//                      Johns Hopkins University (Author: Daniel Povey)
//                      Karel Vesely
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_AM_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_AM_DIAG_GMM_H_

#include <vector>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"

namespace khg {

class AmDiagGmm {
 public:
  AmDiagGmm() = default;
  ~AmDiagGmm();

  AmDiagGmm(const AmDiagGmm &) = delete;
  AmDiagGmm &operator=(const AmDiagGmm &) = delete;

  int32_t Dim() const {
    return (densities_.size() > 0) ? densities_[0]->Dim() : 0;
  }

  int32_t NumPdfs() const { return densities_.size(); }

  int32_t NumGauss() const;

  int32_t NumGaussInPdf(int32_t pdf_index) const;

  /// Initializes with a single "prototype" GMM.
  void Init(const DiagGmm &proto, int32_t num_pdfs);

  /// Adds a GMM to the model, and increments the total number of PDFs.
  void AddPdf(const DiagGmm &gmm);

  /// Copies the parameters from another model. Allocates necessary memory.
  void CopyFromAmDiagGmm(const AmDiagGmm &other);

  void SplitPdf(int32_t idx, int32_t target_components, float perturb_factor);

  // In SplitByCount we use the "target_components" and "power"
  // to work out targets for each state (according to power-of-occupancy rule),
  // and any state less than its target gets mixed up.  If some states
  // were over their target, this may take the #Gauss over the target.
  // we enforce a min-count on Gaussians while splitting (don't split
  // if it would take it below min-count).
  //
  // @param state_occs A 1-D float tensor of shape (num_pdfs,)
  // @param target_components  Expected sum of number of gaussian in all pdfs
  // @param perturb_factor to use when splitting a GMM
  // @param power  It is used to compute state_occs.pow(power)
  // @param min_count If the average of occupancy of gaussians in a pdf is less
  //                  than this number, then we won't split this pdf any more
  void SplitByCount(const FloatVector &state_occs, int32_t target_components,
                    float perturb_factor, float power, float min_count);

  // In MergeByCount we use the "target_components" and "power"
  // to work out targets for each state (according to power-of-occupancy rule),
  // and any state over its target gets mixed down.  If some states
  // were under their target, this may take the #Gauss below the target.
  void MergeByCount(const FloatVector &state_occs,  // 1-D float tensor
                    int32_t target_components, float power, float min_count);

  /// Sets the gconsts for all the PDFs. Returns the total number of Gaussians
  /// over all PDFs that are "invalid" e.g. due to zero weights or variances.
  int32_t ComputeGconsts() const;

  // @param pdf_index
  // @param data 1-D float tensor
  // @return Return the total loglike of the specified pdf
  float LogLikelihood(int32_t pdf_index, const FloatVector &data) const;

  DiagGmm &GetPdf(int32_t pdf_index);
  const DiagGmm &GetPdf(int32_t pdf_index) const;

  // @param pdf_index
  // @param gauss
  // @return Return a 1-D float tensor
  FloatVector GetGaussianMean(int32_t pdf_index, int32_t gauss) const;

  // @param pdf_index
  // @param gauss
  // @return Return a 1-D float tensor
  FloatVector GetGaussianVariance(int32_t pdf_index, int32_t gauss) const;

  /// Mutators
  void SetGaussianMean(int32_t pdf_index, int32_t gauss_index,
                       const FloatVector &in);  // 1-D float tensor

 private:
  std::vector<DiagGmm *> densities_;

  void RemovePdf(int32_t pdf_index);
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_AM_DIAG_GMM_H_
