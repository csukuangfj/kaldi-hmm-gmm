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

  /// Initializes with a single "prototype" GMM.
  void Init(const DiagGmm &proto, int32_t num_pdfs);

  /// Adds a GMM to the model, and increments the total number of PDFs.
  void AddPdf(const DiagGmm &gmm);

 private:
  std::vector<DiagGmm *> densities_;

  void RemovePdf(int32_t pdf_index);
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_AM_DIAG_GMM_H_
