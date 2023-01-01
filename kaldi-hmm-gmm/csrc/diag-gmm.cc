// kaldi-hmm-gmm/csrc/diag-gmm.h

// Copyright 2009-2011  Microsoft Corporation;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Georg Stemmer;  Jan Silovsky
//           2012       Arnab Ghoshal
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
//                2023  Xiaomi Corporation

// this if is copied and modified from
// kaldi/src/gmm/diag-gmm.h

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

void DiagGmm::Resize(int32_t nmix, int32_t dim) {
  KHG_ASSERT(nmix > 0 && dim > 0);

  if (gconsts_.size(0) != nmix) {
    gconsts_ = torch::empty({nmix}, torch::kFloat);
  }

  if (weights_.size(0) != nmix) {
    weights_ = torch::empty({nmix}, torch::kFloat);
  }

  if (inv_vars_.size(0) != nmix || inv_vars_.size(1) != dim) {
    inv_vars_ = torch::full({nmix, dim}, 1.0f, torch::kFloat);
    // must be initialized to unit for case of calling SetMeans while having
    // covars/invcovars that are not set yet (i.e. zero)
  }

  if (means_invvars_.size(0) != nmix || means_invvars_.size(1) != dim) {
    means_invvars_ = torch::empty({nmix, dim}, torch::kFloat);
  }

  valid_gconsts_ = false;
}

void DiagGmm::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  Resize(diaggmm.weights_.numel(), diaggmm.means_invvars_.size(1));

  gconsts_ = diaggmm.gconsts_.clone();
  weights_ = diaggmm.weights_.clone();
  inv_vars_ = diaggmm.inv_vars_.clone();
  means_invvars_ = diaggmm.means_invvars_.clone();

  valid_gconsts_ = diaggmm.valid_gconsts_;
}

}  // namespace khg
