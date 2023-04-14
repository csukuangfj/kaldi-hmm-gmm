// kaldi-hmm-gmm/csrc/diag-gmm-normal.cc
//
// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Yanmin Qian
//                2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/diag-gmm-normal.h"

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

void DiagGmmNormal::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  weights_ = diaggmm.weights_.clone();

  vars_ = 1.0f / diaggmm.inv_vars_;

  means_ = diaggmm.means_invvars_.mul(vars_);
}

void DiagGmmNormal::CopyToDiagGmm(DiagGmm *diaggmm, GmmFlagsType flags) const {
  KHG_ASSERT(
      (static_cast<int32_t>(diaggmm->Dim()) == means_.size(1)) &&
      (static_cast<int32_t>(diaggmm->weights_.size(0)) == weights_.size(0)));

  DiagGmmNormal oldg(*diaggmm);

  // weights_ is torch::kFloat; Converting it to kDouble will copy it
  if (flags & kGmmWeights) diaggmm->weights_ = weights_.to(torch::kDouble);

  if (flags & kGmmVariances) {
    diaggmm->inv_vars_ = (1.0 / vars_).to(torch::kDouble);

    // update the mean related natural part with the old mean, if necessary
    if (!(flags & kGmmMeans)) {
      diaggmm->means_invvars_ = oldg.means_.to(torch::kDouble);
      diaggmm->means_invvars_.mul_(diaggmm->inv_vars_);
    }
  }

  if (flags & kGmmMeans) {
    diaggmm->means_invvars_ = means_.to(torch::kDouble).mul(diaggmm->inv_vars_);
  }

  diaggmm->valid_gconsts_ = false;
}

}  // namespace khg
