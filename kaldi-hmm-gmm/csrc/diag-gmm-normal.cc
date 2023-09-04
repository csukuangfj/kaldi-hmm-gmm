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
  weights_ = diaggmm.weights_.cast<double>();

  vars_ = 1.0f / diaggmm.inv_vars_.cast<double>().array();

  means_ = diaggmm.means_invvars_.cast<double>().array() * vars_.array();
}

void DiagGmmNormal::CopyToDiagGmm(DiagGmm *diaggmm, GmmFlagsType flags) const {
  KHG_ASSERT((diaggmm->Dim() == means_.cols()) &&
             (diaggmm->weights_.size() == weights_.size()));

  DiagGmmNormal oldg(*diaggmm);

  if (flags & kGmmWeights) {
    diaggmm->weights_ = weights_.cast<float>();
  }

  if (flags & kGmmVariances) {
    diaggmm->inv_vars_ = (1.0 / vars_.array()).cast<float>();

    // update the mean related natural part with the old mean, if necessary
    if (!(flags & kGmmMeans)) {
      diaggmm->means_invvars_ =
          oldg.means_.cast<float>().array() * diaggmm->inv_vars_.array();
    }
  }

  if (flags & kGmmMeans) {
    diaggmm->means_invvars_ =
        means_.cast<float>().array() * diaggmm->inv_vars_.array();
  }

  diaggmm->valid_gconsts_ = false;
}

}  // namespace khg
