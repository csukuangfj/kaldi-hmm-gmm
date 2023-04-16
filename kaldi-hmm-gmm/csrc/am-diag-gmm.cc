// kaldi-hmm-gmm/csrc/am-diag-gmm.cc
//
// Copyright 2012   Arnab Ghoshal  Johns Hopkins University (Author: Daniel
// Povey)  Karel Vesely Copyright 2009-2011  Saarland University;  Microsoft
// Corporation;
//                      Georg Stemmer
//                2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
namespace khg {

AmDiagGmm::~AmDiagGmm() { DeletePointers(&densities_); }

void AmDiagGmm::Init(const DiagGmm &proto, int32_t num_pdfs) {
  if (densities_.size() != 0) {
    KHG_WARN << "Init() called on a non-empty object. Contents will be "
                "overwritten";
    DeletePointers(&densities_);
  }
  if (num_pdfs == 0) {
    KHG_WARN << "Init() called with number of pdfs = 0. Will do nothing.";
    return;
  }

  densities_.resize(num_pdfs, nullptr);
  for (auto itr = densities_.begin(), end = densities_.end(); itr != end;
       ++itr) {
    *itr = new DiagGmm();
    (*itr)->CopyFromDiagGmm(proto);
  }
}

void AmDiagGmm::AddPdf(const DiagGmm &gmm) {
  if (densities_.size() != 0)  // not the first gmm
    KHG_ASSERT(gmm.Dim() == this->Dim());

  DiagGmm *gmm_ptr = new DiagGmm();
  gmm_ptr->CopyFromDiagGmm(gmm);
  densities_.push_back(gmm_ptr);
}

void AmDiagGmm::CopyFromAmDiagGmm(const AmDiagGmm &other) {
  if (densities_.size() != 0) {
    DeletePointers(&densities_);
  }

  densities_.resize(other.NumPdfs(), nullptr);
  for (int32_t i = 0, end = densities_.size(); i < end; ++i) {
    densities_[i] = new DiagGmm();
    densities_[i]->CopyFromDiagGmm(*other.densities_[i]);
  }
}

void AmDiagGmm::SplitPdf(int32_t pdf_index, int32_t target_components,
                         float perturb_factor) {
  KHG_ASSERT((static_cast<size_t>(pdf_index) < densities_.size()) &&
             (densities_[pdf_index] != nullptr));
  densities_[pdf_index]->Split(target_components, perturb_factor);
}

}  // namespace khg
