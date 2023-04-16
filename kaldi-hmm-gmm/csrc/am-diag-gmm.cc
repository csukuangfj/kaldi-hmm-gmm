// kaldi-hmm-gmm/csrc/am-diag-gmm.cc
//
// Copyright 2012   Arnab Ghoshal  Johns Hopkins University (Author: Daniel
// Povey)  Karel Vesely Copyright 2009-2011  Saarland University;  Microsoft
// Corporation;
//                      Georg Stemmer
//                2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/model-common.h"
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

int32_t AmDiagGmm::NumGauss() const {
  int32_t ans = 0;
  for (size_t i = 0; i < densities_.size(); i++)
    ans += densities_[i]->NumGauss();
  return ans;
}

void AmDiagGmm::SplitByCount(torch::Tensor state_occs,  // 1-D float tensor
                             int32_t target_components, float perturb_factor,
                             float power, float min_count) {
  int32_t gauss_at_start = NumGauss();
  std::vector<int32_t> targets;
  GetSplitTargets(state_occs, target_components, power, min_count, &targets);

  for (int32_t i = 0; i < NumPdfs(); ++i) {
    if (densities_[i]->NumGauss() < targets[i])
      densities_[i]->Split(targets[i], perturb_factor);
  }

  KHG_LOG << "Split " << NumPdfs()
          << " states with target = " << target_components
          << ", power = " << power << ", perturb_factor = " << perturb_factor
          << " and min_count = " << min_count << ", split #Gauss from "
          << gauss_at_start << " to " << NumGauss();
}

void AmDiagGmm::MergeByCount(torch::Tensor state_occs,
                             int32_t target_components, float power,
                             float min_count) {
  int32_t gauss_at_start = NumGauss();
  std::vector<int32_t> targets;
  GetSplitTargets(state_occs, target_components, power, min_count, &targets);

  for (int32_t i = 0; i < NumPdfs(); i++) {
    if (targets[i] == 0) targets[i] = 1;  // can't merge below 1.
    if (densities_[i]->NumGauss() > targets[i])
      densities_[i]->Merge(targets[i]);
  }

  KHG_LOG << "Merged " << NumPdfs()
          << " states with target = " << target_components
          << ", power = " << power << " and min_count = " << min_count
          << ", merged from " << gauss_at_start << " to " << NumGauss();
}

}  // namespace khg
