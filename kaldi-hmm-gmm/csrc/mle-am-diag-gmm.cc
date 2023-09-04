// kaldi-hmm-gmm/csrc/mle-am-diag-gmm.cc
//
// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal);
//                      Microsoft Corporation;  Georg Stemmer;  Yanmin Qian
//                2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/stl-utils.h"
namespace khg {

AccumAmDiagGmm::~AccumAmDiagGmm() { DeletePointers(&gmm_accumulators_); }

void AccumAmDiagGmm::Init(const AmDiagGmm &model, GmmFlagsType flags) {
  DeletePointers(&gmm_accumulators_);  // in case was non-empty when called.

  gmm_accumulators_.resize(model.NumPdfs(), nullptr);
  for (int32_t i = 0; i < model.NumPdfs(); ++i) {
    gmm_accumulators_[i] = new AccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i), flags);
  }
}

void AccumAmDiagGmm::Init(const AmDiagGmm &model, int32_t dim,
                          GmmFlagsType flags) {
  KHG_ASSERT(dim > 0);
  DeletePointers(&gmm_accumulators_);  // in case was non-empty when called.

  gmm_accumulators_.resize(model.NumPdfs(), nullptr);
  for (int32_t i = 0; i < model.NumPdfs(); ++i) {
    gmm_accumulators_[i] = new AccumDiagGmm();
    gmm_accumulators_[i]->Resize(model.GetPdf(i).NumGauss(), dim, flags);
  }
}

void AccumAmDiagGmm::SetZero(GmmFlagsType flags) {
  for (size_t i = 0; i < gmm_accumulators_.size(); ++i) {
    gmm_accumulators_[i]->SetZero(flags);
  }
}

float AccumAmDiagGmm::AccumulateForGmm(const AmDiagGmm &model,
                                       const FloatVector &data,
                                       int32_t gmm_index, float weight) {
  KHG_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());

  float log_like = gmm_accumulators_[gmm_index]->AccumulateFromDiag(
      model.GetPdf(gmm_index), data, weight);

  total_log_like_ += log_like * weight;
  total_frames_ += weight;
  return log_like;
}

float AccumAmDiagGmm::AccumulateForGmmTwofeats(const AmDiagGmm &model,
                                               const FloatVector &data1,
                                               const FloatVector &data2,
                                               int32_t gmm_index,
                                               float weight) {
  KHG_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());

  const DiagGmm &gmm = model.GetPdf(gmm_index);

  AccumDiagGmm &acc = *(gmm_accumulators_[gmm_index]);
  FloatVector posteriors;
  float log_like = gmm.ComponentPosteriors(data1, &posteriors);

  posteriors *= weight;

  // TODO(fangjun): This looks strange since it uses posteriors from data1
  acc.AccumulateFromPosteriors(data2, posteriors);

  total_log_like_ += log_like * weight;
  total_frames_ += weight;

  return log_like;
}

void AccumAmDiagGmm::AccumulateFromPosteriors(const AmDiagGmm &model,
                                              const FloatVector &data,
                                              int32_t gmm_index,
                                              const FloatVector &posteriors) {
  KHG_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());

  gmm_accumulators_[gmm_index]->AccumulateFromPosteriors(data, posteriors);
  total_frames_ += posteriors.sum();
}

void AccumAmDiagGmm::AccumulateForGaussian(const AmDiagGmm &am,
                                           const FloatVector &data,
                                           int32_t gmm_index,
                                           int32_t gauss_index, float weight) {
  KHG_ASSERT(gmm_index >= 0 && gmm_index < NumAccs());

  KHG_ASSERT(gauss_index >= 0 && gauss_index < am.GetPdf(gmm_index).NumGauss());
  gmm_accumulators_[gmm_index]->AccumulateForComponent(data, gauss_index,
                                                       weight);
}

float AccumAmDiagGmm::TotStatsCount() const {
  double ans = 0.0;
  for (int32_t i = 0; i < NumAccs(); ++i) {
    const AccumDiagGmm &acc = GetAcc(i);
    ans += acc.occupancy().sum();
  }
  return ans;
}

const AccumDiagGmm &AccumAmDiagGmm::GetAcc(int32_t index) const {
  KHG_ASSERT(index >= 0 && index < NumAccs());

  return *(gmm_accumulators_[index]);
}

AccumDiagGmm &AccumAmDiagGmm::GetAcc(int32_t index) {
  KHG_ASSERT(index >= 0 && index < NumAccs());
  return *(gmm_accumulators_[index]);
}

void AccumAmDiagGmm::Add(float scale, const AccumAmDiagGmm &other) {
  total_frames_ += scale * other.total_frames_;
  total_log_like_ += scale * other.total_log_like_;

  int32_t num_accs = NumAccs();
  KHG_ASSERT(num_accs == other.NumAccs());

  for (int32_t i = 0; i < num_accs; ++i)
    gmm_accumulators_[i]->Add(scale, *(other.gmm_accumulators_[i]));
}

void AccumAmDiagGmm::Scale(float scale) {
  for (int32_t i = 0; i < NumAccs(); ++i) {
    AccumDiagGmm &acc = GetAcc(i);
    acc.Scale(scale, acc.Flags());
  }

  total_frames_ *= scale;
  total_log_like_ *= scale;
}

static void ResizeModel(int32_t dim, AmDiagGmm *am_gmm) {
  for (int32_t pdf_id = 0; pdf_id < am_gmm->NumPdfs(); ++pdf_id) {
    DiagGmm &pdf = am_gmm->GetPdf(pdf_id);
    pdf.Resize(pdf.NumGauss(), dim);

    FloatMatrix inv_vars = FloatMatrix::Ones(pdf.NumGauss(), dim);

    pdf.SetInvVars(inv_vars);

    pdf.ComputeGconsts();
  }
}

void MleAmDiagGmmUpdate(const MleDiagGmmOptions &config,
                        const AccumAmDiagGmm &am_diag_gmm_acc,
                        GmmFlagsType flags, AmDiagGmm *am_gmm,
                        float *obj_change_out, float *count_out) {
  if (am_diag_gmm_acc.Dim() != am_gmm->Dim()) {
    KHG_ASSERT(am_diag_gmm_acc.Dim() != 0);
    KHG_WARN << "Dimensions of accumulator " << am_diag_gmm_acc.Dim()
             << " and gmm " << am_gmm->Dim() << " do not match, resizing "
             << " GMM and setting to zero-mean, unit-variance.";
    ResizeModel(am_diag_gmm_acc.Dim(), am_gmm);
  }

  KHG_ASSERT(am_gmm != nullptr);
  KHG_ASSERT(am_diag_gmm_acc.NumAccs() == am_gmm->NumPdfs());
  if (obj_change_out != nullptr) *obj_change_out = 0.0;

  if (count_out != nullptr) *count_out = 0.0;

  float tot_obj_change = 0.0, tot_count = 0.0;

  int32_t tot_elems_floored = 0, tot_gauss_floored = 0, tot_gauss_removed = 0;

  for (int32_t i = 0; i < am_diag_gmm_acc.NumAccs(); ++i) {
    float obj_change, count;
    int32_t elems_floored, gauss_floored, gauss_removed;

    MleDiagGmmUpdate(config, am_diag_gmm_acc.GetAcc(i), flags,
                     &(am_gmm->GetPdf(i)), &obj_change, &count, &elems_floored,
                     &gauss_floored, &gauss_removed);

    tot_obj_change += obj_change;
    tot_count += count;
    tot_elems_floored += elems_floored;
    tot_gauss_floored += gauss_floored;
    tot_gauss_removed += gauss_removed;
  }
  if (obj_change_out != nullptr) *obj_change_out = tot_obj_change;

  if (count_out != nullptr) *count_out = tot_count;

  KHG_LOG << tot_elems_floored << " variance elements floored in "
          << tot_gauss_floored << " Gaussians, out of " << am_gmm->NumGauss();

  if (config.remove_low_count_gaussians) {
    KHG_LOG << "Removed " << tot_gauss_removed
            << " Gaussians due to counts < --min-gaussian-occupancy="
            << config.min_gaussian_occupancy
            << " and --remove-low-count-gaussians=true";
  }
}

void MapAmDiagGmmUpdate(const MapDiagGmmOptions &config,
                        const AccumAmDiagGmm &am_diag_gmm_acc,
                        GmmFlagsType flags, AmDiagGmm *am_gmm,
                        float *obj_change_out, float *count_out) {
  KHG_ASSERT(am_gmm != nullptr && am_diag_gmm_acc.Dim() == am_gmm->Dim() &&
             am_diag_gmm_acc.NumAccs() == am_gmm->NumPdfs());

  if (obj_change_out != nullptr) *obj_change_out = 0.0;

  if (count_out != nullptr) *count_out = 0.0;

  float tmp_obj_change, tmp_count;
  float *p_obj = (obj_change_out != nullptr) ? &tmp_obj_change : nullptr,
        *p_count = (count_out != nullptr) ? &tmp_count : nullptr;

  for (int32_t i = 0; i < am_diag_gmm_acc.NumAccs(); ++i) {
    MapDiagGmmUpdate(config, am_diag_gmm_acc.GetAcc(i), flags,
                     &(am_gmm->GetPdf(i)), p_obj, p_count);

    if (obj_change_out != nullptr) *obj_change_out += tmp_obj_change;

    if (count_out != nullptr) *count_out += tmp_count;
  }
}

}  // namespace khg
