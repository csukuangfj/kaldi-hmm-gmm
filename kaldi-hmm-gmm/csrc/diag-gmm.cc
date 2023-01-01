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

// M_LOG_2PI =  log(2*pi)
#ifndef M_LOG_2PI
#define M_LOG_2PI 1.8378770664093454835606594728112
#endif

#define KALDI_ISINF std::isinf
#define KALDI_ISNAN std::isnan

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

// Constructor that allows us to merge GMMs.
// All GMMs must have the same dim.
DiagGmm::DiagGmm(const std::vector<std::pair<float, const DiagGmm *>> &gmms)
    : valid_gconsts_(false) {
  if (gmms.empty()) {
    return;  // GMM will be empty.
  } else {
    int32_t num_gauss = 0, dim = gmms[0].second->Dim();

    for (size_t i = 0; i < gmms.size(); i++) {
      num_gauss += gmms[i].second->NumGauss();
    }

    Resize(num_gauss, dim);

    int32_t cur_gauss = 0;
    for (size_t i = 0; i < gmms.size(); i++) {
      float weight = gmms[i].first;
      KHG_ASSERT(weight > 0.0);

      const DiagGmm &gmm = *(gmms[i].second);

      for (int32_t g = 0; g < gmm.NumGauss(); g++, cur_gauss++) {
        means_invvars_.slice(/*dim*/ 0, cur_gauss, cur_gauss + 1) =
            gmm.means_invvars_.slice(0, g, g + 1);
        inv_vars_.slice(0, cur_gauss, cur_gauss + 1) =
            gmm.inv_vars_.slice(0, g, g + 1);

        weights_.data_ptr<float>()[cur_gauss] =
            weight * gmm.weights_.data_ptr<float>()[g];
      }
    }
    KHG_ASSERT(cur_gauss == NumGauss());
    ComputeGconsts();
  }
}

int32_t DiagGmm::ComputeGconsts() {
  int32_t num_mix = NumGauss();
  int32_t dim = Dim();
  float offset = -0.5 * M_LOG_2PI * dim;  // constant term in gconst.
  int32_t num_bad = 0;

  // Resize if Gaussians have been removed during Update()
  if (num_mix != static_cast<int32_t>(gconsts_.size(0))) {
    gconsts_ = torch::empty({num_mix}, torch::kFloat);
  }

  auto gconsts_acc = gconsts_.accessor<float, 1>();
  auto weights_acc = weights_.accessor<float, 1>();
  auto inv_vars_acc = inv_vars_.accessor<float, 2>();
  auto means_invvars_acc = means_invvars_.accessor<float, 2>();

  for (int32_t mix = 0; mix < num_mix; mix++) {
    KHG_ASSERT(weights_acc[mix] >= 0);  // Cannot have negative weights.

    // May be -inf if weights == 0
    float gc = std::log(weights_acc[mix]) + offset;

    for (int32_t d = 0; d < dim; d++) {
      gc += 0.5 * std::log(inv_vars_acc[mix][d]) -
            0.5 * means_invvars_acc[mix][d] * means_invvars_acc[mix][d] /
                inv_vars_acc[mix][d];
    }
    // Change sign for logdet because var is inverted. Also, note that
    // mean_invvars(mix, d)*mean_invvars(mix, d)/inv_vars(mix, d) is the
    // mean-squared times inverse variance, since mean_invvars(mix, d) contains
    // the mean times inverse variance.
    // So gc is the likelihood at zero feature value.

    if (KALDI_ISNAN(gc)) {  // negative infinity is OK but NaN is not acceptable
      KHG_ERR << "At component " << mix
              << ", not a number in gconst computation";
    }
    if (KALDI_ISINF(gc)) {
      num_bad++;
      // If positive infinity, make it negative infinity.
      // Want to make sure the answer becomes -inf in the end, not NaN.
      if (gc > 0) gc = -gc;
    }
    gconsts_acc[mix] = gc;
  }

  valid_gconsts_ = true;
  return num_bad;
}

// Gets likelihood of data given this.
float DiagGmm::LogLikelihood(const torch::Tensor &data) const {
  if (!valid_gconsts_) {
    KHG_ERR << "Must call ComputeGconsts() before computing likelihood";
  }

  torch::Tensor loglikes;
  LogLikelihoods(data, &loglikes);

  float log_sum = loglikes.logsumexp(0).item().toFloat();

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum)) {
    KHG_ERR << "Invalid answer (overflow or invalid variances/features?)";
  }

  return log_sum;
}

void DiagGmm::LogLikelihoods(const torch::Tensor &data,
                             torch::Tensor *_loglikes) const {
  torch::Tensor loglikes = gconsts_.clone();

  if (data.numel() != Dim()) {
    KHG_ERR << "DiagGmm::LogLikelihoods, dimension "
            << "mismatch " << data.numel() << " vs. " << Dim();
  }
  torch::Tensor data_sq = data.pow(2);
  // means_invvars_: (nmix, dim)
  // data: (dim,)
  // loglikes: (nmix,)

  // loglikes +=  means * inv(vars) * data.
  // loglikes->AddMatVec(1.0, means_invvars_, kNoTrans, data, 1.0);
  loglikes = loglikes.unsqueeze(1);  // (nmix, 1)
  loglikes.addmm_(means_invvars_, data.unsqueeze(1));

  // loglikes += -0.5 * inv(vars) * data_sq.
  // loglikes->AddMatVec(-0.5, inv_vars_, kNoTrans, data_sq, 1.0);
  loglikes.addmm_(inv_vars_, data_sq.unsqueeze(1), /*beta*/ 1.0,
                  /*alpha*/ -0.5);

  *_loglikes = loglikes.squeeze(1);
}

void DiagGmm::LogLikelihoodsMatrix(const torch::Tensor &data,
                                   torch::Tensor *_loglikes) const {
  KHG_ASSERT(data.size(0) != 0);

  torch::Tensor loglikes = gconsts_.repeat({data.size(0), 1});

  if (data.size(1) != Dim()) {
    KHG_ERR << "DiagGmm::LogLikelihoods, dimension "
            << "mismatch " << data.size(1) << " vs. " << Dim();
  }
  torch::Tensor data_sq = data.pow(2);

  // loglikes +=  means * inv(vars) * data.
  // loglikes->AddMatMat(1.0, data, kNoTrans, means_invvars_, kTrans, 1.0);
  loglikes.addmm_(data, means_invvars_.t());

  // loglikes += -0.5 * inv(vars) * data_sq.
  // loglikes->AddMatMat(-0.5, data_sq, kNoTrans, inv_vars_, kTrans, 1.0);
  loglikes.addmm_(data_sq, inv_vars_.t(), /*beta*/ 1.0, /*alpha*/ -0.5);

  *_loglikes = loglikes;
}

void DiagGmm::LogLikelihoodsPreselect(const torch::Tensor &data,
                                      const std::vector<int32_t> &indices,
                                      torch::Tensor *_loglikes) const {
  KHG_ASSERT(data.size(0) == Dim());
  torch::Tensor data_sq = data.pow(2);

  torch::Tensor indexes = torch::tensor(indices, torch::kLong);
  torch::Tensor loglikes =
      gconsts_.index_select(/*dim*/ 0, indexes).unsqueeze(1);

  torch::Tensor means_invvars_sub = means_invvars_.index_select(0, indexes);
  torch::Tensor inv_vars_sub = inv_vars_.index_select(0, indexes);

  loglikes.addmm_(means_invvars_sub, data.unsqueeze(1));
  loglikes.addmm_(inv_vars_sub, data_sq.unsqueeze(1), /*beta*/ 1.0,
                  /*alpha*/ -0.5);
  *_loglikes = loglikes.squeeze(1);
}

}  // namespace khg
