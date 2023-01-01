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

}  // namespace khg
