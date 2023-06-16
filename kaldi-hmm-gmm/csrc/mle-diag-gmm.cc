// kaldi-hmm-gmm/csrc/mle-diag-gmm.cc
//
// Copyright 2009-2013  Saarland University;  Georg Stemmer;  Jan Silovsky;
//                      Microsoft Corporation; Yanmin Qian;
//                      Johns Hopkins University (author: Daniel Povey);
//                      Cisco Systems (author: Neha Agrawal)

#include "kaldi-hmm-gmm/csrc/mle-diag-gmm.h"

#include <sstream>

#include "kaldi-hmm-gmm/csrc/diag-gmm-normal.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/utils.h"
namespace khg {

std::string MleDiagGmmOptions::ToString() const {
  std::ostringstream os;
  os << "MleDiagGmmOptions(";
  os << "min_gaussian_weight=" << min_gaussian_weight << ", ";
  os << "min_gaussian_occupancy=" << min_gaussian_occupancy << ", ";
  os << "min_variance=" << min_variance << ", ";
  os << "remove_low_count_gaussians="
     << (remove_low_count_gaussians ? "True" : "False") << ")";

  return os.str();
}

std::string MapDiagGmmOptions::ToString() const {
  std::ostringstream os;
  os << "MapDiagGmmOptions(";
  os << "mean_tau=" << mean_tau << ", ";
  os << "variance_tau=" << variance_tau << ", ";
  os << "weight_tau=" << weight_tau << ")";

  return os.str();
}

// num_comp is also called num_gauss
void AccumDiagGmm::Resize(int32_t num_comp, int32_t dim, GmmFlagsType flags) {
  KHG_ASSERT(num_comp > 0 && dim > 0);
  num_comp_ = num_comp;
  dim_ = dim;
  flags_ = AugmentGmmFlags(flags);
  occupancy_ = torch::zeros({num_comp}, torch::kDouble);

  if (flags_ & kGmmMeans) {
    mean_accumulator_ = torch::zeros({num_comp, dim}, torch::kDouble);
  } else {
    mean_accumulator_ = torch::Tensor();
  }

  if (flags_ & kGmmVariances) {
    variance_accumulator_ = torch::zeros({num_comp, dim}, torch::kDouble);
  } else {
    variance_accumulator_ = torch::Tensor();
  }
}

void AccumDiagGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KHG_ERR << "Flags in argument do not match the active accumulators";

  if (flags & kGmmWeights) occupancy_.zero_();

  if (flags & kGmmMeans) mean_accumulator_.zero_();

  if (flags & kGmmVariances) variance_accumulator_.zero_();
}

void AccumDiagGmm::Scale(float f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KHG_ERR << "Flags in argument do not match the active accumulators";

  double d = f;

  if (flags & kGmmWeights) occupancy_.mul_(d);

  if (flags & kGmmMeans) mean_accumulator_.mul_(d);

  if (flags & kGmmVariances) variance_accumulator_.mul_(d);
}

void AccumDiagGmm::AccumulateForComponent(torch::Tensor data,
                                          int32_t comp_index, float weight) {
  if (flags_ & kGmmMeans) KHG_ASSERT(data.size(0) == Dim());

  double wt = weight;

  KHG_ASSERT(comp_index < NumGauss());

  auto occupancy_acc = occupancy_.accessor<double, 1>();

  // accumulate
  occupancy_acc[comp_index] += wt;

  if (flags_ & kGmmMeans) {
    Row(mean_accumulator_, comp_index).add_(data, /*alpha*/ wt);

    if (flags_ & kGmmVariances) {
      Row(variance_accumulator_, comp_index).add_(data.square(), /*alpha*/ wt);
    }
  }
}

void AccumDiagGmm::AccumulateFromPosteriors(torch::Tensor data,
                                            torch::Tensor posteriors) {
  if (flags_ & kGmmMeans)
    KHG_ASSERT(static_cast<int32_t>(data.size(0)) == Dim());

  KHG_ASSERT(static_cast<int32_t>(posteriors.size(0)) == NumGauss());

  // accumulate
  occupancy_.add_(posteriors);

  if (flags_ & kGmmMeans) {
    posteriors = posteriors.unsqueeze(1);
    mean_accumulator_.add_(posteriors.mul(data));

    if (flags_ & kGmmVariances) {
      variance_accumulator_.add_(posteriors.mul(data.square()));
    }
  }
}

float AccumDiagGmm::AccumulateFromDiag(const DiagGmm &gmm, torch::Tensor data,
                                       float frame_posterior) {
  KHG_ASSERT(gmm.NumGauss() == NumGauss());
  KHG_ASSERT(gmm.Dim() == Dim());
  KHG_ASSERT(static_cast<int32_t>(data.size(0)) == Dim());

  torch::Tensor posteriors;
  float log_like = gmm.ComponentPosteriors(data, &posteriors);
  posteriors.mul_(frame_posterior);

  AccumulateFromPosteriors(data, posteriors);
  return log_like;
}

void AccumDiagGmm::AddStatsForComponent(int32_t g, double occ,
                                        torch::Tensor x_stats,
                                        torch::Tensor x2_stats) {
  KHG_ASSERT(g < NumGauss());

  auto occupancy_acc = occupancy_.accessor<double, 1>();

  occupancy_acc[g] += occ;

  if (flags_ & kGmmMeans) {
    Row(mean_accumulator_, g).add_(x_stats);
  }

  if (flags_ & kGmmVariances) {
    Row(variance_accumulator_, g).add_(x2_stats);
  }
}

void AccumDiagGmm::Add(double scale, const AccumDiagGmm &acc) {
  // The functions called here will crash if the dimensions etc.
  // or the flags don't match.
  occupancy_.add_(acc.occupancy_, /*alpha*/ scale);

  if (flags_ & kGmmMeans) {
    mean_accumulator_.add_(acc.mean_accumulator_, scale);
  }

  if (flags_ & kGmmVariances) {
    variance_accumulator_.add_(acc.variance_accumulator_, scale);
  }
}

// Careful: this wouldn't be valid if it were used to update the
// Gaussian weights.
void AccumDiagGmm::SmoothStats(float tau) {
  torch::Tensor smoothing_vec = (occupancy_ + tau) / occupancy_;
  smoothing_vec = smoothing_vec.unsqueeze(1);

  mean_accumulator_.mul_(smoothing_vec);
  variance_accumulator_.mul_(smoothing_vec);

  occupancy_.add_(tau);
}

// want to add tau "virtual counts" of each Gaussian from "src_acc"
// to each Gaussian in this acc.
// Careful: this wouldn't be valid if it were used to update the
// Gaussian weights.
void AccumDiagGmm::SmoothWithAccum(float tau, const AccumDiagGmm &src_acc) {
  KHG_ASSERT(src_acc.NumGauss() == num_comp_ && src_acc.Dim() == dim_);

  auto src_occupancy_acc = src_acc.occupancy_.accessor<double, 1>();
  auto occupancy_acc = occupancy_.accessor<double, 1>();

  for (int32_t i = 0; i < num_comp_; i++) {
    if (src_occupancy_acc[i] != 0.0) {  // can only smooth if src was nonzero...
      occupancy_acc[i] += tau;

      Row(mean_accumulator_, i)
          .add_(Row(src_acc.mean_accumulator_, i), tau / src_occupancy_acc[i]);

      Row(variance_accumulator_, i)
          .add_(Row(src_acc.variance_accumulator_, i),
                tau / src_occupancy_acc[i]);

    } else
      KHG_WARN << "Could not smooth since source acc had zero occupancy.";
  }
}

void AccumDiagGmm::SmoothWithModel(float tau, const DiagGmm &gmm) {
  KHG_ASSERT(gmm.NumGauss() == num_comp_ && gmm.Dim() == dim_);

  torch::Tensor means = gmm.GetMeans();
  torch::Tensor vars = gmm.GetVars();

  mean_accumulator_.add_(means, tau);

  variance_accumulator_.add_(vars + means.square(), tau);

  occupancy_.add_(tau);
}

void MleDiagGmmUpdate(const MleDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc, GmmFlagsType flags,
                      DiagGmm *gmm, float *obj_change_out, float *count_out,
                      int32_t *floored_elements_out,
                      int32_t *floored_gaussians_out,
                      int32_t *removed_gaussians_out) {
  KHG_ASSERT(gmm != nullptr);

  if (flags & ~diag_gmm_acc.Flags())
    KHG_ERR << "Flags in argument do not match the active accumulators";

  KHG_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
             diag_gmm_acc.Dim() == gmm->Dim());

  int32_t num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().sum().item().toDouble();

  int32_t elements_floored = 0, gauss_floored = 0;

  // remember old objective value
  gmm->ComputeGconsts();
  float obj_old = MlObjective(*gmm, diag_gmm_acc);

  // First get the gmm in "normal" representation (not the exponential-model
  // form).
  DiagGmmNormal ngmm(*gmm);

  auto ngmm_weights_acc = ngmm.weights_.accessor<double, 1>();

  auto diag_gmm_occupancy_acc = diag_gmm_acc.occupancy().accessor<double, 1>();

  std::vector<int32_t> to_remove;
  for (int32_t i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_occupancy_acc[i];
    double prob;
    if (occ_sum > 0.0) {
      prob = occ / occ_sum;
    } else {
      prob = 1.0 / num_gauss;
    }

    if (occ > config.min_gaussian_occupancy &&
        prob > config.min_gaussian_weight) {
      ngmm_weights_acc[i] = prob;

      // copy old mean for later normalizations
      torch::Tensor old_mean = Row(ngmm.means_, i).clone();

      // update mean, then variance, as far as there are accumulators
      if (diag_gmm_acc.Flags() & (kGmmMeans | kGmmVariances)) {
        Row(ngmm.means_, i) = Row(diag_gmm_acc.mean_accumulator(), i) / occ;
      }

      if (diag_gmm_acc.Flags() & kGmmVariances) {
        KHG_ASSERT(diag_gmm_acc.Flags() & kGmmMeans);

        torch::Tensor var = Row(diag_gmm_acc.variance_accumulator(), i) / occ;

        var.sub_(Row(ngmm.means_, i).square());  // subtract squared means.

        // if we intend to only update the variances, we need to compensate by
        // adding the difference between the new and old mean
        if (!(flags & kGmmMeans)) {
          old_mean.sub_(Row(ngmm.means_, i));

          var.add_(old_mean.square());
        }

        int32_t floored = 0;

        if (config.variance_floor_vector.defined()) {
          auto variance_floor_acc =
              config.variance_floor_vector.accessor<double, 1>();
          auto var_acc = var.accessor<double, 1>();
          int32_t n = var.size(0);

          for (int32_t k = 0; k != n; ++k) {
            if (var_acc[k] < variance_floor_acc[k]) {
              var_acc[k] = variance_floor_acc[k];
              ++floored;
            }
          }
        } else {
          auto var_acc = var.accessor<double, 1>();
          int32_t n = var.size(0);
          for (int32_t k = 0; k != n; ++k) {
            if (var_acc[k] < config.min_variance) {
              var_acc[k] = config.min_variance;
              ++floored;
            }
          }
        }

        if (floored != 0) {
          elements_floored += floored;
          ++gauss_floored;
        }
        // transfer to estimate
        Row(ngmm.vars_, i) = var;
      }
    } else {  // Insufficient occupancy.
      if (config.remove_low_count_gaussians &&
          static_cast<int32_t>(to_remove.size()) < num_gauss - 1) {
        // remove the component, unless it is the last one.
        KHG_WARN << "Too little data - removing Gaussian (weight " << std::fixed
                 << prob << ", occupation count " << std::fixed
                 << diag_gmm_occupancy_acc[i] << ", vector size " << gmm->Dim()
                 << ")";
        to_remove.push_back(i);
      } else {
        KHG_WARN << "Gaussian has too little data but not removing it because"
                 << (config.remove_low_count_gaussians
                         ? " it is the last Gaussian: i = "
                         : " remove-low-count-gaussians == false: g = ")
                 << i << ", occ = " << diag_gmm_occupancy_acc[i]
                 << ", weight = " << prob;
        ngmm_weights_acc[i] =
            std::max(prob, static_cast<double>(config.min_gaussian_weight));
      }
    }
  }

  // copy to natural representation according to flags
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  float obj_new = MlObjective(*gmm, diag_gmm_acc);

  if (obj_change_out) *obj_change_out = (obj_new - obj_old);
  if (count_out) *count_out = occ_sum;
  if (floored_elements_out) *floored_elements_out = elements_floored;
  if (floored_gaussians_out) *floored_gaussians_out = gauss_floored;

  if (to_remove.size() > 0) {
    gmm->RemoveComponents(to_remove, true /*renormalize weights*/);
    gmm->ComputeGconsts();
  }
  if (removed_gaussians_out != NULL) *removed_gaussians_out = to_remove.size();

  if (gauss_floored > 0)
    KHG_LOG << gauss_floored << " variances floored in " << gauss_floored
            << " Gaussians.";
}

void MapDiagGmmUpdate(const MapDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc, GmmFlagsType flags,
                      DiagGmm *gmm, float *obj_change_out, float *count_out) {
  KHG_ASSERT(gmm != NULL);

  if (flags & ~diag_gmm_acc.Flags()) {
    KHG_ERR << "Flags in argument do not match the active accumulators";
  }

  KHG_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
             diag_gmm_acc.Dim() == gmm->Dim());

  int32_t num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().sum().item().toDouble();

  // remember the old objective function value
  gmm->ComputeGconsts();
  float obj_old = MlObjective(*gmm, diag_gmm_acc);

  // allocate the gmm in normal representation; all parameters of this will be
  // updated, but only the flagged ones will be transferred back to gmm
  DiagGmmNormal ngmm(*gmm);

  auto diag_gmm_occupancy_acc = diag_gmm_acc.occupancy().accessor<double, 1>();
  auto ngmm_weights_acc = ngmm.weights_.accessor<double, 1>();

  for (int32_t i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_occupancy_acc[i];

    // First update the weight.  The weight_tau is a tau for the
    // whole state.
    ngmm_weights_acc[i] = (occ + ngmm_weights_acc[i] * config.weight_tau) /
                          (occ_sum + config.weight_tau);

    if (occ > 0.0 && (flags & kGmmMeans)) {
      // Update the Gaussian mean.
      torch::Tensor old_mean = Row(ngmm.means_, i);

      torch::Tensor mean = Row(diag_gmm_acc.mean_accumulator(), i)
                               .mul(1.0 / (occ + config.mean_tau));

      mean.add_(old_mean, /*alpha*/ config.mean_tau / (occ + config.mean_tau));

      Row(ngmm.means_, i) = mean;
    }

    if (occ > 0.0 && (flags & kGmmVariances)) {
      // Computing the variance around the updated mean; this is:
      // E( (x - mu)^2 ) = E( x^2 - 2 x mu + mu^2 ) =
      // E(x^2) + mu^2 - 2 mu E(x).
      torch::Tensor old_var = Row(ngmm.vars_, i);

      torch::Tensor var =
          Row(diag_gmm_acc.variance_accumulator(), i) / occ;  // E(x^2)

      var.add_(Row(ngmm.means_, i).square());  // mu^2

      torch::Tensor mean_acc =
          Row(diag_gmm_acc.mean_accumulator(), i);  // Sum(x)
      torch::Tensor mean = Row(ngmm.means_, i);     // mu

      var.add_(mean_acc * mean, -2.0 / occ);
      // now var is E(x^2) + mu^2 - 2 mu E(x).
      // Next we do the appropriate weighting using the tau value.
      var.mul_(occ / (config.variance_tau + occ));

      var.add_(old_var, config.variance_tau / (config.variance_tau + occ));

      // Now write to the model.
      Row(ngmm.vars_, i) = var;
    }
  }

  // Copy to natural/exponential representation.
  ngmm.CopyToDiagGmm(gmm, flags);

  gmm->ComputeGconsts();  // or MlObjective will fail.
  float obj_new = MlObjective(*gmm, diag_gmm_acc);

  if (obj_change_out) *obj_change_out = (obj_new - obj_old);

  if (count_out) *count_out = occ_sum;
}

float MlObjective(const DiagGmm &gmm, const AccumDiagGmm &diag_gmm_acc) {
  GmmFlagsType acc_flags = diag_gmm_acc.Flags();

  float obj = diag_gmm_acc.occupancy().vdot(gmm.gconsts()).item().toFloat();

  if (acc_flags & kGmmMeans) {
    // obj += TraceMatMat(mean_accs_bf, gmm.means_invvars(), kTrans);
    obj += (diag_gmm_acc.mean_accumulator() * gmm.means_invvars())
               .sum()
               .item()
               .toFloat();
  }

  if (acc_flags & kGmmVariances) {
    // obj -= 0.5 * TraceMatMat(variance_accs_bf, gmm.inv_vars(), kTrans);
    obj -= 0.5 * (diag_gmm_acc.variance_accumulator() * gmm.inv_vars())
                     .sum()
                     .item()
                     .toFloat();
  }

  return obj;
}

}  // namespace khg
