// kaldi-hmm-gmm/csrc/mle-diag-gmm.cc
//
// Copyright 2009-2013  Saarland University;  Georg Stemmer;  Jan Silovsky;
//                      Microsoft Corporation; Yanmin Qian;
//                      Johns Hopkins University (author: Daniel Povey);
//                      Cisco Systems (author: Neha Agrawal)

#include "kaldi-hmm-gmm/csrc/mle-diag-gmm.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-hmm-gmm/csrc/diag-gmm-normal.h"
#include "kaldi-hmm-gmm/csrc/log.h"

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

  occupancy_ = DoubleVector::Zero(num_comp);

  if (flags_ & kGmmMeans) {
    mean_accumulator_ = DoubleMatrix::Zero(num_comp, dim);
  } else {
    mean_accumulator_ = DoubleMatrix();
  }

  if (flags_ & kGmmVariances) {
    variance_accumulator_ = DoubleMatrix::Zero(num_comp, dim);
  } else {
    variance_accumulator_ = DoubleMatrix();
  }
}

void AccumDiagGmm::SetZero(GmmFlagsType flags) {
  if (flags & ~flags_)
    KHG_ERR << "Flags in argument do not match the active accumulators";

  if (flags & kGmmWeights) {
    occupancy_.setZero();
  }

  if (flags & kGmmMeans) {
    mean_accumulator_.setZero();
  }

  if (flags & kGmmVariances) {
    variance_accumulator_.setZero();
  }
}

void AccumDiagGmm::Scale(float f, GmmFlagsType flags) {
  if (flags & ~flags_)
    KHG_ERR << "Flags in argument do not match the active accumulators";

  double d = f;

  if (flags & kGmmWeights) {
    occupancy_ *= d;
  }

  if (flags & kGmmMeans) {
    mean_accumulator_ *= d;
  }

  if (flags & kGmmVariances) {
    variance_accumulator_ *= d;
  }
}

void AccumDiagGmm::AccumulateForComponent(const FloatVector &data,
                                          int32_t comp_index, float weight) {
  if (flags_ & kGmmMeans) {
    KHG_ASSERT(data.size() == Dim());
  }

  double wt = weight;

  KHG_ASSERT(comp_index < NumGauss());

  // accumulate
  occupancy_[comp_index] += wt;

  if (flags_ & kGmmMeans) {
    mean_accumulator_.row(comp_index) += data.cast<double>().transpose() * wt;

    if (flags_ & kGmmVariances) {
      variance_accumulator_.row(comp_index) +=
          (data.array().square().matrix().transpose() * wt).cast<double>();
    }
  }
}

void AccumDiagGmm::AccumulateFromPosteriors(const FloatVector &data,
                                            const FloatVector &posteriors) {
  if (flags_ & kGmmMeans) {
    KHG_ASSERT(data.size() == Dim());
  }

  KHG_ASSERT(posteriors.size() == NumGauss());

  // accumulate
  occupancy_ += posteriors.cast<double>();

  if (flags_ & kGmmMeans) {
    mean_accumulator_ += (posteriors * data.transpose()).cast<double>();

    if (flags_ & kGmmVariances) {
      variance_accumulator_ +=
          (posteriors * data.array().square().matrix().transpose())
              .cast<double>();
    }
  }
}

float AccumDiagGmm::AccumulateFromDiag(const DiagGmm &gmm,
                                       const FloatVector &data, float weight) {
  KHG_ASSERT(gmm.NumGauss() == NumGauss());
  KHG_ASSERT(gmm.Dim() == Dim());
  KHG_ASSERT(data.size() == Dim());

  FloatVector posteriors;
  float log_like = gmm.ComponentPosteriors(data, &posteriors);
  posteriors *= weight;

  AccumulateFromPosteriors(data, posteriors);

  return log_like;
}

void AccumDiagGmm::AddStatsForComponent(int32_t g, double occ,
                                        const DoubleVector &x_stats,
                                        const DoubleVector &x2_stats) {
  KHG_ASSERT(g < NumGauss());

  occupancy_[g] += occ;

  if (flags_ & kGmmMeans) {
    mean_accumulator_.row(g) += x_stats.transpose();
  }

  if (flags_ & kGmmVariances) {
    variance_accumulator_.row(g) += x2_stats.transpose();
  }
}

void AccumDiagGmm::Add(float scale, const AccumDiagGmm &acc) {
  // The functions called here will crash if the dimensions etc.
  // or the flags don't match.
  occupancy_ += acc.occupancy_ * scale;

  if (flags_ & kGmmMeans) {
    mean_accumulator_ += acc.mean_accumulator_ * scale;
  }

  if (flags_ & kGmmVariances) {
    variance_accumulator_ += acc.variance_accumulator_ * scale;
  }
}

// Careful: this wouldn't be valid if it were used to update the
// Gaussian weights.
void AccumDiagGmm::SmoothStats(float tau) {
  DoubleVector smoothing_vec = (occupancy_.array() + tau) / occupancy_.array();

  mean_accumulator_ =
      mean_accumulator_.array() *
      smoothing_vec.replicate(1, mean_accumulator_.cols()).array();
  variance_accumulator_ =
      variance_accumulator_.array() *
      smoothing_vec.replicate(1, variance_accumulator_.cols()).array();

  occupancy_ = occupancy_.array() + tau;
}

// want to add tau "virtual counts" of each Gaussian from "src_acc"
// to each Gaussian in this acc.
// Careful: this wouldn't be valid if it were used to update the
// Gaussian weights.
void AccumDiagGmm::SmoothWithAccum(float tau, const AccumDiagGmm &src_acc) {
  KHG_ASSERT(src_acc.NumGauss() == num_comp_ && src_acc.Dim() == dim_);

  const auto &src_occupancy = src_acc.occupancy_;

  for (int32_t i = 0; i < num_comp_; i++) {
    if (src_occupancy[i] != 0.0) {  // can only smooth if src was nonzero...
      occupancy_[i] += tau;

      mean_accumulator_.row(i) +=
          src_acc.mean_accumulator_.row(i) * tau / src_occupancy[i];
      variance_accumulator_.row(i) +=
          src_acc.variance_accumulator_.row(i) * tau / src_occupancy[i];
    } else {
      KHG_WARN << "Could not smooth since source acc had zero occupancy.";
    }
  }
}

void AccumDiagGmm::SmoothWithModel(float tau, const DiagGmm &gmm) {
  KHG_ASSERT(gmm.NumGauss() == num_comp_ && gmm.Dim() == dim_);

  FloatMatrix means = gmm.GetMeans();
  FloatMatrix vars = gmm.GetVars();

  mean_accumulator_ += means.cast<double>() * tau;

  variance_accumulator_ +=
      (vars.cast<double>() + means.cast<double>().array().square().matrix()) *
      tau;

  occupancy_ = occupancy_.array() + tau;
}

void MleDiagGmmUpdate(const MleDiagGmmOptions &config,
                      const AccumDiagGmm &diag_gmm_acc, GmmFlagsType flags,
                      DiagGmm *gmm, float *obj_change_out, float *count_out,
                      int32_t *floored_elements_out,
                      int32_t *floored_gaussians_out,
                      int32_t *removed_gaussians_out) {
  KHG_ASSERT(gmm != nullptr);

  if (flags & ~diag_gmm_acc.Flags()) {
    KHG_ERR << "Flags in argument do not match the active accumulators";
  }

  KHG_ASSERT(diag_gmm_acc.NumGauss() == gmm->NumGauss() &&
             diag_gmm_acc.Dim() == gmm->Dim());

  int32_t num_gauss = gmm->NumGauss();
  double occ_sum = diag_gmm_acc.occupancy().sum();

  int32_t elements_floored = 0, gauss_floored = 0;

  // remember old objective value
  gmm->ComputeGconsts();
  float obj_old = MlObjective(*gmm, diag_gmm_acc);

  // First get the gmm in "normal" representation (not the exponential-model
  // form).
  DiagGmmNormal ngmm(*gmm);

  auto &ngmm_weights = ngmm.weights_;

  const auto &diag_gmm_occupancy = diag_gmm_acc.occupancy();

  std::vector<int32_t> to_remove;
  for (int32_t i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_occupancy[i];
    double prob;
    if (occ_sum > 0.0) {
      prob = occ / occ_sum;
    } else {
      prob = 1.0 / num_gauss;
    }

    if (occ > config.min_gaussian_occupancy &&
        prob > config.min_gaussian_weight) {
      ngmm_weights[i] = prob;

      // copy old mean for later normalizations
      DoubleVector old_mean = ngmm.means_.row(i);

      // update mean, then variance, as far as there are accumulators
      if (diag_gmm_acc.Flags() & (kGmmMeans | kGmmVariances)) {
        ngmm.means_.row(i) = diag_gmm_acc.mean_accumulator().row(i) / occ;
      }

      if (diag_gmm_acc.Flags() & kGmmVariances) {
        KHG_ASSERT(diag_gmm_acc.Flags() & kGmmMeans);

        DoubleVector var = diag_gmm_acc.variance_accumulator().row(i) / occ;

        var = var - ngmm.means_.row(i).array().square().matrix().transpose();

        // if we intend to only update the variances, we need to compensate by
        // adding the difference between the new and old mean
        if (!(flags & kGmmMeans)) {
          old_mean = old_mean - ngmm.means_.row(i).transpose();

          var = var.array() + old_mean.array().square();
        }

        int32_t floored = 0;

        if (config.variance_floor_vector.size()) {
          const auto &variance_floor = config.variance_floor_vector;

          int32_t n = var.size();

          for (int32_t k = 0; k != n; ++k) {
            if (var[k] < variance_floor[k]) {
              var[k] = variance_floor[k];
              ++floored;
            }
          }
        } else {
          int32_t n = var.size();
          for (int32_t k = 0; k != n; ++k) {
            if (var[k] < config.min_variance) {
              var[k] = config.min_variance;
              ++floored;
            }
          }
        }

        if (floored != 0) {
          elements_floored += floored;
          ++gauss_floored;
        }
        // transfer to estimate
        ngmm.vars_.row(i) = var.transpose();
      }
    } else {  // Insufficient occupancy.
      if (config.remove_low_count_gaussians &&
          static_cast<int32_t>(to_remove.size()) < num_gauss - 1) {
        // remove the component, unless it is the last one.
        KHG_WARN << "Too little data - removing Gaussian (weight " << std::fixed
                 << prob << ", occupation count " << std::fixed
                 << diag_gmm_occupancy[i] << ", vector size " << gmm->Dim()
                 << ")";
        to_remove.push_back(i);
      } else {
        KHG_WARN << "Gaussian has too little data but not removing it because"
                 << (config.remove_low_count_gaussians
                         ? " it is the last Gaussian: i = "
                         : " remove-low-count-gaussians == false: g = ")
                 << i << ", occ = " << diag_gmm_occupancy[i]
                 << ", weight = " << prob;
        ngmm_weights[i] =
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
  if (removed_gaussians_out != nullptr) {
    *removed_gaussians_out = to_remove.size();
  }

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

  double occ_sum = diag_gmm_acc.occupancy().sum();

  // remember the old objective function value
  gmm->ComputeGconsts();
  float obj_old = MlObjective(*gmm, diag_gmm_acc);

  // allocate the gmm in normal representation; all parameters of this will be
  // updated, but only the flagged ones will be transferred back to gmm
  DiagGmmNormal ngmm(*gmm);

  const auto diag_gmm_occupancy = diag_gmm_acc.occupancy();
  auto &ngmm_weights = ngmm.weights_;

  for (int32_t i = 0; i < num_gauss; i++) {
    double occ = diag_gmm_occupancy[i];

    // First update the weight.  The weight_tau is a tau for the
    // whole state.
    ngmm_weights[i] = (occ + ngmm_weights[i] * config.weight_tau) /
                      (occ_sum + config.weight_tau);

    if (occ > 0.0 && (flags & kGmmMeans)) {
      // Update the Gaussian mean.
      DoubleVector old_mean = ngmm.means_.row(i);

      DoubleVector mean = diag_gmm_acc.mean_accumulator().row(i) *
                          (1.0 / (occ + config.mean_tau));

      mean += old_mean * (config.mean_tau / (occ + config.mean_tau));

      ngmm.means_.row(i) = mean;
      // TODO(fangjun): optimize it
    }

    if (occ > 0.0 && (flags & kGmmVariances)) {
      // Computing the variance around the updated mean; this is:
      // E( (x - mu)^2 ) = E( x^2 - 2 x mu + mu^2 ) =
      // E(x^2) + mu^2 - 2 mu E(x).
      DoubleVector old_var = ngmm.vars_.row(i);

      DoubleVector var =
          diag_gmm_acc.variance_accumulator().row(i) / occ;  // E(x^2)

      var = var +
            ngmm.means_.row(i).array().square().matrix().transpose();  // mu^2

      DoubleVector mean_acc = diag_gmm_acc.mean_accumulator().row(i);  // Sum(x)

      DoubleVector mean = ngmm.means_.row(i);  // mu

      var = var.array() + mean_acc.array() * mean.array() * (-2.0 / occ);
      // now var is E(x^2) + mu^2 - 2 mu E(x).
      // Next we do the appropriate weighting using the tau value.
      var *= (occ / (config.variance_tau + occ));

      var += old_var * (config.variance_tau / (config.variance_tau + occ));

      // Now write to the model.
      ngmm.vars_.row(i) = var;
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

  float obj = diag_gmm_acc.occupancy().dot(gmm.gconsts().cast<double>());

  if (acc_flags & kGmmMeans) {
    // obj += TraceMatMat(mean_accs_bf, gmm.means_invvars(), kTrans);
    obj += (diag_gmm_acc.mean_accumulator().array() *
            gmm.means_invvars().cast<double>().array())
               .sum();
  }

  if (acc_flags & kGmmVariances) {
    // obj -= 0.5 * TraceMatMat(variance_accs_bf, gmm.inv_vars(), kTrans);
    obj -= 0.5 * (diag_gmm_acc.variance_accumulator().array() *
                  gmm.inv_vars().cast<double>().array())
                     .sum();
  }

  return obj;
}

}  // namespace khg
