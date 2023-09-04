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

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/clusterable-classes.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm-normal.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"

namespace khg {

void DiagGmm::Resize(int32_t nmix, int32_t dim) {
  KHG_ASSERT(nmix > 0 && dim > 0);

  if (!gconsts_.size() || gconsts_.size() != nmix) {
    gconsts_.resize(nmix);
  }

  if (!weights_.size() || weights_.size() != nmix) {
    weights_.resize(nmix);
  }

  if (!inv_vars_.size() || inv_vars_.rows() != nmix ||
      inv_vars_.cols() != dim) {
    inv_vars_ = FloatMatrix::Ones(nmix, dim);
    // must be initialized to unit for case of calling SetMeans while having
    // covars/invcovars that are not set yet (i.e. zero)
  }

  if (!means_invvars_.size() || means_invvars_.rows() != nmix ||
      means_invvars_.cols() != dim) {
    means_invvars_.resize(nmix, dim);
  }

  valid_gconsts_ = false;
}

void DiagGmm::CopyFromDiagGmm(const DiagGmm &diaggmm) {
  gconsts_ = diaggmm.gconsts_;  // copy semantics
  weights_ = diaggmm.weights_;
  inv_vars_ = diaggmm.inv_vars_;
  means_invvars_ = diaggmm.means_invvars_;

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
    for (size_t i = 0; i < gmms.size(); ++i) {
      float weight = gmms[i].first;
      KHG_ASSERT(weight > 0.0);

      const DiagGmm &gmm = *(gmms[i].second);

      means_invvars_(Eigen::seqN(cur_gauss, gmm.NumGauss()), Eigen::all) =
          gmm.means_invvars_;

      inv_vars_(Eigen::seqN(cur_gauss, gmm.NumGauss()), Eigen::all) =
          gmm.inv_vars_;

      weights_(Eigen::seqN(cur_gauss, gmm.NumGauss())) = weight * gmm.weights_;

      cur_gauss += gmm.NumGauss();
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
  if (!gconsts_.size() || num_mix != gconsts_.size()) {
    gconsts_.resize(num_mix);
  }

  for (int32_t mix = 0; mix < num_mix; ++mix) {
    KHG_ASSERT(weights_[mix] >= 0);  // Cannot have negative weights.

    // May be -inf if weights == 0
    float gc = std::log(weights_[mix]) + offset;

    // TODO(fangjun): Optimize it
    for (int32_t d = 0; d < dim; d++) {
      gc += 0.5 * std::log(inv_vars_(mix, d)) - 0.5 * means_invvars_(mix, d) *
                                                    means_invvars_(mix, d) /
                                                    inv_vars_(mix, d);
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
    gconsts_[mix] = gc;
  }

  valid_gconsts_ = true;
  return num_bad;
}

// Gets likelihood of data given this.
float DiagGmm::LogLikelihood(const FloatVector &data) const {
  if (!valid_gconsts_) {
    KHG_ERR << "Must call ComputeGconsts() before computing likelihood";
  }

  FloatVector loglikes;
  LogLikelihoods(data, &loglikes);

  float log_sum = LogSumExp(loglikes);

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum)) {
    KHG_ERR << "Invalid answer (overflow or invalid variances/features?)";
  }

  return log_sum;
}

void DiagGmm::LogLikelihoods(const FloatVector &data,
                             FloatVector *loglikes) const {
  if (data.size() != Dim()) {
    KHG_ERR << "DiagGmm::LogLikelihoods, dimension "
            << "mismatch " << data.size() << " vs. " << Dim();
  }

  *loglikes = gconsts_ + means_invvars_ * data -
              0.5 * inv_vars_ * data.array().square().matrix();
}
void DiagGmm::LogLikelihoodsMatrix(const FloatMatrix &data,
                                   FloatMatrix *_loglikes) const {
  KHG_ASSERT(data.rows() != 0);

  if (data.cols() != Dim()) {
    KHG_ERR << "DiagGmm::LogLikelihoods, dimension "
            << "mismatch " << data.cols() << " vs. " << Dim();
  }

  *_loglikes = gconsts_.transpose().replicate(data.rows(), 1) +
               data * means_invvars_.transpose() -
               0.5 * data.array().square().matrix() * inv_vars_.transpose();
}

void DiagGmm::LogLikelihoodsPreselect(const FloatVector &data,
                                      const std::vector<int32_t> &indices,
                                      FloatVector *_loglikes) const {
  KHG_ASSERT(data.size() == Dim());

  *_loglikes =
      gconsts_(indices) + means_invvars_(indices, Eigen::all) * data -
      0.5 * inv_vars_(indices, Eigen::all) * data.array().square().matrix();
}

/// Get gaussian selection information for one frame.
float DiagGmm::GaussianSelection(const FloatVector &data,  // 1-D tensor
                                 int32_t num_gselect,
                                 std::vector<int32_t> *output) const {
  int32_t num_gauss = NumGauss();
  output->clear();

  FloatVector loglikes;
  LogLikelihoods(data, &loglikes);

  float thresh;
  if (num_gselect < num_gauss) {
    FloatVector loglikes_copy = loglikes;
    float *ptr = &loglikes_copy[0];
    std::nth_element(ptr, ptr + num_gauss - num_gselect, ptr + num_gauss);
    thresh = ptr[num_gauss - num_gselect];
  } else {
    thresh = -std::numeric_limits<float>::infinity();
  }

  float tot_loglike = -std::numeric_limits<float>::infinity();

  std::vector<std::pair<float, int32_t>> pairs;
  for (int32_t p = 0; p < num_gauss; ++p) {
    if (loglikes[p] >= thresh) {
      pairs.push_back(std::make_pair(loglikes[p], p));
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            std::greater<std::pair<float, int32_t>>());

  for (int32_t j = 0; j < num_gselect && j < static_cast<int32_t>(pairs.size());
       ++j) {
    output->push_back(pairs[j].second);
    tot_loglike = LogAdd(tot_loglike, pairs[j].first);
  }
  KHG_ASSERT(!output->empty());
  return tot_loglike;
}

float DiagGmm::GaussianSelection(
    const FloatMatrix &data,  // 2-D tensor of shape (num_frames, dim)
    int32_t num_gselect, std::vector<std::vector<int32_t>> *output) const {
  double ans = 0.0;
  int32_t num_frames = data.rows(), num_gauss = NumGauss();

  int32_t max_mem = 10000000;  // Don't devote more than 10Mb to loglikes_mat;
                               // break up the utterance if needed.
  int32_t mem_needed = num_frames * num_gauss * sizeof(float);
  if (mem_needed > max_mem) {
    // Break into parts and recurse, we don't want to consume too
    // much memory.
    int32_t num_parts = (mem_needed + max_mem - 1) / max_mem;
    int32_t part_frames = (data.rows() + num_parts - 1) / num_parts;
    double tot_ans = 0.0;
    std::vector<std::vector<int32_t>> part_output;
    output->clear();
    output->resize(num_frames);
    for (int32_t p = 0; p < num_parts; p++) {
      int32_t start_frame = p * part_frames;
      int32_t this_num_frames = std::min(num_frames - start_frame, part_frames);

      FloatMatrix data_part =
          data(Eigen::seqN(start_frame, this_num_frames), Eigen::all);

      tot_ans += GaussianSelection(data_part, num_gselect, &part_output);

      for (int32_t t = 0; t < this_num_frames; t++)
        (*output)[start_frame + t].swap(part_output[t]);
    }
    KHG_ASSERT(!output->back().empty());
    return tot_ans;
  }

  KHG_ASSERT(num_frames != 0);

  FloatMatrix loglikes_mat;
  LogLikelihoodsMatrix(data, &loglikes_mat);

  output->clear();
  output->resize(num_frames);

  for (int32_t i = 0; i < num_frames; ++i) {
    FloatVector loglikes = loglikes_mat.row(i);

    float thresh;
    if (num_gselect < num_gauss) {
      FloatVector loglikes_copy = loglikes;
      float *ptr = &loglikes_copy[0];
      std::nth_element(ptr, ptr + num_gauss - num_gselect, ptr + num_gauss);
      thresh = ptr[num_gauss - num_gselect];
    } else {
      thresh = -std::numeric_limits<float>::infinity();
    }
    float tot_loglike = -std::numeric_limits<float>::infinity();

    std::vector<std::pair<float, int32_t>> pairs;
    for (int32_t p = 0; p < num_gauss; p++) {
      if (loglikes[p] >= thresh) {
        pairs.push_back(std::make_pair(loglikes[p], p));
      }
    }

    std::sort(pairs.begin(), pairs.end(),
              std::greater<std::pair<float, int32_t>>());

    std::vector<int32_t> &this_output = (*output)[i];
    for (int32_t j = 0;
         j < num_gselect && j < static_cast<int32_t>(pairs.size()); ++j) {
      this_output.push_back(pairs[j].second);
      tot_loglike = LogAdd(tot_loglike, pairs[j].first);
    }
    KHG_ASSERT(!this_output.empty());
    ans += tot_loglike;
  }
  return ans;
}

float DiagGmm::GaussianSelectionPreselect(const FloatVector &data,
                                          const std::vector<int32_t> &preselect,
                                          int32_t num_gselect,
                                          std::vector<int32_t> *output) const {
  static bool warned_size = false;
  int32_t preselect_sz = preselect.size();
  int32_t this_num_gselect = std::min(num_gselect, preselect_sz);
  if (preselect_sz <= num_gselect && !warned_size) {
    warned_size = true;
    KHG_WARN << "Preselect size is less than or equal to final size, "
             << "doing nothing: " << preselect_sz << " < " << num_gselect
             << " [won't warn again]";
  }

  FloatVector loglikes;
  LogLikelihoodsPreselect(data, preselect, &loglikes);

  FloatVector loglikes_copy = loglikes;
  float *ptr = &loglikes_copy[0];

  std::nth_element(ptr, ptr + preselect_sz - this_num_gselect,
                   ptr + preselect_sz);
  float thresh = ptr[preselect_sz - this_num_gselect];

  float tot_loglike = -std::numeric_limits<float>::infinity();

  // we want the output sorted from best likelihood to worse
  // (so we can prune further without the model)...
  std::vector<std::pair<float, int32_t>> pairs;
  for (int32_t p = 0; p < preselect_sz; p++)
    if (loglikes[p] >= thresh)
      pairs.push_back(std::make_pair(loglikes[p], preselect[p]));

  std::sort(pairs.begin(), pairs.end(),
            std::greater<std::pair<float, int32_t>>());

  output->clear();
  for (int32_t j = 0;
       j < this_num_gselect && j < static_cast<int32_t>(pairs.size()); j++) {
    output->push_back(pairs[j].second);
    tot_loglike = LogAdd(tot_loglike, pairs[j].first);
  }

  KHG_ASSERT(!output->empty());

  return tot_loglike;
}

// Gets likelihood of data given this. Also provides per-Gaussian posteriors.
float DiagGmm::ComponentPosteriors(const FloatVector &data,  // 1-D
                                   FloatVector *posterior    // 1-D
) const {
  if (!valid_gconsts_) {
    KHG_ERR << "Must call ComputeGconsts() before computing likelihood";
  }

  if (posterior == nullptr) {
    KHG_ERR << "NULL pointer passed as return argument.";
  }

  FloatVector loglikes;
  LogLikelihoods(data, &loglikes);

  float log_sum;
  FloatVector likes = Softmax(loglikes, &log_sum);

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum)) {
    KHG_ERR << "Invalid answer (overflow or invalid variances/features?)";
  }

  *posterior = std::move(likes);

  return log_sum;
}

float DiagGmm::ComponentLogLikelihood(const FloatVector &data,  // 1-D
                                      int32_t comp_id) const {
  if (!valid_gconsts_) {
    KHG_ERR << "Must call ComputeGconsts() before computing likelihood";
  }

  if (data.size() != Dim()) {
    KHG_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
            << "mismatch " << data.size() << " vs. " << Dim();
  }

  return means_invvars_.row(comp_id).dot(data) -
         0.5 * inv_vars_.row(comp_id).dot(data.array().square().matrix()) +
         gconsts_[comp_id];
}

void DiagGmm::Generate(FloatVector *output) const {
  KHG_ASSERT(static_cast<int32_t>(output->size()) == Dim());

  float tot = weights_.sum();

  KHG_ASSERT(tot > 0.0);

  double r = tot * Randn() * 0.99999;

  int32_t i = 0;
  double sum = 0.0;

  while (sum + weights_[i] < r && i < weights_.size()) {
    sum += weights_[i];
    ++i;
  }

  if (i >= weights_.size()) {
    i = weights_.size() - 1;
  }

  // now i is the index of the Gaussian we chose.

#if 0
  for (int32_t d = 0; d < inv_vars_.cols(); ++d) {
    float stddev = 1.0 / sqrt(inv_vars_(i, d));
    float mean = means_invvars_(i, d) / inv_vars_(i, d);

    (*output)[d] = mean + Randn() * stddev;
  }
#else
  auto t = inv_vars_.row(i);

  *output = means_invvars_.row(i).array() / t.array() +
            RandnVector(Dim()).transpose().array() / t.array().sqrt();
#endif
}

void DiagGmm::Perturb(float perturb_factor) {
  int32_t num_comps = NumGauss(), dim = Dim();

  // as in DiagGmm::Split, we perturb the means_invvars using a random
  // fraction of inv_vars_
  means_invvars_ =
      means_invvars_.array() + RandnMatrix(num_comps, dim).array() *
                                   inv_vars_.array().sqrt() * perturb_factor;

  ComputeGconsts();
}

void DiagGmm::Interpolate(float rho, const DiagGmm &source,
                          GmmFlagsType flags) {
  KHG_ASSERT(NumGauss() == source.NumGauss());
  KHG_ASSERT(Dim() == source.Dim());

  DiagGmmNormal us(*this);
  DiagGmmNormal them(source);

  if (flags & kGmmWeights) {
    us.weights_ = us.weights_ * (1.0 - rho) + them.weights_ * rho;

    us.weights_ /= us.weights_.sum();
  }

  if (flags & kGmmMeans) {
    us.means_ = us.means_ * (1.0 - rho) + them.means_ * rho;
  }

  if (flags & kGmmVariances) {
    us.vars_ = us.vars_ * (1.0 - rho) + them.vars_ * rho;
  }

  us.CopyToDiagGmm(this);
  ComputeGconsts();
}

void DiagGmm::MergeKmeans(
    int32_t target_components,
    const ClusterKMeansOptions &cfg /*= ClusterKMeansOptions()*/) {
  if (target_components <= 0 || NumGauss() < target_components) {
    KHG_ERR << "Invalid argument for target number of Gaussians (="
            << target_components << "), #Gauss = " << NumGauss();
  }

  if (NumGauss() == target_components) {
    KHG_LOG << "No components merged, as target (" << target_components
            << ") = total.";
    return;  // Nothing to do.
  }

  double min_var = 1.0e-10;
  std::vector<Clusterable *> clusterable_vec;

  for (int32_t g = 0; g < NumGauss(); ++g) {
    float count = weights_[g];

    if (count == 0) {
      KHG_WARN << "Not using zero-weight Gaussians in clustering.";
      continue;
    }

    FloatVector var = 1.0f / inv_vars_.row(g).array();
    FloatVector mean_invvar = means_invvars_.row(g);

    FloatVector x_stats = mean_invvar.array() * var.array();

    FloatVector x2_stats = (var.array() + x_stats.array().square()) * count;

    x_stats *= count;

    clusterable_vec.push_back(new GaussClusterable(
        x_stats.cast<double>(), x2_stats.cast<double>(), min_var, count));
  }

  if (clusterable_vec.size() <= target_components) {
    KHG_WARN << "Not doing clustering phase since lost too many Gaussians "
             << "due to zero weight. Warning: zero-weight Gaussians are "
             << "still there.";
    DeletePointers(&clusterable_vec);
    return;
  } else {
    std::vector<Clusterable *> clusters;
    ClusterKMeans(clusterable_vec, target_components, &clusters, nullptr, cfg);

    Resize(clusters.size(), Dim());

    for (int32_t g = 0; g < static_cast<int32_t>(clusters.size()); ++g) {
      GaussClusterable *gc = static_cast<GaussClusterable *>(clusters[g]);
      weights_[g] = gc->count();

      FloatVector mean_invvar = gc->x_stats().cast<float>() / gc->count();
      FloatVector inv_var =
          1.0f / (gc->x2_stats().cast<float>().array() / gc->count() -
                  mean_invvar.array().square());

      mean_invvar = mean_invvar.array() * inv_var.array();

      inv_vars_.row(g) = inv_var;
      means_invvars_.row(g) = mean_invvar;
    }

    ComputeGconsts();
    DeletePointers(&clusterable_vec);
    DeletePointers(&clusters);
  }
}

void DiagGmm::Merge(int32_t target_components, std::vector<int32_t> *history) {
  if (target_components <= 0 || NumGauss() < target_components) {
    KHG_ERR << "Invalid argument for target number of Gaussians (="
            << target_components << "), #Gauss = " << NumGauss();
  }

  if (NumGauss() == target_components) {
    KHG_LOG << "No components merged, as target (" << target_components
            << ") = total.";
    return;  // Nothing to do.
  }

  int32_t num_comp = NumGauss(), dim = Dim();

  if (target_components == 1) {  // global mean and variance
    // Undo variance inversion and multiplication of mean by inv var.
    FloatMatrix vars = 1.0f / inv_vars_.array();

    FloatMatrix means = means_invvars_.array() * vars.array();

    vars = vars.array() + means.array().square();  // 2-nd order stats

    // Slightly more efficient than calling this->Resize(1, dim)
    gconsts_.resize(1);

    // weights_.transpose(): (1, nmix)
    // means: (nmix, dim)
    // means_invvars_: (1, dim)
    means_invvars_ = weights_.transpose() * means;

    inv_vars_ = weights_.transpose() * vars;

    float weights_sum = weights_.sum();
    weights_.resize(1);
    weights_[0] = weights_sum;

    if (!ApproxEqual(weights_[0], 1.0f, 1e-6)) {
      KHG_WARN << "Weights sum to " << weights_[0] << ": rescaling.";
      means_invvars_ *= weights_[0];
      inv_vars_ *= weights_[0];

      weights_[0] = 1.0;
    }

    inv_vars_ = 1.0f / (inv_vars_.array() - means_invvars_.array().square());

    means_invvars_ = means_invvars_.array() * inv_vars_.array();

    ComputeGconsts();
    return;
  }

  // If more than 1 merged component is required, use the hierarchical
  // clustering of components that lead to the smallest decrease in likelihood.
  std::vector<bool> discarded_component(num_comp, false);

  // logdet is of shape (nmix, 1)
  //
  // +0.5 because var is inverted
  FloatVector logdet = 0.5 * inv_vars_.array().log().rowwise().sum();

  // Undo variance inversion and multiplication of mean by this
  // Makes copy of means and vars for all components - memory inefficient?
  FloatMatrix vars = 1.0f / inv_vars_.array();
  FloatMatrix means = means_invvars_.array() * vars.array();

  // add means square to variances; get second-order stats
  // (normalized by zero-order stats)
  vars = vars.array() + means.array().square();

  // TODO(fangjun): We only need a triangular matrix here
  //
  // compute change of likelihood for all combinations of components
  FloatMatrix delta_like(num_comp, num_comp);

  for (int32_t i = 0; i < num_comp; i++) {
    for (int32_t j = 0; j < i; j++) {
      float w1 = weights_[i], w2 = weights_[j], w_sum = w1 + w2;

      // TODO(fangjun): Optimize MergedComponentsLogdet() to avoid copies
      float merged_logdet = MergedComponentsLogdet(
          w1, w2, means.row(i), means.row(j), vars.row(i), vars.row(j));

      delta_like(i, j) =
          w_sum * merged_logdet - w1 * logdet[i] - w2 * logdet[j];
    }
  }

  // Merge components with smallest impact on the loglike
  for (int32_t removed = 0; removed < num_comp - target_components; ++removed) {
    // Search for the least significant change in likelihood
    // (maximum of negative delta_likes)
    float max_delta_like = -std::numeric_limits<float>::max();
    int32_t max_i = -1, max_j = -1;
    for (int32_t i = 0; i < NumGauss(); i++) {
      if (discarded_component[i]) continue;
      for (int32_t j = 0; j < i; j++) {
        if (discarded_component[j]) continue;
        if (delta_like(i, j) > max_delta_like) {
          max_delta_like = delta_like(i, j);
          max_i = i;
          max_j = j;
        }
      }
    }

    // make sure that different components will be merged
    KHG_ASSERT(max_i != max_j && max_i != -1 && max_j != -1);

    // remember the merged candidates
    if (history != nullptr) {
      history->push_back(max_i);
      history->push_back(max_j);
    }

    // Merge components
    float w1 = weights_[max_i], w2 = weights_[max_j];
    float w_sum = w1 + w2;

    // merge means
    means.row(max_i) =
        (means.row(max_i) + w2 / w1 * means.row(max_j)) * w1 / w_sum;

    // merge vars
    vars.row(max_i) =
        (vars.row(max_i) + w2 / w1 * vars.row(max_j)) * w1 / w_sum;

    // merge weights
    weights_[max_i] = w_sum;

    // Update gmm for merged component
    // copy second-order stats (normalized by zero-order stats)
    // and centralize, and invert
    inv_vars_.row(max_i) =
        1.0f / (vars.row(max_i).array() - means.row(max_i).array().square());

    // copy first-order stats (normalized by zero-order stats)
    // and multiply by inv_vars
    means_invvars_.row(max_i) =
        means.row(max_i).array() * inv_vars_.row(max_i).array();

    // Update logdet for merged component
    // +0.5 because var is inverted
    logdet[max_i] = 0.5 * inv_vars_.row(max_i).array().log().sum();

    // Label the removed component as discarded
    discarded_component[max_j] = true;

    // Update delta_like for merged component
    for (int32_t j = 0; j < num_comp; ++j) {
      if ((j == max_i) || (discarded_component[j])) {
        continue;
      }

      float w1 = weights_[max_i], w2 = weights_[j], w_sum = w1 + w2;

      float merged_logdet = MergedComponentsLogdet(
          w1, w2, means.row(max_i), means.row(j), vars.row(max_i), vars.row(j));

      float tmp = w_sum * merged_logdet - w1 * logdet[max_i] - w2 * logdet[j];

      delta_like(max_i, j) = tmp;
      delta_like(j, max_i) = tmp;  // TODO(fangjun): We only need to set one
      // doesn't respect lower triangular indices,
      // relies on implicitly performed swap of coordinates if necessary
    }
  }

  int32_t num_kept = 0;
  for (auto i : discarded_component) {
    if (i) {
      continue;
    }

    ++num_kept;
  }

  FloatVector tmp_weights(num_kept);

  FloatMatrix tmp_means_invvars(num_kept, Dim());
  FloatMatrix tmp_inv_vars(num_kept, Dim());

  // Remove the consumed components
  int32_t m = 0;
  for (int32_t i = 0; i < num_comp; ++i) {
    if (discarded_component[i]) {
      continue;
    }

    tmp_weights[m] = weights_[i];

    tmp_means_invvars.row(m) = means_invvars_.row(i);

    tmp_inv_vars.row(m) = inv_vars_.row(i);
    ++m;
  }

  weights_ = std::move(tmp_weights);
  means_invvars_ = std::move(tmp_means_invvars);
  inv_vars_ = std::move(tmp_inv_vars);

  ComputeGconsts();
}

float DiagGmm::MergedComponentsLogdet(float w1, float w2,
                                      const FloatVector &f1,  // 1-D
                                      const FloatVector &f2,  // 1-D
                                      const FloatVector &s1,  // 1-D
                                      const FloatVector &s2   // 1-D
) const {
  float w_sum = w1 + w2;

  auto tmp_mean = (f1.array() + f2.array() * (w2 / w1)) * (w1 / w_sum);

  auto tmp_var =
      (s1.array() + s2.array() * (w2 / w1)) * (w1 / w_sum) - tmp_mean.square();

  // -0.5 because var is not inverted
  float merged_logdet = -0.5 * tmp_var.log().sum();

  return merged_logdet;
}

void DiagGmm::Split(int32_t target_components, float perturb_factor,
                    std::vector<int32_t> *history /*=nullptr*/) {
  if (target_components < NumGauss() || NumGauss() == 0) {
    KHG_ERR << "Cannot split from " << NumGauss() << " to " << target_components
            << " components";
  }
  if (target_components == NumGauss()) {
    KHG_WARN << "Already have the target # of Gaussians. Doing nothing.";
    return;
  }

  int32_t current_components = NumGauss(), dim = Dim();

  // First do the resize:

  // the original content is copied to the newly allocated weights_
  weights_.conservativeResize(target_components);

  means_invvars_.conservativeResize(target_components, Eigen::NoChange_t{});
  inv_vars_.conservativeResize(target_components, Eigen::NoChange_t{});

  gconsts_.resize(target_components);

  // future work(arnab): Use a priority queue instead?
  while (current_components < target_components) {
    // TODO(fangjun): optimize it
    float max_weight = weights_[0];
    int32_t max_idx = 0;
    for (int32_t i = 1; i < current_components; ++i) {
      if (weights_[i] > max_weight) {
        max_weight = weights_[i];
        max_idx = i;
      }
    }

    // remember what component was split
    if (history != nullptr) {
      history->push_back(max_idx);
    }

    weights_[max_idx] /= 2;
    weights_[current_components] = weights_[max_idx];

    FloatVector rand_vec = RandnVector(dim);

#if 1
    rand_vec =
        rand_vec.transpose().array() * inv_vars_.row(max_idx).array().sqrt();
#else
    // TODO(fangjun): Replace the for loop with tensor operations
    for (int32_t i = 0; i < dim; ++i) {
      rand_vec[i] *= std::sqrt(inv_vars_(max_idx, i));
      // note, this looks wrong but is really right because it's the
      // means_invvars we're multiplying and they have the dimension
      // of an inverse standard variance. [dan]
    }
#endif

    inv_vars_.row(current_components) = inv_vars_.row(max_idx);

    means_invvars_.row(current_components) =
        means_invvars_.row(max_idx).array() +
        rand_vec.transpose().array() * perturb_factor;

    means_invvars_.row(max_idx) = means_invvars_.row(max_idx).array() -
                                  rand_vec.transpose().array() * perturb_factor;

    ++current_components;
  }

  ComputeGconsts();
}

void DiagGmm::RemoveComponents(const std::vector<int32_t> &gauss_in,
                               bool renorm_weights) {
  std::vector<int32_t> gauss(gauss_in);
  std::sort(gauss.begin(), gauss.end());

  KHG_ASSERT(IsSortedAndUniq(gauss));

  // If efficiency is later an issue, will code this specially (unlikely).
  for (size_t i = 0; i < gauss.size(); ++i) {
    RemoveComponent(gauss[i], renorm_weights);

    for (size_t j = i + 1; j < gauss.size(); ++j) {
      gauss[j]--;
    }
  }
}

void DiagGmm::RemoveComponent(int32_t gauss, bool renorm_weights) {
  KHG_ASSERT(gauss < NumGauss());
  KHG_ASSERT(gauss >= 0);

  if (NumGauss() == 1) {
    KHG_ERR << "Attempting to remove the only remaining component.";
  }

  FloatVector new_weights;
  FloatVector new_gconsts;
  FloatMatrix new_means_invvars;
  FloatMatrix new_inv_vars;

  if (gauss == 0) {
    new_weights = weights_(Eigen::seq(1, Eigen::last));

    new_gconsts = gconsts_(Eigen::seq(1, Eigen::last));

    new_means_invvars = means_invvars_(Eigen::seq(1, Eigen::last), Eigen::all);

    new_inv_vars = inv_vars_(Eigen::seq(1, Eigen::last), Eigen::all);
  } else if (gauss == NumGauss() - 1) {
    new_weights = weights_(Eigen::seq(0, Eigen::last - 1));
    new_gconsts = gconsts_(Eigen::seq(0, Eigen::last - 1));

    new_means_invvars =
        means_invvars_(Eigen::seq(0, Eigen::last - 1), Eigen::all);

    new_inv_vars = inv_vars_(Eigen::seq(0, Eigen::last - 1), Eigen::all);
  } else {
    new_weights.resize(NumGauss() - 1);
    new_gconsts.resize(NumGauss() - 1);
    new_means_invvars.resize(NumGauss() - 1, Dim());
    new_inv_vars.resize(NumGauss() - 1, Dim());

    new_weights(Eigen::seq(0, gauss - 1)) = weights_(Eigen::seq(0, gauss - 1));

    new_weights(Eigen::seq(gauss, Eigen::last)) =
        weights_(Eigen::seq(gauss + 1, Eigen::last));

    new_gconsts(Eigen::seq(0, gauss - 1)) = gconsts_(Eigen::seq(0, gauss - 1));

    new_gconsts(Eigen::seq(gauss, Eigen::last)) =
        gconsts_(Eigen::seq(gauss + 1, Eigen::last));

    new_means_invvars(Eigen::seq(0, gauss - 1), Eigen::all) =
        means_invvars_(Eigen::seq(0, gauss - 1), Eigen::all);

    new_means_invvars(Eigen::seq(gauss, Eigen::last), Eigen::all) =
        means_invvars_(Eigen::seq(gauss + 1, Eigen::last), Eigen::all);

    new_inv_vars(Eigen::seq(0, gauss - 1), Eigen::all) =
        inv_vars_(Eigen::seq(0, gauss - 1), Eigen::all);

    new_inv_vars(Eigen::seq(gauss, Eigen::last), Eigen::all) =
        inv_vars_(Eigen::seq(gauss + 1, Eigen::last), Eigen::all);
  }

  if (renorm_weights) {
    new_weights /= new_weights.sum();

    valid_gconsts_ = false;
  }

  weights_ = std::move(new_weights);
  gconsts_ = std::move(new_gconsts);
  means_invvars_ = std::move(new_means_invvars);
  inv_vars_ = std::move(new_inv_vars);
}

void DiagGmm::SetWeights(const FloatVector &w) {
  KHG_ASSERT(weights_.size() == w.size());
  weights_ = w;
  valid_gconsts_ = false;
}

void DiagGmm::SetMeans(const FloatMatrix &m) {
  KHG_ASSERT(means_invvars_.rows() == m.rows() &&
             means_invvars_.cols() == m.cols());
  means_invvars_ = m.array() * inv_vars_.array();

  valid_gconsts_ = false;
}

FloatMatrix DiagGmm::GetMeans() const {
  return means_invvars_.array() / inv_vars_.array();
}

void DiagGmm::SetInvVars(const FloatMatrix &v) {
  KHG_ASSERT(inv_vars_.rows() == v.rows() && inv_vars_.cols() == v.cols());

  means_invvars_ = means_invvars_.array() / inv_vars_.array() * v.array();

  inv_vars_ = v;

  valid_gconsts_ = false;
}

FloatMatrix DiagGmm::GetVars() const { return 1.0 / inv_vars_.array(); }

void DiagGmm::SetComponentWeight(int32_t g, float w) {
  KHG_ASSERT(w > 0.0);
  KHG_ASSERT(g < NumGauss());

  weights_[g] = w;

  valid_gconsts_ = false;
}

void DiagGmm::SetComponentMean(int32_t g, const FloatVector &v) {
  KHG_ASSERT(g < NumGauss() && Dim() == v.size());

  means_invvars_.row(g) = inv_vars_.row(g).array() * v.transpose().array();

  valid_gconsts_ = false;
}

void DiagGmm::SetInvVarsAndMeans(const FloatMatrix &invvars,
                                 const FloatMatrix &means) {
  KHG_ASSERT(means_invvars_.rows() == means.rows() &&
             means_invvars_.cols() == means.cols() &&
             inv_vars_.rows() == invvars.rows() &&
             inv_vars_.cols() == invvars.cols());

  inv_vars_ = invvars;

  means_invvars_ = means.array() * inv_vars_.array();

  valid_gconsts_ = false;
}

void DiagGmm::SetComponentInvVar(int32_t g, const FloatVector &v) {
  KHG_ASSERT(g < NumGauss() && v.size() == Dim());

  means_invvars_.row(g) = means_invvars_.row(g).array() /
                          inv_vars_.row(g).array() * v.transpose().array();

  inv_vars_.row(g) = v;

  valid_gconsts_ = false;
}

FloatVector DiagGmm::GetComponentMean(int32_t gauss) const {
  KHG_ASSERT(gauss < NumGauss());

  return means_invvars_.row(gauss).array() / inv_vars_.row(gauss).array();
}

FloatVector DiagGmm::GetComponentVariance(int32_t gauss) const {
  KHG_ASSERT(gauss < NumGauss());
  return 1.0f / inv_vars_.row(gauss).array();
}

}  // namespace khg
