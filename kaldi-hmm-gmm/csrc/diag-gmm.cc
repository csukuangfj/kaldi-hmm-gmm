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
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"
#include "kaldi-hmm-gmm/csrc/utils.h"

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

/// Get gaussian selection information for one frame.
float DiagGmm::GaussianSelection(const torch::Tensor &data,  // 1-D tensor
                                 int32_t num_gselect,
                                 std::vector<int32_t> *output) const {
  int32_t num_gauss = NumGauss();
  output->clear();

  torch::Tensor loglikes;
  LogLikelihoods(data, &loglikes);

  float thresh;
  if (num_gselect < num_gauss) {
    torch::Tensor loglikes_copy = loglikes.clone();
    float *ptr = loglikes_copy.data_ptr<float>();
    std::nth_element(ptr, ptr + num_gauss - num_gselect, ptr + num_gauss);
    thresh = ptr[num_gauss - num_gselect];
  } else {
    thresh = -std::numeric_limits<float>::infinity();
  }
  float tot_loglike = -std::numeric_limits<float>::infinity();

  auto loglikes_acc = loglikes.accessor<float, 1>();

  std::vector<std::pair<float, int32_t>> pairs;
  for (int32_t p = 0; p < num_gauss; p++) {
    if (loglikes_acc[p] >= thresh) {
      pairs.push_back(std::make_pair(loglikes_acc[p], p));
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            std::greater<std::pair<float, int32_t>>());

  for (int32_t j = 0; j < num_gselect && j < static_cast<int32_t>(pairs.size());
       j++) {
    output->push_back(pairs[j].second);
    tot_loglike = LogAdd(tot_loglike, pairs[j].first);
  }
  KHG_ASSERT(!output->empty());
  return tot_loglike;
}

float DiagGmm::GaussianSelection(
    const torch::Tensor &data,  // 2-D tensor of shape (num_frames, dim)
    int32_t num_gselect, std::vector<std::vector<int32_t>> *output) const {
  double ans = 0.0;
  int32_t num_frames = data.size(0), num_gauss = NumGauss();

  int32_t max_mem = 10000000;  // Don't devote more than 10Mb to loglikes_mat;
                               // break up the utterance if needed.
  int32_t mem_needed = num_frames * num_gauss * sizeof(float);
  if (mem_needed > max_mem) {
    // Break into parts and recurse, we don't want to consume too
    // much memory.
    int32_t num_parts = (mem_needed + max_mem - 1) / max_mem;
    int32_t part_frames = (data.size(0) + num_parts - 1) / num_parts;
    double tot_ans = 0.0;
    std::vector<std::vector<int32_t>> part_output;
    output->clear();
    output->resize(num_frames);
    for (int32_t p = 0; p < num_parts; p++) {
      int32_t start_frame = p * part_frames,
              this_num_frames = std::min(num_frames - start_frame, part_frames);

      torch::Tensor data_part =
          data.slice(/*dim*/ 0, start_frame, start_frame + this_num_frames);
      tot_ans += GaussianSelection(data_part, num_gselect, &part_output);

      for (int32_t t = 0; t < this_num_frames; t++)
        (*output)[start_frame + t].swap(part_output[t]);
    }
    KHG_ASSERT(!output->back().empty());
    return tot_ans;
  }

  KHG_ASSERT(num_frames != 0);
  torch::Tensor loglikes_mat;
  LogLikelihoodsMatrix(data, &loglikes_mat);

  output->clear();
  output->resize(num_frames);

  for (int32_t i = 0; i < num_frames; i++) {
    torch::Tensor loglikes = loglikes_mat.slice(0, i, i + 1);

    float thresh;
    if (num_gselect < num_gauss) {
      torch::Tensor loglikes_copy = loglikes.clone();
      float *ptr = loglikes_copy.data_ptr<float>();
      std::nth_element(ptr, ptr + num_gauss - num_gselect, ptr + num_gauss);
      thresh = ptr[num_gauss - num_gselect];
    } else {
      thresh = -std::numeric_limits<float>::infinity();
    }
    float tot_loglike = -std::numeric_limits<float>::infinity();

    auto loglikes_acc = loglikes.accessor<float, 1>();

    std::vector<std::pair<float, int32_t>> pairs;
    for (int32_t p = 0; p < num_gauss; p++) {
      if (loglikes_acc[p] >= thresh) {
        pairs.push_back(std::make_pair(loglikes_acc[p], p));
      }
    }
    std::sort(pairs.begin(), pairs.end(),
              std::greater<std::pair<float, int32_t>>());
    std::vector<int32_t> &this_output = (*output)[i];
    for (int32_t j = 0;
         j < num_gselect && j < static_cast<int32_t>(pairs.size()); j++) {
      this_output.push_back(pairs[j].second);
      tot_loglike = LogAdd(tot_loglike, pairs[j].first);
    }
    KHG_ASSERT(!this_output.empty());
    ans += tot_loglike;
  }
  return ans;
}

float DiagGmm::GaussianSelectionPreselect(const torch::Tensor &data,
                                          const std::vector<int32_t> &preselect,
                                          int32_t num_gselect,
                                          std::vector<int32_t> *output) const {
  static bool warned_size = false;
  int32_t preselect_sz = preselect.size();
  int32_t this_num_gselect = std::min(num_gselect, preselect_sz);
  if (preselect_sz <= num_gselect && !warned_size) {
    warned_size = true;
    KHG_WARN << "Preselect size is less or equal to than final size, "
             << "doing nothing: " << preselect_sz << " < " << num_gselect
             << " [won't warn again]";
  }
  torch::Tensor loglikes;
  LogLikelihoodsPreselect(data, preselect, &loglikes);

  torch::Tensor loglikes_copy = loglikes.clone();
  float *ptr = loglikes_copy.data_ptr<float>();
  std::nth_element(ptr, ptr + preselect_sz - this_num_gselect,
                   ptr + preselect_sz);
  float thresh = ptr[preselect_sz - this_num_gselect];

  float tot_loglike = -std::numeric_limits<float>::infinity();

  auto loglikes_acc = loglikes.accessor<float, 1>();

  // we want the output sorted from best likelihood to worse
  // (so we can prune further without the model)...
  std::vector<std::pair<float, int32_t>> pairs;
  for (int32_t p = 0; p < preselect_sz; p++)
    if (loglikes_acc[p] >= thresh)
      pairs.push_back(std::make_pair(loglikes_acc[p], preselect[p]));

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
float DiagGmm::ComponentPosteriors(const torch::Tensor &data,  // 1-D
                                   torch::Tensor *posterior    // 1-D
) const {
  if (!valid_gconsts_) {
    KHG_ERR << "Must call ComputeGconsts() before computing likelihood";
  }

  if (posterior == nullptr) {
    KHG_ERR << "NULL pointer passed as return argument.";
  }

  torch::Tensor loglikes;
  LogLikelihoods(data, &loglikes);

  torch::Tensor likes = loglikes.softmax(0);
  float log_sum = likes.sum().log().item().toFloat();

  if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum)) {
    KHG_ERR << "Invalid answer (overflow or invalid variances/features?)";
  }

  *posterior = likes;
  return log_sum;
}

float DiagGmm::ComponentLogLikelihood(const torch::Tensor &data,  // 1-D
                                      int32_t comp_id) const {
  if (!valid_gconsts_) {
    KHG_ERR << "Must call ComputeGconsts() before computing likelihood";
  }

  if (static_cast<int32_t>(data.size(0)) != Dim()) {
    KHG_ERR << "DiagGmm::ComponentLogLikelihood, dimension "
            << "mismatch " << (data.size(0)) << " vs. " << (Dim());
  }

  torch::Tensor data_sq = data.pow(2);

  // loglike =  means * inv(vars) * data.
  // loglike = VecVec(means_invvars_.Row(comp_id), data);
  float loglike =
      means_invvars_.slice(0, comp_id, comp_id + 1).dot(data).item().toFloat();

  // loglike += -0.5 * inv(vars) * data_sq.
  // loglike -= 0.5 * VecVec(inv_vars_.Row(comp_id), data_sq);
  loglike -=
      0.5 *
      inv_vars_.slice(0, comp_id, comp_id + 1).dot(data_sq).item().toFloat();

  return loglike + gconsts_.data_ptr<float>()[comp_id];
}

void DiagGmm::Generate(torch::Tensor *output  // 1-D
) {
  KHG_ASSERT(static_cast<int32_t>(output->size(0)) == Dim());
  float tot = weights_.sum().item().toFloat();
  KHG_ASSERT(tot > 0.0);
  double r = tot * torch::rand({1}, torch::kFloat).item().toFloat() * 0.99999;
  int32_t i = 0;
  double sum = 0.0;
  auto weights_acc = weights_.accessor<float, 1>();
  while (sum + weights_acc[i] < r) {
    sum += weights_acc[i];
    i++;
    KHG_ASSERT(i < static_cast<int32_t>(weights_.size(0)));
  }
  // now i is the index of the Gaussian we chose.
  auto inv_vars_acc = inv_vars_.accessor<float, 2>();
  auto means_invvars_acc = means_invvars_.accessor<float, 2>();

  auto output_acc = output->accessor<float, 1>();

  // TODO(fangjun): Use torch::rand()*mean + stddev to replace the following
  // for loop

  for (int32_t d = 0; d < inv_vars_.size(1); d++) {
    float stddev = 1.0 / sqrt(inv_vars_acc[i][d]),
          mean = means_invvars_acc[i][d] / inv_vars_acc[i][d];

    output_acc[d] =
        mean + torch::rand({1}, torch::kFloat).item().toFloat() * stddev;
  }
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
  auto tmp = std::make_unique<DiagGmm>();
  tmp->CopyFromDiagGmm(*this);  // so we have copies of matrices

  // First do the resize:
  weights_ = torch::empty({target_components}, torch::kFloat);
  weights_.slice(0, 0, current_components) = tmp->weights_;

  means_invvars_ = torch::empty({target_components, dim}, torch::kFloat);
  means_invvars_.slice(0, 0, current_components) = tmp->means_invvars_;

  // inv_vars_.Resize(target_components, dim);
  inv_vars_ = torch::empty({target_components, dim}, torch::kFloat);
  inv_vars_.slice(0, 0, current_components) = tmp->inv_vars_;

  gconsts_ = torch::empty({target_components}, torch::kFloat);

  auto weights_acc = weights_.accessor<float, 1>();
  // future work(arnab): Use a priority queue instead?
  while (current_components < target_components) {
    float max_weight = weights_acc[0];
    int32_t max_idx = 0;
    for (int32_t i = 1; i < current_components; i++) {
      if (weights_acc[i] > max_weight) {
        max_weight = weights_acc[i];
        max_idx = i;
      }
    }

    // remember what component was split
    if (history != nullptr) history->push_back(max_idx);

    weights_acc[max_idx] /= 2;
    weights_acc[current_components] = weights_acc[max_idx];

    torch::Tensor rand_vec = torch::empty({dim}, torch::kFloat);
    auto rand_vec_acc = rand_vec.accessor<float, 1>();
    auto inv_vars_acc = inv_vars_.accessor<float, 2>();

    for (int32_t i = 0; i < dim; i++) {
      rand_vec_acc[i] = torch::randn({1}, torch::kFloat).item().toFloat() *
                        std::sqrt(inv_vars_acc[max_idx][i]);
      // note, this looks wrong but is really right because it's the
      // means_invvars we're multiplying and they have the dimension
      // of an inverse standard variance. [dan]
    }
    inv_vars_.slice(0, current_components, current_components + 1) =
        inv_vars_.slice(0, max_idx, max_idx + 1);

    means_invvars_.slice(0, current_components, current_components + 1) =
        means_invvars_.slice(0, max_idx, max_idx + 1);

    means_invvars_.slice(0, current_components, current_components + 1)
        .add_(rand_vec, /*alpha*/ perturb_factor);

    means_invvars_.slice(0, max_idx, max_idx + 1)
        .add_(rand_vec, /*alpha*/ -perturb_factor);

    current_components++;
  }
  ComputeGconsts();
}

void DiagGmm::Perturb(float perturb_factor) {
  int32_t num_comps = NumGauss(), dim = Dim();

  torch::Tensor rand_mat = torch::randn({num_comps, dim}, torch::kFloat);

  // as in DiagGmm::Split, we perturb the means_invvars using a random
  // fraction of inv_vars_
  rand_mat = rand_mat * inv_vars_.sqrt();

  means_invvars_.add_(rand_mat, /*alpha*/ perturb_factor);

  ComputeGconsts();
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
    torch::Tensor weights = weights_.clone();
    // Undo variance inversion and multiplication of mean by inv var.
    torch::Tensor vars = 1.0f / inv_vars_;
    torch::Tensor means = means_invvars_ * vars;

    vars.add_(means.square(), /*alpha*/ 1.0);

    // Slightly more efficient than calling this->Resize(1, dim)
    gconsts_ = torch::empty({1}, torch::kFloat);

    weights_ = weights.sum();
    means_invvars_ = torch::mm(weights.unsqueeze(0), means);
    inv_vars_ = torch::mm(weights.unsqueeze(0), vars);

    auto weights_acc = weights_.accessor<float, 1>();
    if (!ApproxEqual(weights_acc[0], 1.0f, 1e-6)) {
      KHG_WARN << "Weights sum to " << weights_acc[0] << ": rescaling.";
      means_invvars_ *= weights_acc[0];
      inv_vars_ *= weights_acc[0];

      weights_acc[0] = 1.0;
    }
    inv_vars_.add_(means_invvars_.square(), /*alpha*/ -1.0);
    inv_vars_ = 1.0f / inv_vars_;

    means_invvars_.mul_(inv_vars_);

    ComputeGconsts();
    return;
  }

  // If more than 1 merged component is required, use the hierarchical
  // clustering of components that lead to the smallest decrease in likelihood.
  std::vector<bool> discarded_component(num_comp, false);

  // +0.5 because var is inverted
  torch::Tensor logdet =
      0.5 * inv_vars_.log().sum(1 /*dim*/, false /*keepdim*/);
  auto logdet_acc = logdet.accessor<float, 1>();

  // Undo variance inversion and multiplication of mean by this
  // Makes copy of means and vars for all components - memory inefficient?
  torch::Tensor vars = 1.0f / inv_vars_;
  torch::Tensor means = means_invvars_ * vars;

  // add means square to variances; get second-order stats
  // (normalized by zero-order stats)
  vars.add_(means.square(), /*alpha*/ 1.0);

  // TODO(fangjun): We only need a triangular matrix here
  //
  // compute change of likelihood for all combinations of components
  torch::Tensor delta_like = torch::empty({num_comp, num_comp}, torch::kFloat);
  auto delta_like_acc = delta_like.accessor<float, 2>();

  auto weights_acc = weights_.accessor<float, 1>();

  for (int32_t i = 0; i < num_comp; i++) {
    for (int32_t j = 0; j < i; j++) {
      float w1 = weights_acc[i], w2 = weights_acc[j], w_sum = w1 + w2;
      float merged_logdet = MergedComponentsLogdet(
          w1, w2, Row(means, i), Row(means, j), Row(vars, i), Row(vars, j));

      delta_like_acc[i][j] =
          w_sum * merged_logdet - w1 * logdet_acc[i] - w2 * logdet_acc[j];
    }
  }

  // Merge components with smallest impact on the loglike
  for (int32_t removed = 0; removed < num_comp - target_components; removed++) {
    // Search for the least significant change in likelihood
    // (maximum of negative delta_likes)
    float max_delta_like = -std::numeric_limits<float>::max();
    int32_t max_i = -1, max_j = -1;
    for (int32_t i = 0; i < NumGauss(); i++) {
      if (discarded_component[i]) continue;
      for (int32_t j = 0; j < i; j++) {
        if (discarded_component[j]) continue;
        if (delta_like_acc[i][j] > max_delta_like) {
          max_delta_like = delta_like_acc[i][j];
          max_i = i;
          max_j = j;
        }
      }
    }

    // make sure that different components will be merged
    KHG_ASSERT(max_i != max_j && max_i != -1 && max_j != -1);

    // remember the merge candidates
    if (history != nullptr) {
      history->push_back(max_i);
      history->push_back(max_j);
    }

    // Merge components
    float w1 = weights_acc[max_i], w2 = weights_acc[max_j];
    float w_sum = w1 + w2;
    // merge means
    Row(means, max_i) =
        (Row(means, max_i) + w2 / w1 * Row(means, max_j)) * w1 / w_sum;

    // merge vars
    Row(vars, max_i) =
        (Row(vars, max_i) + w2 / w1 * Row(vars, max_j)) * w1 / w_sum;

    // merge weights
    weights_acc[max_i] = w_sum;

    // Update gmm for merged component
    // copy second-order stats (normalized by zero-order stats)
    // and centralize, and invert
    Row(inv_vars_, max_i) =
        1.0f / (Row(vars, max_i) - Row(means, max_i).square());

    // copy first-order stats (normalized by zero-order stats)
    // and multiply by inv_vars
    Row(means_invvars_, max_i) = Row(means, max_i) * Row(inv_vars_, max_i);

    // Update logdet for merged component
    // +0.5 because var is inverted
    logdet_acc[max_i] =
        0.5 * Row(inv_vars_, max_i).log().sum().item().toFloat();

    // Label the removed component as discarded
    discarded_component[max_j] = true;

    // Update delta_like for merged component
    for (int32_t j = 0; j < num_comp; j++) {
      if ((j == max_i) || (discarded_component[j])) continue;
      float w1 = weights_acc[max_i], w2 = weights_acc[j], w_sum = w1 + w2;
      float merged_logdet =
          MergedComponentsLogdet(w1, w2, Row(means, max_i), Row(means, j),
                                 Row(vars, max_i), Row(vars, j));
      float tmp =
          w_sum * merged_logdet - w1 * logdet_acc[max_i] - w2 * logdet_acc[j];
      delta_like_acc[max_i][j] = tmp;
      delta_like_acc[j][max_i] = tmp;  // TODO(fangjun): We only need to set one
      // doesn't respect lower triangular indices,
      // relies on implicitly performed swap of coordinates if necessary
    }
  }

  int32_t num_kept = 0;
  for (auto i : discarded_component) {
    if (i) continue;

    ++num_kept;
  }

  torch::Tensor tmp_weights = torch::empty({num_kept}, torch::kFloat);
  torch::Tensor tmp_means_invvars =
      torch::empty({num_kept, Dim()}, torch::kFloat);
  torch::Tensor tmp_inv_vars = torch::empty({num_kept, Dim()}, torch::kFloat);

  auto tmp_weights_acc = tmp_weights.accessor<float, 1>();

  // Remove the consumed components
  int32_t m = 0;
  for (int32_t i = 0; i < num_comp; i++) {
    if (discarded_component[i]) continue;

    tmp_weights_acc[m] = weights_acc[i];
    Row(tmp_means_invvars, m) = Row(means_invvars_, i);
    Row(tmp_inv_vars, m) = Row(inv_vars_, i);
    ++m;
  }
  std::swap(tmp_weights, weights_);
  std::swap(tmp_means_invvars, means_invvars_);
  std::swap(tmp_inv_vars, inv_vars_);

  ComputeGconsts();
}

float DiagGmm::MergedComponentsLogdet(float w1, float w2,
                                      torch::Tensor f1,  // 1-D
                                      torch::Tensor f2,  // 1-D
                                      torch::Tensor s1,  // 1-D
                                      torch::Tensor s2   // 1-D
) const {
  float w_sum = w1 + w2;

  torch::Tensor tmp_mean = (f1 + f2 * (w2 / w1)) * (w1 / w_sum);

  torch::Tensor tmp_var =
      (s1 + s2 * (w2 / w1)) * (w1 / w_sum) - tmp_mean.square();

  // -0.5 because var is not inverted
  float merged_logdet = -0.5 * tmp_var.log().sum().item().toFloat();

  return merged_logdet;
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

  auto weights_acc = weights_.accessor<float, 1>();
  for (int32_t g = 0; g < NumGauss(); g++) {
    if (weights_acc[g] == 0) {
      KHG_WARN << "Not using zero-weight Gaussians in clustering.";
      continue;
    }
    float count = weights_acc[g];

    torch::Tensor var = 1.0f / Row(inv_vars_, g);
    torch::Tensor mean_invvar = Row(means_invvars_, g);

    torch::Tensor x_stats = mean_invvar * var;
    torch::Tensor x2_stats = (var + x_stats.square()) * count;
    x_stats.mul_(count);

    clusterable_vec.push_back(
        new GaussClusterable(x_stats, x2_stats, min_var, count));
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
      weights_acc[g] = gc->count();

      torch::Tensor inv_var = gc->x2_stats() / gc->count();
      torch::Tensor mean_invvar = gc->x_stats() / gc->count();
      inv_var.sub_(mean_invvar.square());

      inv_var = 1.0f / inv_var;
      mean_invvar.mul_(inv_var);

      Row(inv_vars_, g) = inv_var;
      Row(means_invvars_, g) = mean_invvar;
    }
    ComputeGconsts();
    DeletePointers(&clusterable_vec);
    DeletePointers(&clusters);
  }
}

void DiagGmm::Interpolate(float rho, const DiagGmm &source,
                          GmmFlagsType flags) {
  KHG_ASSERT(NumGauss() == source.NumGauss());
  KHG_ASSERT(Dim() == source.Dim());

  DiagGmmNormal us(*this);
  DiagGmmNormal them(source);

  if (flags & kGmmWeights) {
    us.weights_.mul_(1.0 - rho);
    us.weights_.add_(them.weights_, /*alpha*/ rho);
    us.weights_.div_(us.weights_.sum());
  }

  if (flags & kGmmMeans) {
    us.means_.mul_(1.0 - rho);
    us.means_.add_(them.means_, /*alpha*/ rho);
  }

  if (flags & kGmmVariances) {
    us.vars_.mul_(1.0 - rho);
    us.vars_.add_(them.vars_, /*alpha*/ rho);
  }

  us.CopyToDiagGmm(this);
  ComputeGconsts();
}

void DiagGmm::RemoveComponent(int32_t gauss, bool renorm_weights) {
  KHG_ASSERT(gauss < NumGauss());
  if (NumGauss() == 1) {
    KHG_ERR << "Attempting to remove the only remaining component.";
  }

  torch::Tensor new_weights = torch::empty({NumGauss() - 1}, torch::kFloat);
  torch::Tensor new_gconsts = torch::empty({NumGauss() - 1}, torch::kFloat);
  torch::Tensor new_means_invvars =
      torch::empty({NumGauss() - 1, Dim()}, torch::kFloat);
  torch::Tensor new_inv_vars =
      torch::empty({NumGauss() - 1, Dim()}, torch::kFloat);

  auto weights_acc = weights_.accessor<float, 1>();
  auto gconsts_acc = gconsts_.accessor<float, 1>();

  auto new_weights_acc = new_weights.accessor<float, 1>();
  auto new_gconsts_acc = new_gconsts.accessor<float, 1>();

  int32_t n = 0;
  for (int32_t i = 0; i != NumGauss(); ++i) {
    if (i == gauss) continue;
    new_weights_acc[n] = weights_acc[i];
    new_gconsts_acc[n] = gconsts_acc[i];
    Row(new_means_invvars, n) = Row(means_invvars_, i);
    Row(new_inv_vars, n) = Row(inv_vars_, i);
    ++n;
  }

  float sum_weights = new_weights.sum().item().toFloat();
  if (renorm_weights) {
    new_weights.mul_(1.0 / sum_weights);
    valid_gconsts_ = false;
  }

  weights_ = std::move(new_weights);
  gconsts_ = std::move(new_gconsts);
  means_invvars_ = std::move(new_means_invvars);
  inv_vars_ = std::move(new_inv_vars);
}

void DiagGmm::RemoveComponents(const std::vector<int32_t> &gauss_in,
                               bool renorm_weights) {
  std::vector<int32_t> gauss(gauss_in);
  std::sort(gauss.begin(), gauss.end());

  KHG_ASSERT(IsSortedAndUniq(gauss));
  // If efficiency is later an issue, will code this specially (unlikely).
  for (size_t i = 0; i < gauss.size(); i++) {
    RemoveComponent(gauss[i], renorm_weights);
    for (size_t j = i + 1; j < gauss.size(); j++) gauss[j]--;
  }
}

void DiagGmm::SetWeights(torch::Tensor w) {
  KHG_ASSERT(weights_.size(0) == w.size(0));
  weights_ = w.clone().to(torch::kFloat);
  valid_gconsts_ = false;
}

void DiagGmm::SetMeans(torch::Tensor m) {
  KHG_ASSERT(means_invvars_.size(0) == m.size(0) &&
             means_invvars_.size(1) == m.size(1));
  means_invvars_ = m.to(torch::kFloat).mul(inv_vars_);

  valid_gconsts_ = false;
}

void DiagGmm::SetInvVarsAndMeans(torch::Tensor invvars, torch::Tensor means) {
  KHG_ASSERT(means_invvars_.size(0) == means.size(0) &&
             means_invvars_.size(1) == means.size(1) &&
             inv_vars_.size(0) == invvars.size(0) &&
             inv_vars_.size(1) == invvars.size(1));

  inv_vars_ = invvars.clone().to(torch::kFloat);
  means_invvars_ = means.to(torch::kFloat).mul(inv_vars_);

  valid_gconsts_ = false;
}

void DiagGmm::SetInvVars(torch::Tensor v) {
  KHG_ASSERT(inv_vars_.size(0) == v.size(0) && inv_vars_.size(1) == v.size(1));

  v = v.to(torch::kFloat);

  means_invvars_ = means_invvars_.div(inv_vars_).mul(v);
  inv_vars_ = v.clone();

  valid_gconsts_ = false;
}

torch::Tensor DiagGmm::GetVars() const { return 1.0 / inv_vars_; }

torch::Tensor DiagGmm::GetMeans() const {
  return means_invvars_.div(inv_vars_);
}

void DiagGmm::SetComponentMean(int32_t g, torch::Tensor in) {
  KHG_ASSERT(g < NumGauss() && Dim() == in.size(0));

  Row(means_invvars_, g) = Row(inv_vars_, g).mul(in.to(torch::kFloat));

  valid_gconsts_ = false;
}

}  // namespace khg
