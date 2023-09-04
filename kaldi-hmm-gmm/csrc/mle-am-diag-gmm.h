// kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h
//
// Copyright 2009-2012  Saarland University (author: Arnab Ghoshal);
//                      Yanmin Qian; Johns Hopkins University (author: Daniel
//                      Povey) Cisco Systems (author: Neha Agrawal)
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_

#include <vector>

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/mle-diag-gmm.h"

namespace khg {

class AccumAmDiagGmm {
 public:
  AccumAmDiagGmm() : total_frames_(0.0), total_log_like_(0.0) {}
  ~AccumAmDiagGmm();
  AccumAmDiagGmm(const AccumAmDiagGmm &) = delete;
  AccumAmDiagGmm &operator=(const AccumAmDiagGmm &) = delete;

  /// Initializes accumulators for each GMM based on the number of components
  /// and dimension.
  void Init(const AmDiagGmm &model, GmmFlagsType flags);

  /// Initialization using different dimension than model.
  void Init(const AmDiagGmm &model, int32_t dim, GmmFlagsType flags);

  void SetZero(GmmFlagsType flags);

  /// Accumulate stats for a single GMM in the model; returns log likelihood.
  /// This does not work with multiple feature transforms.
  ///
  /// @param data 1-D float tensor
  float AccumulateForGmm(const AmDiagGmm &model, const FloatVector &data,
                         int32_t gmm_index, float weight);

  /// Accumulate stats for a single GMM in the model; uses data1 for
  /// getting posteriors and data2 for stats. Returns log likelihood.
  ///
  /// @param data1 1-d float tensor
  /// @param data2 1-d float tensor
  float AccumulateForGmmTwofeats(const AmDiagGmm &model,
                                 const FloatVector &data1,
                                 const FloatVector &data2, int32_t gmm_index,
                                 float weight);

  /// Accumulates stats for a single GMM in the model using pre-computed
  /// Gaussian posteriors.
  ///
  /// @param data 1-d float tensor (dim,)
  /// @param posteriors 1-d float tensor (num_comp,)
  void AccumulateFromPosteriors(const AmDiagGmm &model, const FloatVector &data,
                                int32_t gmm_index,
                                const FloatVector &posteriors);

  /// Accumulate stats for a single Gaussian component in the model.
  ///
  /// @param data 1-d float tensor of shape (dim,)
  void AccumulateForGaussian(const AmDiagGmm &am, const FloatVector &data,
                             int32_t gmm_index, int32_t gauss_index,
                             float weight);

  int32_t NumAccs() const { return gmm_accumulators_.size(); }

  // returns the total count got by summing the count
  // of the actual stats, may differ from TotCount() if e.g. you did
  // I-smoothing.
  float TotStatsCount() const;

  // Be careful since total_frames_ is not updated in AccumulateForGaussian
  float TotCount() const { return total_frames_; }
  float TotLogLike() const { return total_log_like_; }

  const AccumDiagGmm &GetAcc(int32_t index) const;
  AccumDiagGmm &GetAcc(int32_t index);

  void Add(float scale, const AccumAmDiagGmm &other);

  void Scale(float scale);

  int32_t Dim() const {
    return (gmm_accumulators_.empty() || !gmm_accumulators_[0]
                ? 0
                : gmm_accumulators_[0]->Dim());
  }

 private:
  /// MLE accumulators and update methods for the GMMs
  std::vector<AccumDiagGmm *> gmm_accumulators_;

  /// Total counts & likelihood (for diagnostics)
  double total_frames_, total_log_like_;
};

/// for computing the maximum-likelihood estimates of the parameters of
/// an acoustic model that uses diagonal Gaussian mixture models as emission
/// densities.
void MleAmDiagGmmUpdate(const MleDiagGmmOptions &config,
                        const AccumAmDiagGmm &amdiaggmm_acc, GmmFlagsType flags,
                        AmDiagGmm *am_gmm, float *obj_change_out,
                        float *count_out);

/// Maximum A Posteriori update.
void MapAmDiagGmmUpdate(const MapDiagGmmOptions &config,
                        const AccumAmDiagGmm &amdiag_gmm_acc,
                        GmmFlagsType flags, AmDiagGmm *am_gmm,
                        float *obj_change_out, float *count_out);

}  // namespace khg
#endif  // KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_
