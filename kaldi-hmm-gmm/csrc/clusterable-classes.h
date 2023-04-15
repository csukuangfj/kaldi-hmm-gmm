// kaldi-hmm-gmm/csrc/clusterable-classes.h
//
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#define KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#include <string>

#include "kaldi-hmm-gmm/csrc/clusterable-itf.h"
#include "kaldi-hmm-gmm/csrc/utils.h"

namespace khg {

/// \addtogroup clustering_group
/// @{

/// ScalarClusterable clusters scalars with x^2 loss.
class ScalarClusterable : public Clusterable {
 public:
  ~ScalarClusterable() override = default;
  ScalarClusterable() : x_(0), x2_(0), count_(0) {}
  explicit ScalarClusterable(float x) : x_(x), x2_(x * x), count_(1) {}
  std::string Type() const override { return "scalar"; }
  float Objf() const override;
  void SetZero() override { count_ = x_ = x2_ = 0.0; }
  void Add(const Clusterable &other_in) override;
  void Sub(const Clusterable &other_in) override;
  Clusterable *Copy() const override;
  float Normalizer() const override { return static_cast<float>(count_); }

  std::string Info() const;  // For debugging.
  float Mean() const { return (count_ != 0 ? x_ / count_ : 0.0); }

 private:
  float x_;
  float x2_;
  float count_;
};

/// GaussClusterable wraps Gaussian statistics in a form accessible
/// to generic clustering algorithms.
class GaussClusterable : public Clusterable {
 public:
  ~GaussClusterable() override = default;
  GaussClusterable() : count_(0.0), var_floor_(0.0) {}
  GaussClusterable(int32_t dim, float var_floor)
      : count_(0.0),
        stats_{torch::zeros({2, dim}, torch::kDouble)},
        var_floor_(var_floor) {}

  GaussClusterable(torch::Tensor x_stats,   // 1-D tensor of shape (dim,)
                   torch::Tensor x2_stats,  // 1-D tensor of shape (dim,)
                   float var_floor, float count);

  std::string Type() const override { return "gauss"; }
  void AddStats(torch::Tensor vec,  // 1-D tensor of shape (dim,)
                float weight = 1.0);

  float Objf() const override;
  void SetZero() override;
  void Add(const Clusterable &other_in) override;
  void Sub(const Clusterable &other_in) override;
  float Normalizer() const override { return count_; }
  Clusterable *Copy() const override;
  void Scale(float f) override;

  float count() const { return count_; }

  // Return a 1-D tensor
  torch::Tensor x_stats() const { return Row(stats_, 0).squeeze(0); }

  // Return a 1-D tensor
  torch::Tensor x2_stats() const { return Row(stats_, 1).squeeze(0); }

 private:
  double count_;
  torch::Tensor stats_;  // kDouble, two rows: sum, then sum-squared.
  double var_floor_;     // should be common for all objects created.
};

/// @} end of "addtogroup clustering_group"

inline void GaussClusterable::SetZero() {
  count_ = 0;
  stats_.zero_();
}

inline GaussClusterable::GaussClusterable(torch::Tensor x_stats,
                                          torch::Tensor x2_stats,
                                          float var_floor, float count)
    : count_(count),
      stats_(torch::empty({2, x_stats.size(0)}, torch::kDouble)),
      var_floor_(var_floor) {
  Row(stats_, 0) = x_stats;
  Row(stats_, 1) = x2_stats;
}

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
