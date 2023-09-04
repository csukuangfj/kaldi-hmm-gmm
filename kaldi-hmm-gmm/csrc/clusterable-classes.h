// kaldi-hmm-gmm/csrc/clusterable-classes.h
//
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#define KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#include <string>

#include "kaldi-hmm-gmm/csrc/clusterable-itf.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"

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
        x_stats_(DoubleVector::Zero(dim)),
        x2_stats_(DoubleVector::Zero(dim)),
        var_floor_(var_floor) {}

  GaussClusterable(const DoubleVector &x_stats,   // 1-D tensor of shape (dim,)
                   const DoubleVector &x2_stats,  // 1-D tensor of shape (dim,)
                   float var_floor, float count)
      : count_(count),
        x_stats_(x_stats),
        x2_stats_(x2_stats),
        var_floor_(var_floor) {}

  std::string Type() const override { return "gauss"; }

  void AddStats(const DoubleVector &vec,  // 1-D tensor of shape (dim,)
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
  const DoubleVector &x_stats() const { return x_stats_; }

  // Return a 1-D tensor
  const DoubleVector &x2_stats() const { return x2_stats_; }

 private:
  double count_;
  DoubleVector x_stats_;
  DoubleVector x2_stats_;
  double var_floor_;  // should be common for all objects created.
};

/// @} end of "addtogroup clustering_group"

inline void GaussClusterable::SetZero() {
  count_ = 0;
  x_stats_.setZero();
  x2_stats_.setZero();
}

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
