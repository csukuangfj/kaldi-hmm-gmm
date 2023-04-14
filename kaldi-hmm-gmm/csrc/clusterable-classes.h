// kaldi-hmm-gmm/csrc/clusterable-classes.h
//
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#define KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
#include <string>

#include "kaldi-hmm-gmm/csrc/clusterable-itf.h"

namespace khg {

/// \addtogroup clustering_group
/// @{

/// ScalarClusterable clusters scalars with x^2 loss.
class ScalarClusterable : public Clusterable {
 public:
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
/// @} end of "addtogroup clustering_group"

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTERABLE_CLASSES_H_
