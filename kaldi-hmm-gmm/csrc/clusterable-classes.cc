// kaldi-hmm-gmm/csrc/clusterable-classes.cc
//
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey
//                2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/clusterable-classes.h"

#include <algorithm>
#include <cmath>

#include "kaldi-hmm-gmm/csrc/kaldi-math.h"

namespace khg {

// ============================================================================
// Implementations common to all Clusterable classes (may be overridden for
// speed).
// ============================================================================

float Clusterable::ObjfPlus(const Clusterable &other) const {
  Clusterable *copy = this->Copy();
  copy->Add(other);
  float ans = copy->Objf();
  delete copy;
  return ans;
}

float Clusterable::ObjfMinus(const Clusterable &other) const {
  Clusterable *copy = this->Copy();
  copy->Sub(other);
  float ans = copy->Objf();
  delete copy;
  return ans;
}

float Clusterable::Distance(const Clusterable &other) const {
  Clusterable *copy = this->Copy();
  copy->Add(other);
  float ans = this->Objf() + other.Objf() - copy->Objf();
  if (ans < 0) {
    // This should not happen. Check if it is more than just rounding error.
    if (std::fabs(ans) > 0.01 * (1.0 + std::fabs(copy->Objf()))) {
      KHG_WARN << "Negative number returned (badly defined Clusterable "
               << "class?): ans= " << ans;
    }
    ans = 0;
  }
  delete copy;
  return ans;
}

// ============================================================================
// Implementation of ScalarClusterable class.
// ============================================================================

float ScalarClusterable::Objf() const {
  if (count_ == 0) {
    return 0;
  } else {
    KHG_ASSERT(count_ > 0);
    // See https://github.com/kaldi-asr/kaldi/issues/2914
    // for why we use such a formula
    return -(x2_ - x_ * x_ / count_);
  }
}

void ScalarClusterable::Add(const Clusterable &other_in) {
  KHG_ASSERT(other_in.Type() == "scalar");
  const ScalarClusterable *other =
      static_cast<const ScalarClusterable *>(&other_in);
  x_ += other->x_;
  x2_ += other->x2_;
  count_ += other->count_;
}

void ScalarClusterable::Sub(const Clusterable &other_in) {
  KHG_ASSERT(other_in.Type() == "scalar");
  const ScalarClusterable *other =
      static_cast<const ScalarClusterable *>(&other_in);
  x_ -= other->x_;
  x2_ -= other->x2_;
  count_ -= other->count_;
}

Clusterable *ScalarClusterable::Copy() const {
  ScalarClusterable *ans = new ScalarClusterable();
  ans->Add(*this);
  return ans;
}

std::string ScalarClusterable::Info() const {
  std::stringstream str;
  if (count_ == 0) {
    str << "[empty]";
  } else {
    str << "[mean " << (x_ / count_) << ", var "
        << (x2_ / count_ - (x_ * x_ / (count_ * count_))) << "]";
  }
  return str.str();
}

// ============================================================================
// Implementation of GaussClusterable class.
// ============================================================================

void GaussClusterable::AddStats(torch::Tensor vec, float weight /*=1.0*/) {
  count_ += weight;
  Row(stats_, 0).add_(vec, /*alpha*/ weight);
  Row(stats_, 1).add_(vec.square(), /*alpha*/ weight);
}

void GaussClusterable::Add(const Clusterable &other_in) {
  KHG_ASSERT(other_in.Type() == "gauss");
  const GaussClusterable *other =
      static_cast<const GaussClusterable *>(&other_in);
  count_ += other->count_;
  stats_.add_(other->stats_);
}

void GaussClusterable::Sub(const Clusterable &other_in) {
  KHG_ASSERT(other_in.Type() == "gauss");
  const GaussClusterable *other =
      static_cast<const GaussClusterable *>(&other_in);
  count_ -= other->count_;
  stats_.sub_(other->stats_);
}

Clusterable *GaussClusterable::Copy() const {
  KHG_ASSERT(stats_.dim() == 2);
  GaussClusterable *ans = new GaussClusterable(stats_.size(1), var_floor_);
  ans->Add(*this);
  return ans;
}

void GaussClusterable::Scale(float f) {
  KHG_ASSERT(f >= 0.0);
  count_ *= f;
  stats_.mul_(f);
}

float GaussClusterable::Objf() const {
  if (count_ <= 0.0) {
    if (count_ < -0.1) {
      KHG_WARN << "GaussClusterable::Objf(), count is negative " << count_;
    }
    return 0.0;
  } else {
    int32_t dim = stats_.size(1);
    torch::Tensor vars = torch::empty({dim}, torch::kDouble);
    auto vars_acc = vars.accessor<float, 1>();

    auto stats_acc = stats_.accessor<float, 2>();

    // TODO(fangjun): Use tensor operations to replace the for loop
    double objf_per_frame = 0.0;
    for (int32_t d = 0; d < dim; ++d) {
      double mean(stats_acc[0][d] / count_),
          var = stats_acc[1][d] / count_ - mean * mean,
          floored_var = std::max(var, var_floor_);
      vars_acc[d] = floored_var;
      objf_per_frame += -0.5 * var / floored_var;
    }
    objf_per_frame +=
        -0.5 * (vars.log().sum().item().toFloat() + M_LOG_2PI * dim);

    if (KALDI_ISNAN(objf_per_frame)) {
      KHG_WARN << "GaussClusterable::Objf(), objf is NaN";
      return 0.0;
    }
    return objf_per_frame * count_;
  }
}

}  // namespace khg
