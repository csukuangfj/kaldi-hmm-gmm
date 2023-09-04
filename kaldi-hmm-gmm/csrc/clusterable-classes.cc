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

void GaussClusterable::AddStats(const DoubleVector &vec,
                                float weight /*=1.0*/) {
  count_ += weight;

  x_stats_ += vec * weight;

  x2_stats_ = x2_stats_.array() + vec.array().square() * weight;
}

void GaussClusterable::Add(const Clusterable &other_in) {
  KHG_ASSERT(other_in.Type() == "gauss");

  const GaussClusterable *other =
      static_cast<const GaussClusterable *>(&other_in);
  count_ += other->count_;

  x_stats_ += other->x_stats_;
  x2_stats_ += other->x2_stats_;
}

void GaussClusterable::Sub(const Clusterable &other_in) {
  KHG_ASSERT(other_in.Type() == "gauss");
  const GaussClusterable *other =
      static_cast<const GaussClusterable *>(&other_in);
  count_ -= other->count_;

  x_stats_ -= other->x_stats_;
  x2_stats_ -= other->x2_stats_;
}

Clusterable *GaussClusterable::Copy() const {
  GaussClusterable *ans = new GaussClusterable(x_stats_.size(), var_floor_);
  ans->Add(*this);
  return ans;
}

void GaussClusterable::Scale(float f) {
  KHG_ASSERT(f >= 0.0);
  count_ *= f;

  x_stats_ *= f;
  x2_stats_ *= f;
}

float GaussClusterable::Objf() const {
  if (count_ <= 0.0) {
    if (count_ < -0.1) {
      KHG_WARN << "GaussClusterable::Objf(), count is negative " << count_;
    }
    return 0.0;
  } else {
    int32_t dim = x_stats_.size();
    DoubleVector vars(dim);

    // TODO(fangjun): Use tensor operations to replace the for loop
    double objf_per_frame = 0.0;
    for (int32_t d = 0; d < dim; ++d) {
      double mean(x_stats_[d] / count_);
      double var = x2_stats_[d] / count_ - mean * mean;
      double floored_var = std::max(var, var_floor_);

      vars[d] = floored_var;
      objf_per_frame += -0.5 * var / floored_var;
    }

    objf_per_frame += -0.5 * (vars.array().log().sum() + M_LOG_2PI * dim);

    if (KALDI_ISNAN(objf_per_frame)) {
      KHG_WARN << "GaussClusterable::Objf(), objf is NaN";
      return 0.0;
    }

    return objf_per_frame * count_;
  }
}

}  // namespace khg
