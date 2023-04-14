// kaldi-hmm-gmm/csrc/clusterable-classes.cc
//
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2014  Daniel Povey
//                2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/clusterable-classes.h"

#include <cmath>

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

}  // namespace khg
