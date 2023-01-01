// kaldi-hmm-gmm/csrc/transition-model.cc
//
// Copyright (c)  2022  Xiaomi Corporation

// this if is copied and modified from
// kaldi/src/hmm/transition-model.cc

#include "kaldi-hmm-gmm/csrc/transition-model.h"

namespace khg {

TransitionModel::TransitionModel(const ContextDependencyInterface &ctx_dep,
                                 const HmmTopology &hmm_topo)
    : topo_(hmm_topo) {
  // First thing is to get all possible tuples.
  // ComputeTuples(ctx_dep);
  // ComputeDerived();
  // InitializeProbs();
  // Check();
}

}  // namespace khg
