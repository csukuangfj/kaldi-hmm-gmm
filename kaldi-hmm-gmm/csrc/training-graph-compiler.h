// kaldi-hmm-gmm/csrc/training-graph-compiler.h
//
// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_
#define KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_

#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "kaldifst/csrc/table-matcher.h"

namespace khg {

struct TrainingGraphCompilerOptions {
  // Scale of transition probabilities (excluding self-loops)
  float transition_scale = 1.0;

  // Scale of self-loop vs. non-self-loop probability mass
  float self_loop_scale = 1.0;

  // Remove [most] epsilons before minimization (only applicable
  // if disambig symbols present)
  bool rm_eps = false;

  // Reorder transition ids for greater decoding efficiency.
  bool reorder = true;  // (Dan-style graphs)
                        //
  explicit TrainingGraphCompilerOptions(float transition_scale = 1.0,
                                        float self_loop_scale = 1.0,
                                        bool b = true)
      : transition_scale(transition_scale),
        self_loop_scale(self_loop_scale),
        rm_eps(false),
        reorder(b) {}
};

class TrainingGraphCompiler {
 public:
  TrainingGraphCompiler(
      const TransitionModel
          &trans_model,                  // Maintains reference to this object.
      const ContextDependency &ctx_dep,  // And this.
      fst::VectorFst<fst::StdArc> *lex_fst,  // Takes ownership of this object.
      // It should not contain disambiguation symbols or subsequential symbol,
      // but it should contain optional silence.
      const std::vector<int32_t>
          &disambig_syms,  // disambig symbols in phone symbol table.
      const TrainingGraphCompilerOptions &opts);

  ~TrainingGraphCompiler() { delete lex_fst_; }

 private:
  const TransitionModel &trans_model_;
  const ContextDependency &ctx_dep_;
  fst::VectorFst<fst::StdArc> *lex_fst_;  // lexicon FST (an input; we take
  // ownership as we need to modify it).
  std::vector<int32> disambig_syms_;  // disambig symbols (if any) in the phone
  int32 subsequential_symbol_;  // search in ../fstext/context-fst.h for more
                                // info.
  // symbol table.
  fst::TableComposeCache<fst::Fst<fst::StdArc>> lex_cache_;  // stores matcher..
  // this is one of Dan's extensions.

  TrainingGraphCompilerOptions opts_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_
