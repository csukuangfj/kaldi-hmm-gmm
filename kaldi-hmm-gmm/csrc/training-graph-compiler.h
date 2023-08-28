// kaldi-hmm-gmm/csrc/training-graph-compiler.h
//
// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_
#define KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_

#include <string>
#include <vector>

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
                                        bool reorder = true)
      : transition_scale(transition_scale),
        self_loop_scale(self_loop_scale),
        rm_eps(false),
        reorder(reorder) {}

  std::string ToString() const {
    std::ostringstream os;
    os << "TrainingGraphCompilerOptions(";
    os << "transition_scale=" << transition_scale << ", ";
    os << "self_loop_scale=" << self_loop_scale << ", ";
    os << "reorder=" << (reorder ? "True" : "False");
    os << ")";
    return os.str();
  }
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

  // This version creates an FST from the text and calls CompileGraph.
  bool CompileGraphFromText(const std::vector<int32_t> &transcript,
                            fst::VectorFst<fst::StdArc> *out_fst);

  // CompileGraph compiles a single training graph its input is a
  // weighted acceptor (G) at the word level, its output is HCLG.
  // Note: G could actually be a transducer, it would also work.
  // This function is not const for technical reasons involving the cache.
  // if not for "table_compose" we could make it const.
  bool CompileGraph(const fst::VectorFst<fst::StdArc> &word_grammar,
                    fst::VectorFst<fst::StdArc> *out_fst);

  // Same as `CompileGraph`, but uses an external LG fst.
  bool CompileGraphFromLG(const fst::VectorFst<fst::StdArc> &phone2word_fst,
                          fst::VectorFst<fst::StdArc> *out_fst);

  // CompileGraphs allows you to compile a number of graphs at the same
  // time.  This consumes more memory but is faster.
  bool CompileGraphs(
      const std::vector<const fst::VectorFst<fst::StdArc> *> &word_fsts,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);

  // This function creates FSTs from the text and calls CompileGraphs.
  bool CompileGraphsFromText(
      const std::vector<std::vector<int32_t>> &word_grammar,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);

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
