// kaldi-hmm-gmm/csrc/training-graph-compiler.h

#include <cstdint>
// Compat typedefs formerly provided transitively by old openfst headers.
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;
//
// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation
//

#ifndef KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_
#define KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_

/**
 * \file training-graph-compiler.h
 * \brief Compilation of training graphs (per-utterance FSTs labeled with
 *  transition-ids).
 *
 * This file is copied/modified from kaldi/src/decoder/training-graph-compiler.h.
 *
 * This module provides TrainingGraphCompiler, the class behind kaldi's
 * compile-train-graphs binary: it combines the lexicon FST (L), the phonetic
 * decision tree (ContextDependency) and the TransitionModel into a
 * context-dependent, transition-id-labeled graph (HCLG) for each utterance,
 * which forced alignment then searches.  The API consists of:
 *
 * - TrainingGraphCompilerOptions: transition/self-loop probability scales and
 *   graph-construction options (\c rm_eps, \c reorder).
 * - TrainingGraphCompiler: compile one graph from a word-level acceptor G
 *   (CompileGraph()) or from a raw word-id sequence
 *   (CompileGraphFromText()), or several at a time (CompileGraphs(),
 *   CompileGraphsFromText()); CompileGraphFromLG() takes an externally built
 *   LG fst.
 *
 * Compile-train-graphs semantics (important): the training graphs are
 * compiled with the transition and self-loop scales set to 0 - kaldi's
 * compile-train-graphs binary (kaldi/src/bin/compile-train-graphs.cc)
 * overrides the TrainingGraphCompilerOptions defaults of 1.0 to 0.0 "since we
 * will generally add the transition probs in the alignment phase (since they
 * change each time)".  The transition probabilities are then applied at
 * alignment time via AddTransitionProbs() (see hmm-utils.h; e.g. inside
 * gmm-align-compiled), because the transition probabilities change at every
 * training iteration while the graphs are kept fixed.  If instead the graphs
 * are compiled with non-zero scales and AddTransitionProbs() is applied on
 * top of that, the transition probabilities end up applied twice; this
 * double application matches kaldi and is intentional - do not "fix" it.
 *
 * See also kaldi/src/decoder/training-graph-compiler.h.
 */

#include <string>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "kaldifst/csrc/table-matcher.h"

namespace khg {

/// \brief Options for TrainingGraphCompiler.
///
/// See the file documentation for the compile-train-graphs scale
/// convention (scales 0 at compile time, transition probs added at
/// alignment time).
struct TrainingGraphCompilerOptions {
  /// Scale of transition probabilities (excluding self-loops).
  float transition_scale = 1.0;

  /// Scale of self-loop vs. non-self-loop probability mass.
  float self_loop_scale = 1.0;

  /// Remove [most] epsilons before minimization (only applicable
  /// if disambig symbols present)
  bool rm_eps = false;

  /// Reorder transition ids for greater decoding efficiency.
  /// (Dan-style graphs)
  bool reorder = true;

  /// Constructor; the defaults are the ones shown on the members (note the
  /// compile-train-graphs binary overrides the two scales to 0.0).
  explicit TrainingGraphCompilerOptions(float transition_scale = 1.0,
                                        float self_loop_scale = 1.0,
                                        bool reorder = true)
      : transition_scale(transition_scale),
        self_loop_scale(self_loop_scale),
        rm_eps(false),
        reorder(reorder) {}

  /// Returns a human-readable string representation, for logging.
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

/**
 * \brief Compiles per-utterance training graphs labeled with transition-ids.
 *
 * A TrainingGraphCompiler assembles HCLG-style graphs for training: the
 * lexicon fst (L) is composed with the utterance's grammar/acceptor (G),
 * the result is expanded to phone context (C) via the InverseContextFst, the
 * H transducer is built by GetHTransducer() and composed in, the result is
 * determinized and minimized, and finally AddSelfLoops() adds the self-loops.
 * The transition/self-loop probabilities are baked in at the scales
 * configured in TrainingGraphCompilerOptions (the compile-train-graphs
 * convention is to use scales 0 and apply AddTransitionProbs() at alignment
 * time; see the file documentation).
 *
 * The compiler keeps (and modifies) the lexicon fst it was given and caches
 * matcher state between compilations, so compiling many graphs at once is
 * faster than compiling them one at a time.
 */
class TrainingGraphCompiler {
 public:
  /**
   * \brief Constructor.
   *
   * \param trans_model  Maintains reference to this object.  (Not owned.)
   * \param ctx_dep      Maintains reference to this object.  (Not owned.)
   * \param lex_fst      Takes ownership of this object.  It should not
   *                     contain disambiguation symbols or subsequential
   *                     symbol, but it should contain optional silence.
   * \param disambig_syms  Disambig symbols in phone symbol table.
   * \param opts         Compilation options; see
   *                     TrainingGraphCompilerOptions.
   */
  TrainingGraphCompiler(
      const TransitionModel
          &trans_model,
      const ContextDependency &ctx_dep,
      fst::VectorFst<fst::StdArc> *lex_fst,
      const std::vector<int32_t>
          &disambig_syms,
      const TrainingGraphCompilerOptions &opts);

  /// Destructor; deletes the lexicon fst (which this object owns).
  ~TrainingGraphCompiler() { delete lex_fst_; }

  /// This version creates an FST from the text and calls CompileGraph.
  /// \param transcript  The utterance's transcript as a sequence of
  ///                    word-ids.
  /// \param out_fst     [out] The compiled training graph (transition-id to
  ///                    word transducer).
  /// \return True on success; false if the graph could not be compiled.
  bool CompileGraphFromText(const std::vector<int32_t> &transcript,
                            fst::VectorFst<fst::StdArc> *out_fst);

  /// CompileGraph compiles a single training graph its input is a
  /// weighted acceptor (G) at the word level, its output is HCLG.
  /// Note: G could actually be a transducer, it would also work.
  /// This function is not const for technical reasons involving the cache.
  /// if not for "table_compose" we could make it const.
  /// \param word_grammar  [in] The utterance's grammar at the word level.
  /// \param out_fst       [out] The compiled training graph (transition-id to
  ///                     word transducer).
  /// \return True on success; false if the graph could not be compiled.
  bool CompileGraph(const fst::VectorFst<fst::StdArc> &word_grammar,
                    fst::VectorFst<fst::StdArc> *out_fst);

  /// Same as `CompileGraph`, but uses an external LG fst.
  /// \param phone2word_fst  [in] The LG fst (phones to words), e.g. built by
  ///                        composing the lexicon with the grammar outside
  ///                        this class.
  /// \param out_fst         [out] The compiled training graph.
  /// \return True on success; false if the graph could not be compiled.
  bool CompileGraphFromLG(const fst::VectorFst<fst::StdArc> &phone2word_fst,
                          fst::VectorFst<fst::StdArc> *out_fst);

  /// CompileGraphs allows you to compile a number of graphs at the same
  /// time.  This consumes more memory but is faster.
  /// \param word_fsts  [in] The word-level grammars, one per utterance.
  /// \param out_fsts   [out] The compiled training graphs, one per input
  ///                   (owned by the caller).
  /// \return True on success; false if any of the graphs could not be
  ///         compiled.
  bool CompileGraphs(
      const std::vector<const fst::VectorFst<fst::StdArc> *> &word_fsts,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);

  /// This function creates FSTs from the text and calls CompileGraphs.
  /// \param word_grammar  [in] The utterance transcripts (sequences of
  ///                     word-ids), one per utterance.
  /// \param out_fsts      [out] The compiled training graphs (owned by the
  ///                     caller).
  /// \return True on success; false if any of the graphs could not be
  ///         compiled.
  bool CompileGraphsFromText(
      const std::vector<std::vector<int32_t>> &word_grammar,
      std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts);

 private:
  /// The transition model (not owned; maintained as a reference).
  const TransitionModel &trans_model_;

  /// The context-dependency tree (not owned; maintained as a reference).
  const ContextDependency &ctx_dep_;

  /// lexicon FST (an input; we take ownership as we need to modify it).
  fst::VectorFst<fst::StdArc> *lex_fst_;

  /// disambig symbols (if any) in the phone symbol table.
  std::vector<int32> disambig_syms_;

  /// search in ../fstext/context-fst.h for more info.
  int32 subsequential_symbol_;

  /// stores matcher state for table_compose; this is one of Dan's
  /// extensions.
  fst::TableComposeCache<fst::Fst<fst::StdArc>> lex_cache_;

  /// The options this compiler was constructed with.
  TrainingGraphCompilerOptions opts_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_TRAINING_GRAPH_COMPILER_H_
