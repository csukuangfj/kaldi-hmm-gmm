// kaldi-hmm-gmm/csrc/training-graph-compiler.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
//                2021  Xiaomi Corporation (Author: Junbo Zhang)
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/training-graph-compiler.h"

#include "kaldi-hmm-gmm/csrc/hmm-utils.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldifst/csrc/context-fst.h"
#include "kaldifst/csrc/fstext-utils.h"
#include "kaldifst/csrc/remove-eps-local.h"
#include "kaldifst/csrc/stl-utils.h"
#include "kaldifst/csrc/table-matcher.h"

namespace khg {

TrainingGraphCompiler::TrainingGraphCompiler(
    const TransitionModel &trans_model,
    const ContextDependency &ctx_dep,  // Does not maintain reference to this.
    fst::VectorFst<fst::StdArc> *lex_fst,
    const std::vector<int32_t> &disambig_syms,
    const TrainingGraphCompilerOptions &opts)
    : trans_model_(trans_model),
      ctx_dep_(ctx_dep),
      lex_fst_(lex_fst),
      disambig_syms_(disambig_syms),
      opts_(opts) {
  using namespace fst;  // NOLINT
  const std::vector<int32_t> &phone_syms =
      trans_model_.GetPhones();  // needed to create context fst.

  KHG_ASSERT(!phone_syms.empty());
  KHG_ASSERT(IsSortedAndUniq(phone_syms));
  SortAndUniq(&disambig_syms_);
  for (int32_t i = 0; i < disambig_syms_.size(); i++)
    if (std::binary_search(phone_syms.begin(), phone_syms.end(),
                           disambig_syms_[i]))
      KHG_ERR << "Disambiguation symbol " << disambig_syms_[i]
              << " is also a phone.";

  subsequential_symbol_ = 1 + phone_syms.back();
  if (!disambig_syms_.empty() && subsequential_symbol_ <= disambig_syms_.back())
    subsequential_symbol_ = 1 + disambig_syms_.back();

  if (lex_fst == nullptr) return;

  {
    int32_t N = ctx_dep.ContextWidth(), P = ctx_dep.CentralPosition();
    if (P != N - 1)
      AddSubsequentialLoop(subsequential_symbol_,
                           lex_fst_);  // This is needed for
    // systems with right-context or we will not successfully compose
    // with C.
  }

  {  // make sure lexicon is olabel sorted.
    fst::OLabelCompare<fst::StdArc> olabel_comp;
    fst::ArcSort(lex_fst_, olabel_comp);
  }
}

bool TrainingGraphCompiler::CompileGraphFromText(
    const std::vector<int32_t> &transcript,
    fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;  // NOLINT
  VectorFst<StdArc> word_fst;
  MakeLinearAcceptor(transcript, &word_fst);
  return CompileGraph(word_fst, out_fst);
}

bool TrainingGraphCompiler::CompileGraph(
    const fst::VectorFst<fst::StdArc> &word_fst,
    fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;  // NOLINT
  KHG_ASSERT(lex_fst_ != nullptr);
  KHG_ASSERT(out_fst != nullptr);

  VectorFst<StdArc> phone2word_fst;
  // TableCompose more efficient than compose.
  TableCompose(*lex_fst_, word_fst, &phone2word_fst, &lex_cache_);
  return CompileGraphFromLG(phone2word_fst, out_fst);
}

bool TrainingGraphCompiler::CompileGraphFromLG(
    const fst::VectorFst<fst::StdArc> &phone2word_fst,
    fst::VectorFst<fst::StdArc> *out_fst) {
  using namespace fst;  // NOLINT

  KHG_ASSERT(phone2word_fst.Start() != kNoStateId);

  const std::vector<int32_t> &phone_syms =
      trans_model_.GetPhones();  // needed to create context fst.

  // inv_cfst will be expanded on the fly, as needed.
  InverseContextFst inv_cfst(subsequential_symbol_, phone_syms, disambig_syms_,
                             ctx_dep_.ContextWidth(),
                             ctx_dep_.CentralPosition());

  VectorFst<StdArc> ctx2word_fst;
  ComposeDeterministicOnDemandInverse(phone2word_fst, &inv_cfst, &ctx2word_fst);
  // now ctx2word_fst is C * LG, assuming phone2word_fst is written as LG.
  KHG_ASSERT(ctx2word_fst.Start() != kNoStateId);

  HTransducerConfig h_cfg;
  h_cfg.transition_scale = opts_.transition_scale;

  std::vector<int32_t> disambig_syms_h;  // disambiguation symbols on
  // input side of H.
  VectorFst<StdArc> *H = GetHTransducer(inv_cfst.IlabelInfo(), ctx_dep_,
                                        trans_model_, h_cfg, &disambig_syms_h);

  VectorFst<StdArc> &trans2word_fst = *out_fst;  // transition-id to word.
  TableCompose(*H, ctx2word_fst, &trans2word_fst);

  KHG_ASSERT(trans2word_fst.Start() != kNoStateId);

  // Epsilon-removal and determinization combined. This will fail if not
  // determinizable.
  DeterminizeStarInLog(&trans2word_fst);

  if (!disambig_syms_h.empty()) {
    RemoveSomeInputSymbols(disambig_syms_h, &trans2word_fst);
    // we elect not to remove epsilons after this phase, as it is
    // a little slow.
    if (opts_.rm_eps) RemoveEpsLocal(&trans2word_fst);
  }

  // Encoded minimization.
  MinimizeEncoded(&trans2word_fst);

  std::vector<int32_t> disambig;
  bool check_no_self_loops = true;
  AddSelfLoops(trans_model_, disambig, opts_.self_loop_scale, opts_.reorder,
               check_no_self_loops, &trans2word_fst);

  delete H;
  return true;
}

bool TrainingGraphCompiler::CompileGraphs(
    const std::vector<const fst::VectorFst<fst::StdArc> *> &word_fsts,
    std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts) {
  out_fsts->resize(word_fsts.size(), nullptr);
  for (size_t i = 0; i < word_fsts.size(); i++) {
    fst::VectorFst<fst::StdArc> trans2word_fst;
    if (!CompileGraph(*(word_fsts[i]), &trans2word_fst)) return false;
    (*out_fsts)[i] = trans2word_fst.Copy();
  }
  return true;
}

bool TrainingGraphCompiler::CompileGraphsFromText(
    const std::vector<std::vector<int32_t>> &transcripts,
    std::vector<fst::VectorFst<fst::StdArc> *> *out_fsts) {
  using namespace fst;  // NOLINT
  std::vector<const VectorFst<StdArc> *> word_fsts(transcripts.size());
  for (size_t i = 0; i < transcripts.size(); i++) {
    VectorFst<StdArc> *word_fst = new VectorFst<StdArc>();
    MakeLinearAcceptor(transcripts[i], word_fst);
    word_fsts[i] = word_fst;
  }
  bool ans = CompileGraphs(word_fsts, out_fsts);
  for (size_t i = 0; i < transcripts.size(); i++) delete word_fsts[i];
  return ans;
}

}  // namespace khg
