// kaldi-hmm-gmm/csrc/training-graph-compiler.cc
//
// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)
//                2021  Xiaomi Corporation (Author: Junbo Zhang)
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/training-graph-compiler.h"

#include "kaldifst/csrc/context-fst.h"
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
  using namespace fst;
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

  if (lex_fst == NULL) return;

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

}  // namespace khg
