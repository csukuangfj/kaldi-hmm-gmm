// kaldi-hmm-gmm/python/csrc/add-self-loops.cc

// Copyright 2009-2011  Microsoft Corporation
//                2015  Johns Hopkins University (author: Daniel Povey)
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/bin/add-self-loops.cc

#include "kaldi-hmm-gmm/python/csrc/add-self-loops.h"

#include <vector>

#include "kaldi-hmm-gmm/csrc/hmm-utils.h"

namespace khg {

static fst::VectorFst<fst::StdArc> *AddSelfLoopsWrapper(
    float self_loop_scale, const std::vector<int32_t> &disambig_syms,
    bool reorder, const TransitionModel &trans_model,
    const fst::VectorFst<fst::StdArc> &ifst) {
  // ans should be deleted by the user
  auto ans = ifst.Copy();

  bool check_no_self_loops = true;
  AddSelfLoops(trans_model, disambig_syms, self_loop_scale, reorder,
               check_no_self_loops, ans);
  return ans;
}

void PybindAddSelfLoops(py::module *m) {
  m->def("add_self_loops", &AddSelfLoopsWrapper,
         py::arg("self_loop_scale") = 1.0,
         py::arg("disambig_syms") = std::vector<int32_t>{},
         py::arg("reorder") = true, py::arg("trans_model"), py::arg("ifst"));
}

}  // namespace khg
