// kaldi-hmm-gmm/python/csrc/hmm-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/hmm-utils.h"

#include "kaldi-hmm-gmm/csrc/hmm-utils.h"

namespace khg {
static void PybindAddTransitionProbs(py::module *m) {
  m->def("add_transition_probs", &AddTransitionProbs, py::arg("trans_model"),
         py::arg("disambig_syms") = std::vector<int32_t>{},
         py::arg("transition_scale"), py::arg("self_loop_scale"),
         py::arg("fst"));
}

void PybindHmmUtils(py::module *m) { PybindAddTransitionProbs(m); }

}  // namespace khg
