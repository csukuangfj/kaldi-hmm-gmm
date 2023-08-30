// kaldi-hmm-gmm/python/csrc/hmm-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/hmm-utils.h"

#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/hmm-utils.h"

namespace khg {

static void PybindAddTransitionProbs(py::module *m) {
  m->def("add_transition_probs", &AddTransitionProbs, py::arg("trans_model"),
         py::arg("disambig_syms") = std::vector<int32_t>{},
         py::arg("transition_scale"), py::arg("self_loop_scale"),
         py::arg("fst"));
}

static void PybindHTransducerConfig(py::module *m) {
  using PyClass = HTransducerConfig;
  py::class_<PyClass>(*m, "HTransducerConfig")
      .def(py::init<float, int>(), py::arg("transition_scale") = 1.0,
           py::arg("nonterm_phones_offset") = -1)
      .def_readwrite("transition_scale", &PyClass::transition_scale)
      .def_readwrite("nonterm_phones_offset", &PyClass::nonterm_phones_offset)
      .def("__str__", &PyClass::ToString);
}

static void PybindGetHTransducer(py::module *m) {
  m->def(
      "get_h_transducer",
      [](const std::vector<std::vector<int32_t>> &ilabel_info,
         const ContextDependencyInterface &ctx_dep,
         const TransitionModel &trans_model, const HTransducerConfig &config)
          -> std::pair<fst::VectorFst<fst::StdArc> *, std::vector<int32_t>> {
        std::vector<int32_t> disambig_syms_out;

        fst::VectorFst<fst::StdArc> *H = GetHTransducer(
            ilabel_info, ctx_dep, trans_model, config, &disambig_syms_out);

        return std::make_pair(H, disambig_syms_out);
      },
      py::arg("ilabel_info"), py::arg("ctx_dep"), py::arg("trans_model"),
      py::arg("config"), py::return_value_policy::take_ownership);
}

void PybindHmmUtils(py::module *m) {
  PybindAddTransitionProbs(m);
  PybindHTransducerConfig(m);
  PybindGetHTransducer(m);
}

}  // namespace khg
