// kaldi-hmm-gmm/python/csrc/hmm-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/hmm-utils.h"

#include <set>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/hmm-utils.h"

namespace khg {

static void PybindAddTransitionProbs(py::module *m) {
  m->def("add_transition_probs", &AddTransitionProbs, py::arg("trans_model"),
         py::arg("disambig_syms") = std::vector<int32_t>{},
         py::arg("transition_scale"), py::arg("self_loop_scale"),
         py::arg("fst"),
         py::call_guard<py::gil_scoped_release>());
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
          -> std::pair<fst::VectorFst<fst::StdArc>, std::vector<int32_t>> {
        std::vector<int32_t> disambig_syms_out;

        fst::VectorFst<fst::StdArc> *H = GetHTransducer(
            ilabel_info, ctx_dep, trans_model, config, &disambig_syms_out);

        std::pair<fst::VectorFst<fst::StdArc>, std::vector<int32_t>> ans(
            *H, disambig_syms_out);
        delete H;
        return ans;
      },
      py::arg("ilabel_info"), py::arg("ctx_dep"), py::arg("trans_model"),
      py::arg("config"));
}

static void PybindSplitToPhones(py::module *m) {
  m->def(
      "split_to_phones",
      [](const TransitionModel &trans_model,
         const std::vector<int32_t> &alignment)
          -> std::pair<bool, std::vector<std::vector<int32_t>>> {
        std::vector<std::vector<int32_t>> split_alignment;
        bool ok = SplitToPhones(trans_model, alignment, &split_alignment);
        return std::make_pair(ok, split_alignment);
      },
      py::arg("trans_model"), py::arg("alignment"),
      "Splits a transition-id alignment into per-phone segments. Returns "
      "(ok, split_alignment); ok is false if the alignment looks incomplete.");
}

static void PybindGetPdfToPhonesMap(py::module *m) {
  m->def(
      "get_pdf_to_phones_map",
      [](const TransitionModel &trans_model)
          -> std::vector<std::set<int32_t>> {
        std::vector<std::set<int32_t>> pdf2phones;
        GetPdfToPhonesMap(trans_model, &pdf2phones);
        return pdf2phones;
      },
      py::arg("trans_model"),
      "Returns a list mapping each pdf-id to the set of phones it can belong "
      "to.");
}

void PybindHmmUtils(py::module *m) {
  PybindAddTransitionProbs(m);
  PybindHTransducerConfig(m);
  PybindGetHTransducer(m);
  PybindSplitToPhones(m);
  PybindGetPdfToPhonesMap(m);
}

}  // namespace khg
