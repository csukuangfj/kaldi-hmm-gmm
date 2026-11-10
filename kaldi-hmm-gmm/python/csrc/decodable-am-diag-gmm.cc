// kaldi-hmm-gmm/python/csrc/decodable-am-diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/decodable-am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h"

#include "kaldi-decoder/csrc/decodable-itf.h"

namespace khg {

static void PybindDecodableAmDiagGmmUnmapped(py::module *m) {
  using PyClass = DecodableAmDiagGmmUnmapped;
  // module_local: the DecodableInterface base is registered module_local in
  // this module; a module_local base requires module_local derived classes.
  py::class_<PyClass, kaldi_decoder::DecodableInterface>(
      *m, "DecodableAmDiagGmmUnmapped", py::module_local())
      .def(py::init<const AmDiagGmm &, const FloatMatrix &, float>(),
           py::arg("am"), py::arg("feats"), py::keep_alive<1, 2>(),
           py::arg("log_sum_exp_prune") = -1.0,
           py::call_guard<py::gil_scoped_release>());
}

static void PybindDecodableAmDiagGmmScaled(py::module *m) {
  using PyClass = DecodableAmDiagGmmScaled;
  py::class_<PyClass, DecodableAmDiagGmmUnmapped>(
      *m, "DecodableAmDiagGmmScaled", py::module_local())
      .def(py::init<const AmDiagGmm &, const TransitionModel &,
                    const FloatMatrix &, float, float>(),
           py::arg("am"), py::arg("tm"), py::arg("feats"), py::arg("scale"),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::arg("log_sum_exp_prune") = -1.0,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("transition_model", &PyClass::TransModel);
}

void PybindDecodableAmDiagGmm(py::module *m) {
  PybindDecodableAmDiagGmmUnmapped(m);
  PybindDecodableAmDiagGmmScaled(m);
}

}  // namespace khg
