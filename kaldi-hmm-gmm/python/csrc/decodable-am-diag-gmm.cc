// kaldi-hmm-gmm/python/csrc/decodable-am-diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/decodable-am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h"

namespace khg {

static void PybindDecodableAmDiagGmmUnmapped(py::module *m) {
  using PyClass = DecodableAmDiagGmmUnmapped;
  py::class_<PyClass, DecodableInterface>(*m, "DecodableAmDiagGmmUnmapped")
      .def(py::init<const AmDiagGmm &, const FloatMatrix &, float>(),
           py::arg("am"), py::arg("feats"),
           py::arg("log_sum_exp_prune") = -1.0);
}

static void PybindDecodableAmDiagGmmScaled(py::module *m) {
  using PyClass = DecodableAmDiagGmmScaled;
  py::class_<PyClass, DecodableAmDiagGmmUnmapped>(*m,
                                                  "DecodableAmDiagGmmScaled")
      .def(py::init<const AmDiagGmm &, const TransitionModel &,
                    const FloatMatrix &, float, float>(),
           py::arg("am"), py::arg("tm"), py::arg("feats"), py::arg("scale"),
           py::arg("log_sum_exp_prune") = -1.0)
      .def_property_readonly("transition_model", &PyClass::TransModel);
}

void PybindDecodableAmDiagGmm(py::module *m) {
  PybindDecodableAmDiagGmmUnmapped(m);
  PybindDecodableAmDiagGmmScaled(m);
}

}  // namespace khg
