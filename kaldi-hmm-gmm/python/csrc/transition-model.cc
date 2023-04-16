// kaldi-hmm-gmm/python/csrc/transition-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/transition-model.h"

#include "kaldi-hmm-gmm/csrc/transition-model.h"

namespace khg {

void PybindTransitionModel(py::module *m) {
  using PyClass = TransitionModel;
  py::class_<PyClass, TransitionInformation>(*m, "TransitionModel")
      .def(py::init<>())
      .def(py::init<const ContextDependencyInterface &, const HmmTopology &>(),
           py::arg("ctx_dep"), py::arg("hmm_topo"))
      .def_property_readonly("topo", &PyClass::GetTopo);
}

}  // namespace khg
