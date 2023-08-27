// kaldi-hmm-gmm/python/csrc/model-common.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/model-common.h"

#include "kaldi-hmm-gmm/csrc/model-common.h"
namespace khg {

static void PybindGmmUpdateFlags(py::module *m) {
  using PyClass = GmmUpdateFlags;
  py::enum_<PyClass>(*m, "GmmUpdateFlags")
      .value("kGmmMeans", PyClass::kGmmMeans)
      .value("kGmmVariances", PyClass::kGmmVariances)
      .value("kGmmWeights", PyClass::kGmmWeights)
      .value("kGmmTransitions", PyClass::kGmmTransitions)
      .value("kGmmAll", PyClass::kGmmAll)
      .export_values();
}

void PybindModelCommon(py::module *m) {
  PybindGmmUpdateFlags(m);

  m->def("str_to_gmm_flags", &StringToGmmFlags);
  m->def("gmm_flags_to_str", &GmmFlagsToString);
}

}  // namespace khg
