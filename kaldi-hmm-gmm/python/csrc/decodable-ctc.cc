// kaldi-hmm-gmm/python/csrc/decodable-ctc.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/decodable-ctc.h"

#include "kaldi-hmm-gmm/csrc/decodable-ctc.h"

namespace khg {

void PybindDecodableCtc(py::module *m) {
  using PyClass = DecodableCtc;
  py::class_<PyClass, DecodableInterface>(*m, "DecodableCtc")
      .def(py::init<const FloatMatrix &>(), py::arg("feats"));
}

}  // namespace khg
