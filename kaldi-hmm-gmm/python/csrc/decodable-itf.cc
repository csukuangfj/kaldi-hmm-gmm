// kaldi-hmm-gmm/python/csrc/decodable-itf.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/decodable-itf.h"

#include "kaldi-hmm-gmm/csrc/decodable-itf.h"

namespace khg {

void PybindDecodableItf(py::module *m) {
  using PyClass = DecodableInterface;

  py::class_<PyClass>(*m, "DecodableInterface")
      .def("log_likelihood", &PyClass::LogLikelihood, py::arg("frame"),
           py::arg("index"))
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"))
      .def("num_frames_ready", &PyClass::NumFramesReady)
      .def("num_indices", &PyClass::NumIndices);
}

}  // namespace khg
