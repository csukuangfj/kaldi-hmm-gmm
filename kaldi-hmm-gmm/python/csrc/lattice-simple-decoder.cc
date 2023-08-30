// kaldi-hmm-gmm/python/csrc/lattice-simple-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/lattice-simple-decoder.h"

#include "kaldi-hmm-gmm/csrc/lattice-simple-decoder.h"

namespace khg {

static void PybindLatticeSimpleDecoderConfig(py::module *m) {
  using PyClass = LatticeSimpleDecoderConfig;

  py::class_<PyClass>(*m, "LatticeSimpleDecoderConfig")
      .def(py::init<float, float, int32_t, bool, float, float,
                    const DeterminizeLatticePhonePrunedOptions &>(),
           py::arg("beam") = 16.0, py::arg("lattice_beam") = 10.0,
           py::arg("prune_interval") = 25,
           py::arg("determinize_lattice") = true, py::arg("beam_ratio") = 0.9,
           py::arg("prune_scale") = 0.1,
           py::arg("det_opts") = DeterminizeLatticePhonePrunedOptions{})
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("lattice_beam", &PyClass::lattice_beam)
      .def_readwrite("prune_interval", &PyClass::prune_interval)
      .def_readwrite("determinize_lattice", &PyClass::determinize_lattice)
      .def_readwrite("beam_ratio", &PyClass::beam_ratio)
      .def_readwrite("prune_scale", &PyClass::prune_scale)
      .def_readwrite("det_opts", &PyClass::det_opts)
      .def("__str__", &PyClass::ToString);
}

void PybindLatticeSimpleDecoder(py::module *m) {
  PybindLatticeSimpleDecoderConfig(m);

  using PyClass = LatticeSimpleDecoder;
  py::class_<PyClass>(*m, "LatticeSimpleDecoder")
      .def(py::init<const fst::Fst<fst::StdArc> &,
                    const LatticeSimpleDecoderConfig &>(),
           py::arg("fst"), py::arg("config"));
}

}  // namespace khg
