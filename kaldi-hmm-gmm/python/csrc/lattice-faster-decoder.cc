// kaldi-hmm-gmm/python/csrc/lattice-faster-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/lattice-faster-decoder.h"

#include <limits>

#include "kaldi-hmm-gmm/csrc/lattice-faster-decoder.h"

namespace khg {

static void PybindLatticeFasterDecoderConfig(py::module *m) {
  using PyClass = LatticeFasterDecoderConfig;

  py::class_<PyClass>(*m, "LatticeFasterDecoderConfig")
      .def(py::init<float, int32_t, int32_t, float, int32_t, bool, float, float,
                    float, int32_t, int32_t,
                    const DeterminizeLatticePhonePrunedOptions &>(),
           py::arg("beam") = 16.0,
           py::arg("max_active") = std::numeric_limits<int32_t>::max(),
           py::arg("min_active") = 200, py::arg("lattice_beam") = 10.0,
           py::arg("prune_interval") = 25,
           py::arg("determinize_lattice") = true, py::arg("beam_delta") = 0.5,
           py::arg("hash_ratio") = 2.0, py::arg("prune_scale") = 0.1,
           py::arg("memory_pool_tokens_block_size") = (1 << 8),
           py::arg("memory_pool_links_block_size") = (1 << 8),
           py::arg("det_opts") = DeterminizeLatticePhonePrunedOptions{})
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("max_active", &PyClass::max_active)
      .def_readwrite("min_active", &PyClass::min_active)
      .def_readwrite("lattice_beam", &PyClass::lattice_beam)
      .def_readwrite("prune_interval", &PyClass::prune_interval)
      .def_readwrite("determinize_lattice", &PyClass::determinize_lattice)
      .def_readwrite("beam_delta", &PyClass::beam_delta)
      .def_readwrite("hash_ratio", &PyClass::hash_ratio)
      .def_readwrite("prune_scale", &PyClass::prune_scale)
      .def_readwrite("memory_pool_tokens_block_size",
                     &PyClass::memory_pool_tokens_block_size)
      .def_readwrite("memory_pool_links_block_size",
                     &PyClass::memory_pool_links_block_size)
      .def_readwrite("det_opts", &PyClass::det_opts)
      .def("__str__", &PyClass::ToString);
}

template <typename FST>
void PybindLatticeFasterDecoderTpl(py::module *m, const char *class_name) {
  using PyClass = LatticeFasterDecoderTpl<FST>;

  py::class_<PyClass>(*m, class_name)
      .def(py::init<const FST &, const LatticeFasterDecoderConfig &>(),
           py::arg("fst"), py::arg("config"));
}

void PybindLatticeFasterDecoder(py::module *m) {
  PybindLatticeFasterDecoderConfig(m);

  PybindLatticeFasterDecoderTpl<fst::Fst<fst::StdArc>>(m,
                                                       "LatticeFasterDecoder");

  PybindLatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>>(
      m, "LatticeFasterDecoderStdVectorFst");

  PybindLatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>>(
      m, "LatticeFasterDecoderStdConstFst");
}

}  // namespace khg
