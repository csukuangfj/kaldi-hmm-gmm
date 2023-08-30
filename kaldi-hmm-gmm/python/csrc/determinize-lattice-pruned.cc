// kaldi-hmm-gmm/python/csrc/determinize-lattice-pruned.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/determinize-lattice-pruned.h"

#include "kaldi-hmm-gmm/csrc/determinize-lattice-pruned.h"

namespace khg {

static void PybindDeterminizeLatticePhonePrunedOptions(py::module *m) {
  using PyClass = DeterminizeLatticePhonePrunedOptions;
  py::class_<PyClass>(*m, "DeterminizeLatticePhonePrunedOptions")
      .def(py::init<float, int32_t, bool, bool, bool>(),
           py::arg("delta") = fst::kDelta, py::arg("max_mem") = 50000000,
           py::arg("phone_determinize") = true,
           py::arg("word_determinize") = true, py::arg("minimize") = false)
      .def_readwrite("delta", &PyClass::delta)
      .def_readwrite("max_mem", &PyClass::max_mem)
      .def_readwrite("phone_determinize", &PyClass::phone_determinize)
      .def_readwrite("word_determinize", &PyClass::word_determinize)
      .def_readwrite("minimize", &PyClass::minimize)
      .def("__str__", &PyClass::ToString);
}

void PybindDeterminizeLatticePruned(py::module *m) {
  PybindDeterminizeLatticePhonePrunedOptions(m);
}

}  // namespace khg
