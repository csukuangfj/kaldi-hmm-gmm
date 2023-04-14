// kaldi-hmm-gmm/python/csrc/clusterable-classes.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/clusterable-classes.h"

#include "kaldi-hmm-gmm/csrc/clusterable-classes.h"
#include "kaldi-hmm-gmm/csrc/clusterable-itf.h"

namespace khg {

static void PybindClusterable(py::module *m) {
  using PyClass = Clusterable;
  py::class_<PyClass>(*m, "Clusterable")
      .def("copy", &PyClass::Copy, py::return_value_policy::take_ownership)
      .def("objf", &PyClass::Objf)
      .def("normalizer", &PyClass::Normalizer)
      .def("set_zero", &PyClass::SetZero)
      .def("add", &PyClass::Add, py::arg("other"))
      .def("sub", &PyClass::Sub, py::arg("other"))
      .def("scale", &PyClass::Scale, py::arg("f"))
      .def("type", &PyClass::Type)
      .def("objf_plus", &PyClass::ObjfPlus, py::arg("other"))
      .def("objf_minus", &PyClass::ObjfMinus, py::arg("other"))
      .def("distance", &PyClass::Distance, py::arg("oter"));
}

static void PybindScalarClusterable(py::module *m) {
  using PyClass = ScalarClusterable;
  py::class_<PyClass, Clusterable>(*m, "ScalarClusterable")
      .def(py::init<>())
      .def(py::init<float>(), py::arg("x"))
      .def("info", &PyClass::Info)
      .def("mean", &PyClass::Mean);
}

static void PybindGaussClusterable(py::module *m) {
  using PyClass = GaussClusterable;
  py::class_<PyClass, Clusterable>(*m, "GaussClusterable")
      .def(py::init<>())
      .def(py::init<int32_t, float>(), py::arg("dim"), py::arg("var_floor"))
      .def(py::init<torch::Tensor, torch::Tensor, float, float>(),
           py::arg("x_stats"), py::arg("x2_stats"), py::arg("var_floor"),
           py::arg("count"))
      .def("add_stats", &PyClass::AddStats, py::arg("vec"), py::arg("weight"))
      .def("count", &PyClass::count)
      .def("x_stats", &PyClass::x_stats)
      .def("x2_stats", &PyClass::x2_stats);
}

void PybindClusterableClass(py::module *m) {
  PybindClusterable(m);
  PybindScalarClusterable(m);
  PybindGaussClusterable(m);
}

}  // namespace khg
