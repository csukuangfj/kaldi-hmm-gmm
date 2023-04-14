// kaldi-hmm-gmm/python/csrc/cluster-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/python/csrc/cluster-utils.h"

#include "kaldi-hmm-gmm/csrc/cluster-utils.h"

namespace khg {

static void PybindRefineClustersOptions(py::module *m) {
  using PyClass = RefineClustersOptions;
  py::class_<PyClass>(*m, "RefineClustersOptions")
      .def(py::init<>())
      .def(py::init<int32_t, int32_t>(), py::arg("num_iters") = 100,
           py::arg("top_n") = 5)
      .def_readwrite("num_iters", &PyClass::num_iters)
      .def_readwrite("top_n", &PyClass::top_n);
}

static void PybindClusterKMeansOptions(py::module *m) {
  using PyClass = ClusterKMeansOptions;
  py::class_<PyClass>(*m, "ClusterKMeansOptions")
      .def(py::init<>())
      .def(py::init<const RefineClustersOptions &, int32_t, int32_t, bool>(),
           py::arg("refine_cfg") = RefineClustersOptions{},
           py::arg("num_iters") = 20, py::arg("num_tries") = 2,
           py::arg("verbose") = true)
      .def_readwrite("refine_cfg", &PyClass::refine_cfg)
      .def_readwrite("num_iters", &PyClass::num_iters)
      .def_readwrite("num_tries", &PyClass::num_tries)
      .def_readwrite("verbose", &PyClass::verbose);
}

void PybindClusterUtils(py::module *m) {
  PybindRefineClustersOptions(m);
  PybindClusterKMeansOptions(m);
}

}  // namespace khg
