// kaldi-hmm-gmm/python/csrc/context-dep.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "kaldi-hmm-gmm/python/csrc/context-dep.h"

#include <string>

#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi_native_io/csrc/kaldi-io.h"
namespace khg {

void PybinContextDep(py::module *m) {
  using PyClass = ContextDependency;
  py::class_<PyClass>(*m, "ContextDependency")
      .def(
          "write",
          [](const PyClass &self, bool binary, const std::string &filename) {
            self.Write(kaldiio::Output(filename, binary).Stream(), binary);
          },
          py::arg("binary"), py::arg("filename"));

  m->def("monophone_context_dependency", &MonophoneContextDependency,
         py::arg("phones"), py::arg("phone2num_pdf_classes"),
         py::return_value_policy::take_ownership);

  m->def("monophone_context_dependency_shared",
         &MonophoneContextDependencyShared, py::arg("phone_classes"),
         py::arg("phone2num_pdf_classes"),
         py::return_value_policy::take_ownership);
}

}  // namespace khg
