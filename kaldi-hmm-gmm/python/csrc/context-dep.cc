// kaldi-hmm-gmm/python/csrc/context-dep.cc
//
// Copyright (c)  2022  Xiaomi Corporation
#include "kaldi-hmm-gmm/python/csrc/context-dep.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi_native_io/csrc/kaldi-io.h"
namespace khg {

static void PybindContextDependencyInterface(py::module *m) {
  using PyClass = ContextDependencyInterface;
  py::class_<PyClass>(*m, "ContextDependencyInterface")
      .def_property_readonly("context_width", &PyClass::ContextWidth)
      .def_property_readonly("central_position", &PyClass::CentralPosition)
      .def_property_readonly("num_pdfs", &PyClass::NumPdfs)
      .def(
          "compute",
          [](const PyClass &self, const std::vector<int32_t> &phoneseq,
             int32_t pdf_class) -> std::pair<bool, int32_t> {
            int32_t pdf_id = -1;
            bool found = self.Compute(phoneseq, pdf_class, &pdf_id);
            return std::make_pair(found, pdf_id);
          },
          py::arg("phone_seq"), py::arg("pdf_class"))
      .def(
          "get_pdf_info",
          [](const PyClass &self, const std::vector<int32_t> &phones,
             const std::vector<int32_t> &num_pdf_classes)
              -> std::vector<std::vector<std::pair<int32_t, int32_t>>> {
            std::vector<std::vector<std::pair<int32_t, int32_t>>> pdf_info;
            self.GetPdfInfo(phones, num_pdf_classes, &pdf_info);
            return pdf_info;

            // it is the inverse operation of compute().
            // while compute() converts (phone, pdf_class) to pdf_id,
            // get_pdf_info() converts pdf_id to a list of (phone, pdf_class).
            // The size of the returned list is usually 1
            //
          },
          py::arg("phones"), py::arg("num_pdf_classes"));
}

void PybindContextDep(py::module *m) {
  PybindContextDependencyInterface(m);
  using PyClass = ContextDependency;
  py::class_<PyClass, ContextDependencyInterface>(*m, "ContextDependency")
      .def(
          "write",
          [](const PyClass &self, bool binary, const std::string &filename) {
            self.Write(kaldiio::Output(filename, binary).Stream(), binary);
          },
          py::arg("binary"), py::arg("filename"))
      .def("__str__",
           [](const PyClass &self) -> std::string {
             std::ostringstream os;
             self.Write(os, false);
             return os.str();
           })
      .def(py::pickle(
          [](const PyClass &self) -> py::tuple {
            std::ostringstream os;
            self.Write(os, true);
            std::string s = os.str();
            std::vector<int8_t> data(s.begin(), s.end());
            return py::make_tuple(data);
          },
          [](const py::tuple &t) -> std::unique_ptr<PyClass> {
            auto data = t[0].cast<std::vector<int8_t>>();
            std::string s(data.begin(), data.end());

            std::istringstream is(s);
            auto ans = std::make_unique<PyClass>();
            ans->Read(is, true);
            return ans;
          }));

  m->def("monophone_context_dependency", &MonophoneContextDependency,
         py::arg("phones"), py::arg("phone2num_pdf_classes"),
         py::return_value_policy::take_ownership);

  m->def("monophone_context_dependency_shared",
         &MonophoneContextDependencyShared, py::arg("phone_classes"),
         py::arg("phone2num_pdf_classes"),
         py::return_value_policy::take_ownership);
}

}  // namespace khg
