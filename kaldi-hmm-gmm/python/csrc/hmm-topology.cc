// kaldi-hmm-gmm/python/csrc/hmm-topology.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/hmm-topology.h"

#include <sstream>
#include <string>
#include <vector>

#include "kaldi-hmm-gmm/csrc/hmm-topology.h"

namespace khg {

static void PybindHmmTopologyHmmState(py::module *m) {
  using PyClass = HmmTopology::HmmState;
  py::class_<PyClass>(*m, "HmmState")
      .def_readwrite("forward_pdf_class", &PyClass::forward_pdf_class)
      .def_readwrite("self_loop_pdf_class", &PyClass::self_loop_pdf_class)
      .def_readwrite("transitions", &PyClass::transitions)
      .def("__str__", [](const PyClass &self) {
        std::ostringstream os;
        os << "HmmState(";
        os << "forward_pdf_class=" << self.forward_pdf_class << ", ";
        os << "self_loop_pdf_class=" << self.self_loop_pdf_class << ", ";
        os << "transitions=[";
        std::string sep;
        for (const auto &pair : self.transitions) {
          os << sep << "(" << pair.first << ", " << pair.second << ")";
          sep = ", ";
        }
        os << "])";
        return os.str();
      });
  // no need to wrap the constructors
}

void PybindHmmTopology(py::module *m) {
  PybindHmmTopologyHmmState(m);
  using PyClass = HmmTopology;
  py::class_<PyClass>(*m, "HmmTopology")
      .def(py::init<>())
      .def("check", &PyClass::Check)
      .def("topology_for_phone", &PyClass::TopologyForPhone, py::arg("phone"))
      .def("num_pdf_classes", &PyClass::NumPdfClasses, py::arg("phone"))
      .def("get_phone_to_num_pdf_classes",
           [](const PyClass &self) {
             std::vector<int32_t> phone2num_pdf_classes;
             self.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
             return phone2num_pdf_classes;
           })
      .def("min_length", &PyClass::MinLength, py::arg("phone"))
      .def("read",
           [](PyClass &self, const std::string &s) {
             std::stringstream ss(s);
             bool binary = false;
             self.Read(ss, binary);
           })
      .def("__str__",
           [](const PyClass &self) {
             std::ostringstream os;
             bool binary = false;
             self.Write(os, binary);
             return os.str();
           })
      .def_property_readonly("phones", &PyClass::GetPhones)
      .def_property_readonly("is_hmm", &PyClass::IsHmm);
}

}  // namespace khg
