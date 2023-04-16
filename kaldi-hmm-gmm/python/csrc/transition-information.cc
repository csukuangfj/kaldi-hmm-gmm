// kaldi-hmm-gmm/python/csrc/transition-information.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/transition-information.h"

#include "kaldi-hmm-gmm/csrc/transition-information.h"

namespace khg {

void PybindTransitionInformation(py::module *m) {
  using PyClass = TransitionInformation;
  py::class_<PyClass>(*m, "TransitionInformation")
      .def("transition_ids_equivalent", &PyClass::TransitionIdsEquivalent,
           py::arg("trans_id1"), py::arg("trans_id2"))
      .def("transition_ids_is_start_of_phone",
           &PyClass::TransitionIdIsStartOfPhone, py::arg("trans_id"))
      .def("transition_id_to_phone", &PyClass::TransitionIdToPhone,
           py::arg("trans_id"))
      .def("is_final", &PyClass::IsFinal, py::arg("trans_id"))
      .def("is_self_loop", &PyClass::IsSelfLoop, py::arg("trans_id"))
      .def("transition_id_to_pdf", &PyClass::TransitionIdToPdf,
           py::arg("trans_id"))
      .def("transition_id_to_pdf_array", &PyClass::TransitionIdToPdfArray)
      .def_property_readonly("num_transition_ids", &PyClass::NumTransitionIds)
      .def_property_readonly("num_pdfs", &PyClass::NumPdfs);
}

}  // namespace khg
