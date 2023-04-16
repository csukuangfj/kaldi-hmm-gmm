// kaldi-hmm-gmm/python/csrc/am-diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"

namespace khg {

void PybindAmDiagGmm(py::module *m) {
  using PyClass = AmDiagGmm;
  py::class_<PyClass>(*m, "AmDiagGmm")
      .def(py::init<>())
      .def_property_readonly("dim", &PyClass::Dim)
      .def_property_readonly("num_pdfs", &PyClass::NumPdfs)
      .def_property_readonly("num_gauss", &PyClass::NumGauss)
      .def("num_gauss_in_pdf", &PyClass::NumGaussInPdf, py::arg("pdf_index"))
      .def("init", &PyClass::Init, py::arg("proto"), py::arg("num_pdfs"))
      .def("add_pdf", &PyClass::AddPdf, py::arg("gmm"))
      .def("copy_from_am_diag_gmm", &PyClass::CopyFromAmDiagGmm,
           py::arg("other"))
      .def("split_pdf", &PyClass::SplitPdf, py::arg("idx"),
           py::arg("target_components"), py::arg("perturb_factor"))
      .def("split_by_count", &PyClass::SplitByCount, py::arg("state_occs"),
           py::arg("target_components"), py::arg("perturb_factor"),
           py::arg("power"), py::arg("min_count"))
      .def("merge_by_count", &PyClass::MergeByCount, py::arg("state_occs"),
           py::arg("target_components"), py::arg("power"), py::arg("min_count"))
      .def("compute_gconsts", &PyClass::ComputeGconsts)
      .def("log_likelihood", &PyClass::LogLikelihood, py::arg("pdf_index"),
           py::arg("data"))
      .def(
          "get_pdf",
          [](PyClass &self, int32_t pdf_index) -> DiagGmm & {
            return self.GetPdf(pdf_index);
          },
          py::arg("pdf_index"), py::return_value_policy::reference)
      .def("get_gaussian_mean", &PyClass::GetGaussianMean, py::arg("pdf_index"),
           py::arg("gauss"))
      .def("get_gaussian_variance", &PyClass::GetGaussianVariance,
           py::arg("pdf_index"), py::arg("gauss"))
      .def("set_gaussian_mean", &PyClass::SetGaussianMean, py::arg("pdf_index"),
           py::arg("gauss_index"), py::arg("in"));
}

}  // namespace khg
