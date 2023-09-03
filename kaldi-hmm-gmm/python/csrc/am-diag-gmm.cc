// kaldi-hmm-gmm/python/csrc/am-diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/am-diag-gmm.h"

#include <memory>

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
      .def("split_pdf", &PyClass::SplitPdf, py::arg("pdf_idx"),
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
           py::arg("gauss_index"), py::arg("in"))
      .def(py::pickle(
          [](const PyClass &self) -> py::tuple {
            int32_t num_pdfs = self.NumPdfs();
            py::tuple tuple(num_pdfs * 3);
            for (int32_t i = 0; i != num_pdfs; ++i) {
              const auto &gmm = self.GetPdf(i);
              tuple[3 * i + 0] = gmm.weights();
              tuple[3 * i + 1] = gmm.inv_vars();
              tuple[3 * i + 2] = gmm.means_invvars();
            }

            return tuple;
          },
          [](const py::tuple &t) -> std::unique_ptr<PyClass> {
            int32_t num_pdfs = t.size() / 3;
            auto ans = std::make_unique<PyClass>();

            for (int32_t i = 0; i != num_pdfs; ++i) {
              FloatVector weights = t[3 * i + 0].cast<FloatVector>();
              FloatMatrix inv_vars = t[3 * i + 1].cast<FloatMatrix>();
              FloatMatrix means_invvars = t[3 * i + 2].cast<FloatMatrix>();
              ans->AddPdf({weights, inv_vars, means_invvars});
            }
            return ans;
          }));
}

}  // namespace khg
