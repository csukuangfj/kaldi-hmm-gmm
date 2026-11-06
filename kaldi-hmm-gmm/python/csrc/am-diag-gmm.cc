// kaldi-hmm-gmm/python/csrc/am-diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/am-diag-gmm.h"

#include <memory>
#include <tuple>
#include <utility>

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

void PybindAmDiagGmm(py::module *m) {
  using PyClass = AmDiagGmm;
  py::class_<PyClass>(*m, "AmDiagGmm")
      .def(py::init<>())
      .def_property_readonly("dim", &PyClass::Dim)
      .def_property_readonly("num_pdfs", &PyClass::NumPdfs)
      .def_property_readonly("num_gauss", &PyClass::NumGauss)
      .def("num_gauss_in_pdf", &PyClass::NumGaussInPdf, py::arg("pdf_index"))
      .def("init", &PyClass::Init, py::arg("proto"), py::arg("num_pdfs"), py::call_guard<py::gil_scoped_release>())
      .def("add_pdf", &PyClass::AddPdf, py::arg("gmm"), py::call_guard<py::gil_scoped_release>())
      .def("copy_from_am_diag_gmm", &PyClass::CopyFromAmDiagGmm,
           py::arg("other"),
           py::call_guard<py::gil_scoped_release>())
      .def("split_pdf", &PyClass::SplitPdf, py::arg("pdf_idx"),
           py::arg("target_components"), py::arg("perturb_factor"),
           py::call_guard<py::gil_scoped_release>())
      .def("split_by_count", &PyClass::SplitByCount, py::arg("state_occs"),
           py::arg("target_components"), py::arg("perturb_factor"),
           py::arg("power"), py::arg("min_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("merge_by_count", &PyClass::MergeByCount, py::arg("state_occs"),
           py::arg("target_components"), py::arg("power"), py::arg("min_count"),
           py::call_guard<py::gil_scoped_release>())
      .def("compute_gconsts", &PyClass::ComputeGconsts, py::call_guard<py::gil_scoped_release>())
      .def("log_likelihood", &PyClass::LogLikelihood, py::arg("pdf_index"),
           py::arg("data"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "get_pdf",
          [](PyClass &self, int32_t pdf_index) -> DiagGmm & {
            return self.GetPdf(pdf_index);
          },
          py::arg("pdf_index"), py::return_value_policy::reference)
      .def("get_gaussian_mean", &PyClass::GetGaussianMean, py::arg("pdf_index"),
           py::arg("gauss"),
           py::call_guard<py::gil_scoped_release>())
      .def("get_gaussian_variance", &PyClass::GetGaussianVariance,
           py::arg("pdf_index"), py::arg("gauss"),
           py::call_guard<py::gil_scoped_release>())
      .def("set_gaussian_mean", &PyClass::SetGaussianMean, py::arg("pdf_index"),
           py::arg("gauss_index"), py::arg("in"),
           py::call_guard<py::gil_scoped_release>())
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

  // kaldi writes .mdl files as TransitionModel followed by AmDiagGmm on the
  // same stream (see ../kaldi/src/gmmbin/gmm-est.cc).
  m->def(
      "write_model",
      [](const std::string &filename, const TransitionModel &trans_model,
         const AmDiagGmm &am_gmm, bool binary) {
        kaldiio::Output ko(filename, binary);
        trans_model.Write(ko.Stream(), binary);
        am_gmm.Write(ko.Stream(), binary);
      },
      py::arg("filename"), py::arg("trans_model"), py::arg("am_gmm"),
      py::arg("binary") = true,
      "Writes a kaldi-compatible .mdl file (TransitionModel + AmDiagGmm).");

  m->def(
      "read_model",
      [](const std::string &filename)
          -> std::tuple<std::unique_ptr<TransitionModel>,
                        std::unique_ptr<AmDiagGmm>> {
        bool binary = true;
        kaldiio::Input ki(filename, &binary);
        auto trans_model = std::make_unique<TransitionModel>();
        trans_model->Read(ki.Stream(), binary);
        auto am_gmm = std::make_unique<AmDiagGmm>();
        am_gmm->Read(ki.Stream(), binary);
        return std::make_tuple(std::move(trans_model), std::move(am_gmm));
      },
      py::arg("filename"),
      "Reads a kaldi .mdl file; returns (TransitionModel, AmDiagGmm).");
}

}  // namespace khg
