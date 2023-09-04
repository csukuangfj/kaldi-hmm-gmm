// kaldi-hmm-gmm/python/csrc/mle-diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/mle-diag-gmm.h"

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/mle-diag-gmm.h"

namespace khg {

static void PybindMleDiagGmmOptions(py::module *m) {
  using PyClass = MleDiagGmmOptions;
  py::class_<PyClass>(*m, "MleDiagGmmOptions")
      .def(py::init([](float min_gaussian_weight, float min_gaussian_occupancy,
                       double min_variance, bool remove_low_count_gaussians) {
             auto ans = std::make_unique<PyClass>();
             ans->min_gaussian_weight = min_gaussian_weight;
             ans->min_gaussian_occupancy = min_gaussian_occupancy;
             ans->min_variance = min_variance;
             ans->remove_low_count_gaussians = remove_low_count_gaussians;
             return ans;
           }),
           py::arg("min_gaussian_weight") = 1.0e-05,
           py::arg("min_gaussian_occupancy") = 10.0,
           py::arg("min_variance") = 0.001,
           py::arg("remove_low_count_gaussians") = true)
      .def_readwrite("min_gaussian_weight", &PyClass::min_gaussian_weight)
      .def_readwrite("min_gaussian_occupancy", &PyClass::min_gaussian_occupancy)
      .def_readwrite("min_variance", &PyClass::min_variance)
      .def_readwrite("remove_low_count_gaussians",
                     &PyClass::remove_low_count_gaussians)
      .def("__str__", &PyClass::ToString);
}

static void PybindMapDiagGmmOptions(py::module *m) {
  using PyClass = MapDiagGmmOptions;
  py::class_<PyClass>(*m, "MapDiagGmmOptions")
      .def(py::init([](float mean_tau, float variance_tau, float weight_tau) {
             auto ans = std::make_unique<PyClass>();
             ans->mean_tau = mean_tau;
             ans->variance_tau = variance_tau;
             ans->weight_tau = weight_tau;
             return ans;
           }),
           py::arg("mean_tau") = 10.0, py::arg("variance_tau") = 50.0,
           py::arg("weight_tau") = 10.0)
      .def_readwrite("mean_tau", &PyClass::mean_tau)
      .def_readwrite("variance_tau", &PyClass::variance_tau)
      .def_readwrite("weight_tau", &PyClass::weight_tau)
      .def("__str__", &PyClass::ToString);
}

void PybindMleDiagGmm(py::module *m) {
  PybindMleDiagGmmOptions(m);
  PybindMapDiagGmmOptions(m);

  using PyClass = AccumDiagGmm;
  py::class_<PyClass>(*m, "AccumDiagGmm")
      .def(py::init<>())
      .def(py::init<const DiagGmm &, GmmFlagsType>(), py::arg("gmm"),
           py::arg("flags"))

      .def(
          "resize",
          (void (PyClass::*)(int32_t, int32_t, GmmFlagsType))(&PyClass::Resize),
          py::arg("num_gauss"), py::arg("dim"), py::arg("flags"))
      .def_property_readonly("num_gauss", &PyClass::NumGauss)
      .def_property_readonly("dim", &PyClass::Dim)
      .def_property_readonly("flags", &PyClass::Flags)
      .def_property(
          "occupancy",
          [](PyClass &self) -> DoubleVector & { return self.occupancy(); },
          [](PyClass &self, const DoubleVector &m) { self.occupancy() = m; },
          py::return_value_policy::reference_internal)

      .def_property(
          "mean_accumulator",
          [](PyClass &self) -> DoubleMatrix & {
            return self.mean_accumulator();
          },
          [](PyClass &self, const DoubleMatrix &m) {
            self.mean_accumulator() = m;
          },
          py::return_value_policy::reference_internal)

      .def_property(
          "variance_accumulator",
          [](PyClass &self) -> DoubleMatrix & {
            return self.variance_accumulator();
          },
          [](PyClass &self, const DoubleMatrix &m) {
            self.variance_accumulator() = m;
          },
          py::return_value_policy::reference_internal)

      .def("set_zero", &PyClass::SetZero, py::arg("flags"))
      .def("scale", &PyClass::Scale, py::arg("f"), py::arg("flags"))
      .def("accumulate_for_component", &PyClass::AccumulateForComponent,
           py::arg("data"), py::arg("comp_index"), py::arg("weight"))
      .def("accumulate_from_posteriors", &PyClass::AccumulateFromPosteriors,
           py::arg("data"), py::arg("gauss_posteriors"))
      .def("accumulate_from_diag", &PyClass::AccumulateFromDiag, py::arg("gmm"),
           py::arg("data"), py::arg("weight"))
      .def("add_stats_for_component", &PyClass::AddStatsForComponent,
           py::arg("g"), py::arg("occ"), py::arg("x_stats"),
           py::arg("x2_stats"))
      // TODO(fangjun): Add tests for the following methods and functions
      .def("add", &PyClass::Add, py::arg("scale"), py::arg("acc"))
      .def("smooth_stats", &PyClass::SmoothStats, py::arg("tau"))
      .def("smooth_with_accum", &PyClass::SmoothWithAccum, py::arg("tau"),
           py::arg("src_acc"))
      .def("smooth_with_model", &PyClass::SmoothWithModel, py::arg("tau"),
           py::arg("src_gmm"));
  m->def(
      "mle_diag_gmm_update",
      [](const MleDiagGmmOptions &config, const AccumDiagGmm &diag_gmm_acc,
         GmmFlagsType flags,
         DiagGmm *gmm) -> std::tuple<float, float, int32_t, int32_t, int32_t> {
        float obj_change_out;
        float count_out;
        int32_t floored_elements_out;
        int32_t floored_gauss_out;
        int32_t removed_gauss_out;

        MleDiagGmmUpdate(config, diag_gmm_acc, flags, gmm, &obj_change_out,
                         &count_out, &floored_elements_out, &floored_gauss_out,
                         &removed_gauss_out);
        return std::make_tuple(obj_change_out, count_out, floored_elements_out,
                               floored_gauss_out, removed_gauss_out);
      },
      py::arg("config"), py::arg("diag_gmm_acc"), py::arg("flags"),
      py::arg("gmm"));

  m->def(
      "map_diag_gmm_update",
      [](const MapDiagGmmOptions &config, const AccumDiagGmm &diag_gmm_acc,
         GmmFlagsType flags, DiagGmm *gmm) -> std::pair<float, float> {
        float obj_change_out;
        float count_out;

        MapDiagGmmUpdate(config, diag_gmm_acc, flags, gmm, &obj_change_out,
                         &count_out);

        return std::make_pair(obj_change_out, count_out);
      },
      py::arg("config"), py::arg("diag_gmm_acc"), py::arg("flags"),
      py::arg("gmm"));

  m->def("ml_objective", &MlObjective, py::arg("gmm"), py::arg("diaggmm_acc"));
}

}  // namespace khg
