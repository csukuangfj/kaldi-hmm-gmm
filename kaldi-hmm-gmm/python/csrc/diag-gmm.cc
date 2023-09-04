// kaldi-hmm-gmm/python/csrc/diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/diag-gmm.h"

#include <memory>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"

namespace khg {

void PybindDiagGmm(py::module *m) {
  using PyClass = DiagGmm;
  py::class_<PyClass>(*m, "DiagGmm")
      .def(py::init<>())
      .def(py::init<const PyClass &>(), py::arg("gmm"))
      .def(py::init<int32_t, int32_t>(), py::arg("nmix"), py::arg("dim"))
      .def(py::init<const std::vector<std::pair<float, const DiagGmm *>> &>(),
           py::arg("gmms"))

      .def("resize", &PyClass::Resize, py::arg("nmix"), py::arg("dim"))
      .def("copy_from_diag_gmm", &PyClass::CopyFromDiagGmm, py::arg("diaggmm"))
      .def("compute_gconsts", &PyClass::ComputeGconsts)
      .def("set_weights", &PyClass::SetWeights, py::arg("w"))
      .def("set_means", &PyClass::SetMeans, py::arg("m"))
      .def("set_invvars", &PyClass::SetInvVars, py::arg("inv_vars"))
      .def("log_likelihood", &PyClass::LogLikelihood, py::arg("data"),
           "Return the total loglikes in a float")
      .def(
          "log_likelihoods",
          [](const PyClass &self, const FloatVector &data) {
            FloatVector ans;
            self.LogLikelihoods(data, &ans);
            return ans;
          },
          py::arg("data"),
          "Return the loglike of each component in a 1-D tensor")
      .def_property(
          "weights",
          [](PyClass &self) -> FloatVector & { return self.weights(); },
          [](PyClass &self, const FloatVector &m) { self.SetWeights(m); },
          py::return_value_policy::reference_internal)
      .def_property_readonly("means", &PyClass::GetMeans)
      .def_property_readonly("vars", &PyClass::GetVars)
      .def_property_readonly("num_gauss", &PyClass::NumGauss)
      .def_property_readonly("dim", &PyClass::Dim)
      .def_property_readonly("valid_gconsts", &PyClass::valid_gconsts)
      .def_property_readonly("gconsts", &PyClass::gconsts)
      .def_property_readonly("means_invvars", &PyClass::means_invvars)
      .def_property_readonly("inv_vars", &PyClass::inv_vars)
      .def("set_component_weight", &PyClass::SetComponentWeight,
           py::arg("gauss"), py::arg("weight"))
      .def("set_component_mean", &PyClass::SetComponentMean, py::arg("gauss"),
           py::arg("mean"))
      .def("set_invvars_and_means", &PyClass::SetInvVarsAndMeans,
           py::arg("inv_vars"), py::arg("means"))
      .def("set_component_inv_var", &PyClass::SetComponentInvVar,
           py::arg("gauss"), py::arg("inv_var"))
      .def("get_component_mean", &PyClass::GetComponentMean, py::arg("gauss"))
      .def("get_component_variance", &PyClass::GetComponentVariance,
           py::arg("gauss"))
      .def("remove_component", &PyClass::RemoveComponent, py::arg("gauss"),
           py::arg("renorm_weights"))
      .def("remove_components", &PyClass::RemoveComponents, py::arg("gauss"),
           py::arg("renorm_weights"))
      .def(
          "split",
          [](PyClass &self, int32_t target_components,
             float perturb_factor) -> std::vector<int32_t> {
            std::vector<int32_t> history;
            self.Split(target_components, perturb_factor, &history);
            return history;
          },
          py::arg("target_components"), py::arg("perturb_factor"))
      .def(
          "merge",
          [](PyClass &self, int32_t target_components) -> std::vector<int32_t> {
            std::vector<int32_t> history;
            self.Merge(target_components, &history);
            return history;
          },
          py::arg("target_components"))
      .def("merge_kmeans", &PyClass::MergeKmeans, py::arg("target_components"),
           py::arg("cfg") = ClusterKMeansOptions())
      .def(
          "log_likelihoods_matrix",
          [](const PyClass &self, const FloatMatrix &data) {
            FloatMatrix ans;
            self.LogLikelihoodsMatrix(data, &ans);
            return ans;
          },
          py::arg("data"),
          "data is a 2-D tensor of shape (N, dim);"
          "it returns a 2-D tensor of shape (N, nmix) containing the "
          "loglike of each component")
      .def(
          "log_likelihoods_preselect",
          [](const PyClass &self, const FloatVector &data,
             const std::vector<int32_t> &indices) {
            FloatVector ans;
            self.LogLikelihoodsPreselect(data, indices, &ans);
            return ans;
          },
          py::arg("data"), py::arg("indices"))
      .def(
          "gaussian_selection_1d",
          [](const PyClass &self, const FloatVector &data,
             int32_t num_gselect) -> std::pair<float, std::vector<int32_t>> {
            std::vector<int32_t> output;
            float f = self.GaussianSelection(data, num_gselect, &output);
            return std::make_pair(f, output);
          },
          py::arg("data"), py::arg("num_gselect"))
      .def(
          "gaussian_selection_2d",
          [](const PyClass &self, const FloatMatrix &data, int32_t num_gselect)
              -> std::pair<float, std::vector<std::vector<int32_t>>> {
            std::vector<std::vector<int32_t>> output;
            float f = self.GaussianSelection(data, num_gselect, &output);
            return std::make_pair(f, output);
          },
          py::arg("data"), py::arg("num_gselect"))
      .def(
          "gaussian_selection_preselect",
          [](const PyClass &self, const FloatVector &data,
             const std::vector<int32_t> &preselect,
             int32_t num_gselect) -> std::pair<float, std::vector<int32_t>> {
            std::vector<int32_t> output;
            float f = self.GaussianSelectionPreselect(data, preselect,
                                                      num_gselect, &output);
            return std::make_pair(f, output);
          },
          py::arg("data"), py::arg("preselect"), py::arg("num_gselect"))
      .def(
          "component_posteriors",
          [](const PyClass &self,
             const FloatVector &data) -> std::pair<float, FloatVector> {
            FloatVector posteriors;
            float f = self.ComponentPosteriors(data, &posteriors);
            return std::make_pair(f, posteriors);
          },
          py::arg("data"))
      .def("component_log_likelihood", &PyClass::ComponentLogLikelihood,
           py::arg("data"), py::arg("comp_id"))
      .def("generate",
           [](const PyClass &self) {
             FloatVector data(self.Dim());
             self.Generate(&data);
             return data;
           })
      .def("perturb", &PyClass::Perturb, py::arg("perturb_factor"))
      .def("interpolate", &PyClass::Interpolate, py::arg("rho"),
           py::arg("source"), py::arg("flags") = kGmmAll)
      .def(py::pickle(
          [](const PyClass &self) -> py::tuple {
            return py::make_tuple(self.weights(), self.inv_vars(),
                                  self.means_invvars());
          },
          [](const py::tuple &t) -> std::unique_ptr<PyClass> {
            FloatVector weights = t[0].cast<FloatVector>();
            FloatMatrix inv_vars = t[1].cast<FloatMatrix>();
            FloatMatrix means_invvars = t[2].cast<FloatMatrix>();
            return std::make_unique<PyClass>(weights, inv_vars, means_invvars);
          }));
}

}  // namespace khg
