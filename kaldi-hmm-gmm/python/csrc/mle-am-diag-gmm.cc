// kaldi-hmm-gmm/python/csrc/mle-am-diag-gmm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/mle-am-diag-gmm.h"

#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h"

namespace khg {

void PybindMleAmDiagGmm(py::module *m) {
  using PyClass = AccumAmDiagGmm;
  py::class_<PyClass>(*m, "AccumAmDiagGmm")
      .def(py::init<>())
      .def("init",
           (void (PyClass::*)(const AmDiagGmm &, GmmFlagsType))(&PyClass::Init),
           py::arg("model"), py::arg("flags"))
      .def("init",
           (void (PyClass::*)(const AmDiagGmm &, int32_t, GmmFlagsType))(
               &PyClass::Init),
           py::arg("model"), py::arg("dim"), py::arg("flags"))
      .def("set_zero", &PyClass::SetZero, py::arg("flags"))
      .def("accumulate_for_gmm", &PyClass::AccumulateForGmm, py::arg("model"),
           py::arg("data"), py::arg("gmm_index"), py::arg("weight"))
      .def("accumulate_for_gmm_two_feats", &PyClass::AccumulateForGmmTwofeats,
           py::arg("model"), py::arg("data1"), py::arg("data2"),
           py::arg("gmm_index"), py::arg("weight"))
      .def("accumulate_from_posteriors", &PyClass::AccumulateFromPosteriors,
           py::arg("model"), py::arg("data"), py::arg("gmm_index"),
           py::arg("weight"))
      .def("accumulate_for_gaussian", &PyClass::AccumulateForGaussian,
           py::arg("am"), py::arg("data"), py::arg("gmm_index"),
           py::arg("gauss_index"), py::arg("weight"))
      .def_property_readonly("num_accs", &PyClass::NumAccs)
      .def_property_readonly("tot_stats_count", &PyClass::TotStatsCount)
      .def_property_readonly("tot_count", &PyClass::TotCount)
      .def_property_readonly("tot_log_like", &PyClass::TotLogLike)
      .def("get_acc",
           [](PyClass &self, int32_t index) { return self.GetAcc(index); })
      .def("add", &PyClass::Add, py::arg("scale"), py::arg("other"))
      .def("scale", &PyClass::Scale, py::arg("scale"))
      .def_property_readonly("dim", &PyClass::Dim);

  m->def(
      "mle_am_diag_gmm_update",
      [](const MleDiagGmmOptions &config, const AccumAmDiagGmm &amdiag_gmm_acc,
         GmmFlagsType flags, AmDiagGmm *am_gmm) -> std::pair<float, float> {
        float obj_change_out;
        float count_out;

        MleAmDiagGmmUpdate(config, amdiag_gmm_acc, flags, am_gmm,
                           &obj_change_out, &count_out);
        return std::make_pair(obj_change_out, count_out);
      },
      py::arg("config"), py::arg("amdiag_gmm_acc"), py::arg("flags"),
      py::arg("am_gmm"));

  m->def(
      "map_am_diag_gmm_update",
      [](const MapDiagGmmOptions &config, const AccumAmDiagGmm &amdiag_gmm_acc,
         GmmFlagsType flags, AmDiagGmm *am_gmm) -> std::pair<float, float> {
        float obj_change_out;
        float count_out;

        MapAmDiagGmmUpdate(config, amdiag_gmm_acc, flags, am_gmm,
                           &obj_change_out, &count_out);

        return std::make_pair(obj_change_out, count_out);
      },
      py::arg("config"), py::arg("amdiag_gmm_acc"), py::arg("flags"),
      py::arg("am_gmm"));
}

}  // namespace khg
