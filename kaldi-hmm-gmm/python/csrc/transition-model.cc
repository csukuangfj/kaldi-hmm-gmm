// kaldi-hmm-gmm/python/csrc/transition-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/transition-model.h"

#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "torch/torch.h"

namespace khg {

static void PybindMleTransitionUpdateConfig(py::module *m) {
  using PyClass = MleTransitionUpdateConfig;
  py::class_<PyClass>(*m, "MleTransitionUpdateConfig")
      .def(py::init<float, float, bool>(), py::arg("floor") = 0.01,
           py::arg("mincount") = 5.0, py::arg("share_for_pdfs") = false)
      .def_readwrite("floor", &PyClass::floor)
      .def_readwrite("mincount", &PyClass::mincount)
      .def_readwrite("share_for_pdfs", &PyClass::share_for_pdfs);
}

void PybindTransitionModel(py::module *m) {
  PybindMleTransitionUpdateConfig(m);

  using PyClass = TransitionModel;
  py::class_<PyClass, TransitionInformation>(*m, "TransitionModel")
      .def(py::init<>())
      .def(py::init<const ContextDependencyInterface &, const HmmTopology &>(),
           py::arg("ctx_dep"), py::arg("hmm_topo"))
      .def_property_readonly("topo", &PyClass::GetTopo)
      .def_property_readonly("phones", &PyClass::GetPhones)
      .def_property_readonly("num_transition_states",
                             &PyClass::NumTransitionStates)
      .def("__str__",
           [](const PyClass &self) -> std::string {
             std::ostringstream os;
             bool binary = false;
             self.Write(os, binary);
             return os.str();
           })
      .def("init_stats",
           [](const PyClass &self) -> torch::Tensor {
             torch::Tensor stats;
             self.InitStats(&stats);
             return stats;
           })
      .def("accumulate", &PyClass::Accumulate, py::arg("prob"),
           py::arg("trans_id"), py::arg("stats"))
      .def("mle_update",
           [](PyClass &self, torch::Tensor stats,
              const MleTransitionUpdateConfig &cfg) -> std::pair<float, float> {
             float objf_impr_out = 0;
             float count_out = 0;
             self.MleUpdate(stats, cfg, &objf_impr_out, &count_out);
             return std::make_pair(objf_impr_out, count_out);
           });

  m->def(
      "get_pdfs_for_phones",
      [](const TransitionModel &trans_model, const std::vector<int32_t> &phones)
          -> std::pair<bool, std::vector<int32_t>> {
        std::vector<int32_t> pdfs;
        bool is_unique = GetPdfsForPhones(trans_model, phones, &pdfs);
        return std::make_pair(is_unique, pdfs);
      });
}

}  // namespace khg
