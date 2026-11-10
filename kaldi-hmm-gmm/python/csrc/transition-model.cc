// kaldi-hmm-gmm/python/csrc/transition-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/transition-model.h"

#include <string>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/transition-model.h"

#include "kaldi-decoder/csrc/transition-information.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

static void PybindMleTransitionUpdateConfig(py::module *m) {
  using PyClass = MleTransitionUpdateConfig;
  py::class_<PyClass>(*m, "MleTransitionUpdateConfig")
      .def(py::init<float, float, bool>(), py::arg("floor") = 0.01,
           py::arg("mincount") = 5.0, py::arg("share_for_pdfs") = false, py::call_guard<py::gil_scoped_release>())
      .def_readwrite("floor", &PyClass::floor)
      .def_readwrite("mincount", &PyClass::mincount)
      .def_readwrite("share_for_pdfs", &PyClass::share_for_pdfs);
}

static void PybindMleTransitionModelTuple(py::module *m) {
  using PyClass = TransitionModel::Tuple;
  py::class_<PyClass>(*m, "TransitionModelTuple")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>())
      .def(py::init<int32_t, int32_t, int32_t, int32_t>(), py::arg("phone"),
           py::arg("hmm_state"), py::arg("forward_pdf"),
           py::arg("self_loop_pdf"), py::call_guard<py::gil_scoped_release>())
      .def("__str__",
           [](const PyClass &self) -> std::string {
             std::ostringstream os;
             os << "TransitionModelTuple(";
             os << "phone=" << self.phone << ",";
             os << "hmm_state=" << self.hmm_state << ",";
             os << "forward_pdf=" << self.forward_pdf << ",";
             os << "self_loop_pdf=" << self.self_loop_pdf << ")";
             return os.str();
           })
      .def("__eq__",
           [](const PyClass &self, const PyClass &other) -> bool {
             return self.phone == other.phone &&
                    self.hmm_state == other.hmm_state &&
                    self.forward_pdf == other.forward_pdf &&
                    self.self_loop_pdf == other.self_loop_pdf;
           })
      .def(py::pickle(
          [](const PyClass &self) -> py::tuple {
            return py::make_tuple(self.phone, self.hmm_state, self.forward_pdf,
                                  self.self_loop_pdf);
          },
          [](const py::tuple &t) -> PyClass {
            auto phone = t[0].cast<int32_t>();
            auto hmm_state = t[1].cast<int32_t>();
            auto forward_pdf = t[2].cast<int32_t>();
            auto self_loop_pdf = t[3].cast<int32_t>();
            return {phone, hmm_state, forward_pdf, self_loop_pdf};
          }));
}

void PybindTransitionModel(py::module *m) {
  PybindMleTransitionUpdateConfig(m);
  PybindMleTransitionModelTuple(m);

  using PyClass = TransitionModel;
  // module_local: the TransitionInformation base is registered module_local
  // in this module; a module_local base requires module_local derived
  // classes.
  py::class_<PyClass, kaldi_decoder::TransitionInformation>(
      *m, "TransitionModel", py::module_local())
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>())
      .def(py::init<const ContextDependencyInterface &, const HmmTopology &>(),
           py::arg("ctx_dep"), py::arg("hmm_topo"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("topo", &PyClass::GetTopo)
      .def_property_readonly("tuples", &PyClass::GetTuples)
      .def_property_readonly("state2id", &PyClass::GetState2Id)
      .def_property_readonly("id2state", &PyClass::GetId2State)
      .def_property_readonly("id2pdf_id", &PyClass::GetId2PdfId)
      .def_property_readonly(
          "log_probs",
          [](const PyClass &self) {
            return std::vector<float>(
                self.GetLogProbs().Data(),
                self.GetLogProbs().Data() + self.GetLogProbs().Dim());
          })
      .def_property_readonly("non_self_loop_log_probs",
                             [](const PyClass &self) {
                               return std::vector<float>(
                                   self.GetNonSelfLoopLogProbs().Data(),
                                   self.GetNonSelfLoopLogProbs().Data() +
                                       self.GetNonSelfLoopLogProbs().Dim());
                             })
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
           [](const PyClass &self) -> DoubleVector {
             DoubleVector stats;
             self.InitStats(&stats);
             return stats;
           },
           py::call_guard<py::gil_scoped_release>())
      .def(
          "accumulate",
          [](PyClass &self, float prob, int32_t trans_id, DoubleVector *stats) {
            self.Accumulate(prob, trans_id, stats);
            return *stats;
          },
          py::arg("prob"), py::arg("trans_id"), py::arg("stats"))
      .def("mle_update",
           [](PyClass &self, const DoubleVector &stats,
              const MleTransitionUpdateConfig &cfg) -> std::pair<float, float> {
             float objf_impr_out = 0;
             float count_out = 0;
             self.MleUpdate(stats, cfg, &objf_impr_out, &count_out);
             return std::make_pair(objf_impr_out, count_out);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("pair_to_transition_id", &PyClass::PairToTransitionId,
           py::arg("trans_state"), py::arg("trans_index"))
      .def("self_loop_of", &PyClass::SelfLoopOf, py::arg("trans_state"))
      .def("transition_state_to_phone", &PyClass::TransitionStateToPhone,
           py::arg("trans_state"))
      .def("transition_state_to_hmm_state",
           &PyClass::TransitionStateToHmmState, py::arg("trans_state"))
      .def(
          "read",
          [](PyClass &self, const std::string &filename) {
            bool binary_in = true;
            kaldiio::Input ki(filename, &binary_in);
            self.Read(ki.Stream(), binary_in);
          },
          py::arg("filename"),
          "Reads the TransitionModel from a kaldi-format file (binary or "
          "text). The stream starts at the <TransitionModel> token, e.g. the "
          "beginning of a kaldi .mdl file.", py::call_guard<py::gil_scoped_release>())
      .def(
          "write",
          [](const PyClass &self, const std::string &filename, bool binary) {
            kaldiio::Output ko(filename, binary);
            self.Write(ko.Stream(), binary);
          },
          py::arg("filename"), py::arg("binary") = true,
          "Writes the TransitionModel to a kaldi-format file.", py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const PyClass &self) -> py::tuple {
            std::vector<float> log_probs{
                self.GetLogProbs().Data(),
                self.GetLogProbs().Data() + self.GetLogProbs().Dim()};

            std::vector<float> non_self_loop_log_probs{
                self.GetNonSelfLoopLogProbs().Data(),
                self.GetNonSelfLoopLogProbs().Data() +
                    self.GetNonSelfLoopLogProbs().Dim()};

            return py::make_tuple(self.GetTuples(), self.GetTopo(),
                                  self.GetState2Id(), self.GetId2State(),
                                  self.GetId2PdfId(), self.NumPdfs(), log_probs,
                                  non_self_loop_log_probs);
          },
          [](const py::tuple &t) -> PyClass {
            auto tuples = t[0].cast<std::vector<PyClass::Tuple>>();
            auto hmm_topo = t[1].cast<HmmTopology>();
            auto state2id = t[2].cast<std::vector<int32_t>>();
            auto id2state = t[3].cast<std::vector<int32_t>>();
            auto id2pdf_id = t[4].cast<std::vector<int32_t>>();
            auto num_pdfs = t[5].cast<int32_t>();
            auto log_probs = t[6].cast<std::vector<float>>();
            auto non_self_loop_log_probs = t[7].cast<std::vector<float>>();

            return {tuples,    hmm_topo, state2id,  id2state,
                    id2pdf_id, num_pdfs, log_probs, non_self_loop_log_probs};
          }));

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
