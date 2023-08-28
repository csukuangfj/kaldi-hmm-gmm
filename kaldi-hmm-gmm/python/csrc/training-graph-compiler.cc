// kaldi-hmm-gmm/python/csrc/training-graph-compiler.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/training-graph-compiler.h"

#include <memory>
#include <vector>

#include "kaldi-hmm-gmm/csrc/training-graph-compiler.h"

namespace khg {

static void PybindTrainingGraphCompilerOptions(py::module *m) {
  using PyClass = TrainingGraphCompilerOptions;
  py::class_<PyClass>(*m, "TrainingGraphCompilerOptions")
      .def(py::init<float, float, bool>(), py::arg("transition_scale") = 1.0f,
           py::arg("self_loop_scale") = 1.0f, py::arg("reorder") = true)
      .def_readwrite("transition_scale", &PyClass::transition_scale)
      .def_readwrite("self_loop_scale", &PyClass::self_loop_scale)
      .def_readwrite("rm_eps", &PyClass::rm_eps)
      .def_readwrite("reorder", &PyClass::reorder)
      .def("__str__", &PyClass::ToString);
}

void PybindTrainingGraphCompiler(py::module *m) {
  PybindTrainingGraphCompilerOptions(m);

  using PyClass = TrainingGraphCompiler;
  py::class_<PyClass>(*m, "TrainingGraphCompiler")
      .def(py::init([](const TransitionModel &trans_model,
                       const ContextDependency &ctx_dep,
                       fst::VectorFst<fst::StdArc> *lex_fst,
                       const std::vector<int32_t> &disambig_syms,
                       const TrainingGraphCompilerOptions &opts)
                        -> std::unique_ptr<PyClass> {
             return std::make_unique<PyClass>(
                 trans_model, ctx_dep,
                 lex_fst->Copy(),  // takes ownership
                 disambig_syms, opts);
           }),
           py::arg("trans_model"), py::arg("ctx_dep"), py::arg("lex_fst"),
           py::arg("disambig_syms"), py::arg("opts"))
      .def(
          "compile_graph_from_text",
          [](PyClass &self, const std::vector<int32_t> &transcript)
              -> fst::VectorFst<fst::StdArc> {
            fst::VectorFst<fst::StdArc> out_fst;
            bool succeeded = self.CompileGraphFromText(transcript, &out_fst);
            KHG_ASSERT(succeeded);
            return out_fst;
          },
          py::arg("transcript"));
}

}  // namespace khg
