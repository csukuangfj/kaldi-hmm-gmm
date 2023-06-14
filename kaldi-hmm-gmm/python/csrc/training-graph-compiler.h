// kaldi-hmm-gmm/python/csrc/training-graph-compiler.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_TRAINING_GRAPH_COMPILER_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_TRAINING_GRAPH_COMPILER_H_

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

namespace khg {

void PybindTrainingGraphCompiler(py::module *m);

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_TRAINING_GRAPH_COMPILER_H_
