// kaldi-hmm-gmm/python/csrc/add-self-loops.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_ADD_SELF_LOOPS_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_ADD_SELF_LOOPS_H_

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

namespace khg {

void PybindAddSelfLoops(py::module *m);

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_ADD_SELF_LOOPS_H_
