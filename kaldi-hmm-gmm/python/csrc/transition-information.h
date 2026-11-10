// kaldi-hmm-gmm/python/csrc/transition-information.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Registration of kaldi-decoder's TransitionInformation in THIS module
// (module_local): kaldi-hmm-gmm is self-contained at the Python level and
// does not depend on the kaldi-decoder Python package.

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_TRANSITION_INFORMATION_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_TRANSITION_INFORMATION_H_

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

namespace khg {

void PybindTransitionInformation(py::module *m);

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_TRANSITION_INFORMATION_H_
