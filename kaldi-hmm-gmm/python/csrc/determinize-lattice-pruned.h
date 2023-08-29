// kaldi-hmm-gmm/python/csrc/determinize-lattice-pruned.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_DETERMINIZE_LATTICE_PRUNED_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_DETERMINIZE_LATTICE_PRUNED_H_

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

namespace khg {

void PybindDeterminizeLatticePruned(py::module *m);

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_DETERMINIZE_LATTICE_PRUNED_H_
