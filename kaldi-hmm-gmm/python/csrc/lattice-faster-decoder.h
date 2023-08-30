// kaldi-hmm-gmm/python/csrc/lattice-faster-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_LATTICE_FASTER_DECODER_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_LATTICE_FASTER_DECODER_H_

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

namespace khg {

void PybindLatticeFasterDecoder(py::module *m);

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_LATTICE_FASTER_DECODER_H_
