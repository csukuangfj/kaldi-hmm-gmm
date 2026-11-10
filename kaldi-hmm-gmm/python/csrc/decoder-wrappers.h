// kaldi-hmm-gmm/python/csrc/decoder-wrappers.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_DECODER_WRAPPERS_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_DECODER_WRAPPERS_H_

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

namespace khg {

// Thin convenience wrappers over kaldi-decoder-core: the C++ decoder lives
// in ../kaldi-decoder; these functions keep the recipe's hot path entirely
// inside C++ (the khg-registered decodable/transition model convert
// intra-module, no cross-module python object exchange).
void PybindDecoderWrappers(py::module *m);

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_DECODER_WRAPPERS_H_
