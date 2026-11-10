// kaldi-hmm-gmm/python/csrc/decodable-itf.h
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Registration of kaldi-decoder's DecodableInterface in THIS module
// (module_local): kaldi-hmm-gmm is self-contained at the Python level and
// does not depend on the kaldi-decoder Python package.

#ifndef KALDI_HMM_GMM_PYTHON_CSRC_DECODABLE_ITF_H_
#define KALDI_HMM_GMM_PYTHON_CSRC_DECODABLE_ITF_H_

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

namespace khg {

void PybindDecodableItf(py::module *m);

}

#endif  // KALDI_HMM_GMM_PYTHON_CSRC_DECODABLE_ITF_H_
