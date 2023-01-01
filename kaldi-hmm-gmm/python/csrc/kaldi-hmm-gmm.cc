// kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

#include "kaldi-hmm-gmm/python/csrc/context-dep.h"
#include "kaldi-hmm-gmm/python/csrc/tree-renderer.h"

namespace khg {

PYBIND11_MODULE(_kaldi_hmm_gmm, m) {
  m.doc() = "pybind11 binding of kaldi-hmm-gmm";
  PybinTreeRenderer(&m);
  PybinContextDep(&m);
}

}  // namespace khg
