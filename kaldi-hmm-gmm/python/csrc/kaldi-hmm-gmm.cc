// kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

#include "kaldi-hmm-gmm/python/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/python/csrc/cluster-utils.h"
#include "kaldi-hmm-gmm/python/csrc/clusterable-classes.h"
#include "kaldi-hmm-gmm/python/csrc/context-dep.h"
#include "kaldi-hmm-gmm/python/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/python/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/python/csrc/model-common.h"
#include "kaldi-hmm-gmm/python/csrc/tree-renderer.h"
#include "torch/torch.h"

namespace khg {

PYBIND11_MODULE(_kaldi_hmm_gmm, m) {
  m.doc() = "pybind11 binding of kaldi-hmm-gmm";
  PybinTreeRenderer(&m);
  PybindContextDep(&m);

  PybindModelCommon(&m);
  PybindClusterUtils(&m);
  PybindClusterableClass(&m);
  PybindDiagGmm(&m);

  PybindAmDiagGmm(&m);

  PybindHmmTopology(&m);
}

}  // namespace khg
