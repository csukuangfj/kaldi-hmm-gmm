// kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.cc
//
// Copyright (c)  2022  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/kaldi-hmm-gmm.h"

#include "kaldi-hmm-gmm/python/csrc/add-self-loops.h"
#include "kaldi-hmm-gmm/python/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/python/csrc/cluster-utils.h"
#include "kaldi-hmm-gmm/python/csrc/clusterable-classes.h"
#include "kaldi-hmm-gmm/python/csrc/context-dep.h"
#include "kaldi-hmm-gmm/python/csrc/decodable-am-diag-gmm.h"
#include "kaldi-hmm-gmm/python/csrc/decodable-itf.h"
#include "kaldi-hmm-gmm/python/csrc/decoder-wrappers.h"
#include "kaldi-hmm-gmm/python/csrc/determinize-lattice-pruned.h"
#include "kaldi-hmm-gmm/python/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/python/csrc/eigen-test.h"
#include "kaldi-hmm-gmm/python/csrc/event-map.h"
#include "kaldi-hmm-gmm/python/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/python/csrc/hmm-utils.h"
#include "kaldi-hmm-gmm/python/csrc/lattice-faster-decoder.h"
#include "kaldi-hmm-gmm/python/csrc/lattice-simple-decoder.h"
#include "kaldi-hmm-gmm/python/csrc/mle-am-diag-gmm.h"
#include "kaldi-hmm-gmm/python/csrc/mle-diag-gmm.h"
#include "kaldi-hmm-gmm/python/csrc/model-common.h"
#include "kaldi-hmm-gmm/python/csrc/training-graph-compiler.h"
#include "kaldi-hmm-gmm/python/csrc/transition-information.h"
#include "kaldi-hmm-gmm/python/csrc/transition-model.h"
#include "kaldi-hmm-gmm/python/csrc/tree-renderer.h"

namespace khg {

PYBIND11_MODULE(_kaldi_hmm_gmm, m) {
  m.doc() = "pybind11 binding of kaldi-hmm-gmm";
  PybinTreeRenderer(&m);
  PybindContextDep(&m);
  PybindEventMap(&m);

  PybindModelCommon(&m);
  PybindClusterUtils(&m);
  PybindClusterableClass(&m);
  PybindDiagGmm(&m);

  PybindAmDiagGmm(&m);
  PybindHmmUtils(&m);
  PybindHmmTopology(&m);
  PybindTransitionInformation(&m);
  PybindTransitionModel(&m);

  PybindTrainingGraphCompiler(&m);
  PybindMleDiagGmm(&m);
  PybindMleAmDiagGmm(&m);

  PybindDecodableItf(&m);
  PybindDecodableAmDiagGmm(&m);
  PybindAddSelfLoops(&m);

  PybindDeterminizeLatticePruned(&m);
  PybindLatticeFasterDecoder(&m);
  PybindLatticeSimpleDecoder(&m);

  PybindDecoderWrappers(&m);

  PybindEigenTest(&m);
}

}  // namespace khg
