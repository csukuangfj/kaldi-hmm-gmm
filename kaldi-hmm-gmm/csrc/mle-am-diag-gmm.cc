// kaldi-hmm-gmm/csrc/mle-am-diag-gmm.cc
//
// Copyright 2009-2011  Saarland University (Author: Arnab Ghoshal);
//                      Microsoft Corporation;  Georg Stemmer;  Yanmin Qian
//                2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/stl-utils.h"
namespace khg {

AccumAmDiagGmm::~AccumAmDiagGmm() { DeletePointers(&gmm_accumulators_); }
}  // namespace khg
