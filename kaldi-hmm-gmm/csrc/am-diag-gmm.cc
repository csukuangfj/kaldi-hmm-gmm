// kaldi-hmm-gmm/csrc/am-diag-gmm.cc
//
// Copyright 2012   Arnab Ghoshal  Johns Hopkins University (Author: Daniel
// Povey)  Karel Vesely Copyright 2009-2011  Saarland University;  Microsoft
// Corporation;
//                      Georg Stemmer
//                2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"

#include "kaldi-hmm-gmm/csrc/stl-utils.h"
namespace khg {

AmDiagGmm::~AmDiagGmm() { DeletePointers(&densities_); }

}  // namespace khg
