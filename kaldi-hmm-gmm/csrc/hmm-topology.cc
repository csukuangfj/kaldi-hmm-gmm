// kaldi-hmm-gmm/csrc/hmm-topology.cc
//
// Copyright (c)  2022  Xiaomi Corporation

// This file is copied and modified from
// kaldi/src/hmm/hmm-topology.cc
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"

#include "kaldi_native_io/csrc/io-funcs.h"

namespace khg {

void HmmTopology::Read(std::istream &is, bool binary) {
  kaldiio::ExpectToken(is, binary, "<Topology>");
}

}  // namespace khg
