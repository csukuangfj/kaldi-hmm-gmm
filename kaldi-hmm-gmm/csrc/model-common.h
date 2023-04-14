// kaldi-hmm-gmm/csrc/model-common.h
//
// Copyright 2009-2012  Saarland University;  Microsoft Corporation;
//                      Johns Hopkins University (author: Daniel Povey)

//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_
#define KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_
#include <cstdint>

namespace khg {

enum GmmUpdateFlags {
  kGmmMeans = 0x001,        // m
  kGmmVariances = 0x002,    // v
  kGmmWeights = 0x004,      // w
  kGmmTransitions = 0x008,  // t ... not really part of GMM.
  kGmmAll = 0x00F           // a
};

typedef uint16_t GmmFlagsType;  ///< Bitwise OR of the above flags.

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_
