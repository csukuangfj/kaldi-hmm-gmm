// kaldi-hmm-gmm/csrc/utils.h
//
// Copyright 2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_UTILS_H_
#define KALDI_HMM_GMM_CSRC_UTILS_H_

#include "torch/script.h"

// Return a shallow copy of the i-th row of a 2-D matrix.
// @param m A 2-D tensor
// @param i The i-th row to return.
inline torch::Tensor Row(torch::Tensor m, int32_t i) {
  return m.slice(/*dim*/ 0, i, i + 1);
}

#endif  // KALDI_HMM_GMM_CSRC_UTILS_H_
