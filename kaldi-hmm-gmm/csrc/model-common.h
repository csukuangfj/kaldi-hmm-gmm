// kaldi-hmm-gmm/csrc/model-common.h
//
// Copyright 2009-2012  Saarland University;  Microsoft Corporation;
//                      Johns Hopkins University (author: Daniel Povey)

//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_
#define KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "kaldi-hmm-gmm/csrc/eigen.h"

namespace khg {

enum GmmUpdateFlags {
  kGmmMeans = 0x001,        // m
  kGmmVariances = 0x002,    // v
  kGmmWeights = 0x004,      // w
  kGmmTransitions = 0x008,  // t ... not really part of GMM.
  kGmmAll = 0x00F           // a
};

typedef uint16_t GmmFlagsType;  ///< Bitwise OR of the above flags.

/// Get Gaussian-mixture or substate-mixture splitting targets,
/// according to a power rule (e.g. typically power = 0.2).
/// Returns targets for number of mixture components (Gaussians,
/// or sub-states), allocating the Gaussians or whatever according
/// to a power of occupancy in order to achieve the total supplied
/// "target".  During splitting we ensure that
/// each Gaussian [or sub-state] would get a count of at least
/// "min-count", assuming counts were evenly distributed between
/// Gaussians in a state.
/// The vector "targets" will be resized to the appropriate dimension;
/// its value at input is ignored.
void GetSplitTargets(const FloatVector &state_occs,  // 1-D float tensor
                     int32_t target_components, float power, float min_count,
                     std::vector<int32_t> *targets);

// Make sure that the flags make sense, i.e. if there is variance
// accumulation that there is also mean accumulation
GmmFlagsType AugmentGmmFlags(GmmFlagsType flags);

/// Convert string which is some subset of "mvwt" to
/// flags.
GmmFlagsType StringToGmmFlags(const std::string &str);

/// Convert GMM flags to string
std::string GmmFlagsToString(GmmFlagsType gmm_flags);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_
