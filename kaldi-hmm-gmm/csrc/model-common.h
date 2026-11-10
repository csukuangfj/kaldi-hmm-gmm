// kaldi-hmm-gmm/csrc/model-common.h
//
// Copyright 2009-2012  Saarland University;  Microsoft Corporation;
//                      Johns Hopkins University (author: Daniel Povey)

//                2023  Xiaomi Corporation
//
#ifndef KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_
#define KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_

/**
 * \file model-common.h
 * \brief GMM update flags and mixture-splitting targets.
 *
 * This file is copied/modified from kaldi/src/gmm/model-common.h.
 *
 * This module collects small shared pieces of the GMM training machinery:
 *
 * - GmmUpdateFlags / GmmFlagsType: bit flags selecting which GMM parameters
 *   (means, variances, weights, transitions) an accumulation or update works
 *   on, plus the string conversions StringToGmmFlags()/GmmFlagsToString()
 *   ("mvwt"/"a" style flag strings) and AugmentGmmFlags() for sanity-fixing
 *   flag combinations.
 * - GetSplitTargets(): computes per-state numbers of mixture components
 *   (Gaussians or sub-states) for splitting, according to a power rule of the
 *   state occupancies.
 *
 * See also kaldi/src/gmm/model-common.h.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "kaldi-hmm-gmm/csrc/eigen.h"

namespace khg {

/// Flags selecting which GMM parameters are accumulated or updated; the
/// letters in parentheses are the characters used by StringToGmmFlags() and
/// GmmFlagsToString().
enum GmmUpdateFlags {
  kGmmMeans = 0x001,        ///< Update the Gaussian means (m).
  kGmmVariances = 0x002,    ///< Update the Gaussian variances (v).
  kGmmWeights = 0x004,      ///< Update the mixture weights (w).
  kGmmTransitions = 0x008,  ///< Update transition probabilities (t); not
                            ///< really part of GMM.
  kGmmAll = 0x00F           ///< Update everything (a).
};

typedef uint16_t GmmFlagsType;  ///< Bitwise OR of the above flags.

/// Get Gaussian-mixture or substate-mixture splitting targets,
/// according to a power rule (e.g. typically power = 0.2).
/// Returns targets for number of mixture components (Gaussians,
/// or sub-states), allocating the Gaussians or whatever according
/// to a power of occupancy in order to achieve the total supplied
/// "target".  The unnormalized target of a state with occupancy
/// \f$c\f$ is proportional to \f$c^{\,\text{power}}\f$; during splitting we
/// ensure that each Gaussian [or sub-state] would get a count of at least
/// "min-count", assuming counts were evenly distributed between
/// Gaussians in a state.
/// The vector "targets" will be resized to the appropriate dimension;
/// its value at input is ignored.
/// \param state_occs       1-D float vector of state occupancies (counts).
/// \param target_components Total number of mixture components to allocate
///                         over the states.
/// \param power            Exponent of the power rule (e.g. 0.2).
/// \param min_count        Minimum count per Gaussian/sub-state assumed at
///                         split time.
/// \param targets          [out] Resized to the number of states; element
///                         \c i is the number of mixture components assigned
///                         to state \c i.
void GetSplitTargets(const FloatVector &state_occs,
                     int32_t target_components, float power, float min_count,
                     std::vector<int32_t> *targets);

/// Make sure that the flags make sense, i.e. if there is variance
/// accumulation that there is also mean accumulation.
/// \param flags  Input flags.
/// \return The flags with implied bits added (e.g. kGmmVariances without
///         kGmmMeans gains kGmmMeans).
GmmFlagsType AugmentGmmFlags(GmmFlagsType flags);

/// Convert string which is some subset of "mvwt" to
/// flags.
/// \param str  String of flag letters, e.g. "mvw" or "a".
/// \return The corresponding bitwise-OR of GmmUpdateFlags.
GmmFlagsType StringToGmmFlags(const std::string &str);

/// Convert GMM flags to string
/// \param gmm_flags  Bitwise-OR of GmmUpdateFlags.
/// \return The flag letters (subset of "mvwt", or "a" for all) as a string.
std::string GmmFlagsToString(GmmFlagsType gmm_flags);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_MODEL_COMMON_H_
