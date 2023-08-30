// kaldi-hmm-gmm/csrc/determinize-lattice-pruned.h

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen
// Copyright (c)  2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/lat/determinize-lattice-pruned.h

#ifndef KALDI_HMM_GMM_CSRC_DETERMINIZE_LATTICE_PRUNED_H_
#define KALDI_HMM_GMM_CSRC_DETERMINIZE_LATTICE_PRUNED_H_

#include <string>

#include "fst/fst.h"
#include "fst/fstlib.h"

namespace khg {

struct DeterminizeLatticePrunedOptions {
  float delta;  // A small offset used to measure equality of weights.
  int max_mem;  // If >0, determinization will fail and return false
  // when the algorithm's (approximate) memory consumption crosses this
  // threshold.
  int max_loop;  // If >0, can be used to detect non-determinizable input
  // (a case that wouldn't be caught by max_mem).
  int max_states;
  int max_arcs;
  float retry_cutoff;
  DeterminizeLatticePrunedOptions(float delta = fst::kDelta, int max_mem = -1,
                                  int max_loop = -1, int max_states = -1,
                                  int max_arcs = -1, float retry_cutoff = 0.5)
      : delta(delta),
        max_mem(max_mem),
        max_loop(max_loop),
        max_states(max_states),
        max_arcs(max_arcs),
        retry_cutoff(retry_cutoff) {}

  std::string ToString() const {
    std::ostringstream os;

    os << "DeterminizeLatticePrunedOptions(";
    os << "delta=" << delta << ", ";
    os << "max_mem=" << max_mem << ", ";
    os << "max_loop=" << max_loop << ", ";
    os << "max_states=" << max_states << ", ";
    os << "max_arcs=" << max_arcs << ", ";
    os << "retry_cutoff=" << retry_cutoff << ")";

    return os.str();
  }
};

struct DeterminizeLatticePhonePrunedOptions {
  // delta: a small offset used to measure equality of weights.
  // Tolerance used in determinization
  float delta;
  // max_mem: if > 0, determinization will fail and return false when the
  // algorithm's (approximate) memory consumption crosses this threshold.
  int max_mem;
  // phone_determinize: if true, do a first pass determinization on both phones
  // and words.
  bool phone_determinize;
  // word_determinize: if true, do a second pass determinization on words only.
  bool word_determinize;
  // minimize: if true, push and minimize after determinization.
  bool minimize;

  DeterminizeLatticePhonePrunedOptions(float delta = fst::kDelta,
                                       int32_t max_mem = 50000000,
                                       bool phone_determinize = true,
                                       bool word_determinize = true,
                                       bool minimize = false)
      : delta(delta),
        max_mem(max_mem),
        phone_determinize(phone_determinize),
        word_determinize(word_determinize),
        minimize(minimize) {}

  std::string ToString() const {
    std::ostringstream os;

    os << "DeterminizeLatticePhonePrunedOptions(";
    os << "delta=" << delta << ", ";

    os << "max_mem=" << max_mem << ", ";

    os << "phone_determinize=" << (phone_determinize ? "True" : "False")
       << ", ";

    os << "word_determinize=" << (word_determinize ? "True" : "False") << ", ";

    os << "minimize=" << (minimize ? "True" : "False") << ")";

    return os.str();
  }
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DETERMINIZE_LATTICE_PRUNED_H_
