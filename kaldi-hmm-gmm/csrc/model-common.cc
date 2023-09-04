// kaldi-hmm-gmm/csrc/model-common.cc
//
// Copyright 2009-2011  Microsoft Corporation

//                2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/csrc/model-common.h"

#include <cmath>
#include <queue>
#include <string>

#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

struct CountStats {
  CountStats(int32_t p, int32_t n, float occ)
      : pdf_index(p), num_components(n), occupancy(occ) {}

  int32_t pdf_index;
  int32_t num_components;
  float occupancy;
  bool operator<(const CountStats &other) const {
    return occupancy / (num_components + 1.0e-10) <
           other.occupancy / (other.num_components + 1.0e-10);
  }
};

void GetSplitTargets(const FloatVector &state_occs,  // 1-D float tensor
                     int32_t target_components, float power, float min_count,
                     std::vector<int32_t> *targets) {
  std::priority_queue<CountStats> split_queue;
  int32_t num_pdfs = state_occs.size();

  for (int32_t pdf_index = 0; pdf_index < num_pdfs; ++pdf_index) {
    float occ = pow(state_occs[pdf_index], power);
    // initialize with one Gaussian per PDF, to put a floor
    // of 1 on the #Gauss
    split_queue.push(CountStats(pdf_index, 1, occ));
  }

  for (int32_t num_gauss = num_pdfs; num_gauss < target_components;) {
    CountStats state_to_split = split_queue.top();
    if (state_to_split.occupancy == 0) {
      KHG_WARN << "Could not split up to " << target_components
               << " due to min-count = " << min_count
               << " (or no counts at all)\n";
      break;
    }
    split_queue.pop();
    float orig_occ = state_occs[state_to_split.pdf_index];

    if ((state_to_split.num_components + 1) * min_count >= orig_occ) {
      state_to_split.occupancy = 0;  // min-count active -> disallow splitting
      // this state any more by setting occupancy = 0.
    } else {
      ++state_to_split.num_components;
      ++num_gauss;
    }
    split_queue.push(state_to_split);
  }
  targets->resize(num_pdfs);

  while (!split_queue.empty()) {
    int32_t pdf_index = split_queue.top().pdf_index;
    int32_t pdf_tgt_comp = split_queue.top().num_components;
    (*targets)[pdf_index] = pdf_tgt_comp;
    split_queue.pop();
  }
}

GmmFlagsType AugmentGmmFlags(GmmFlagsType flags) {
  KHG_ASSERT((flags & ~kGmmAll) ==
             0);  // make sure only valid flags are present.
  if (flags & kGmmVariances) flags |= kGmmMeans;
  if (flags & kGmmMeans) flags |= kGmmWeights;
  if (!(flags & kGmmWeights)) {
    KHG_WARN << "Adding in kGmmWeights (\"w\") to empty flags.";
    flags |= kGmmWeights;  // Just add this in regardless:
    // if user wants no stats, this will stop programs from crashing due to dim
    // mismatches.
  }
  return flags;
}

// CharToString prints the character in a human-readable form, for debugging.
std::string CharToString(const char &c) {
  char buf[20];

  if (std::isprint(c)) {
    std::snprintf(buf, sizeof(buf), "\'%c\'", c);
  } else {
    std::snprintf(buf, sizeof(buf), "[character %d]", static_cast<int32_t>(c));
  }

  return buf;
}

GmmFlagsType StringToGmmFlags(const std::string &str) {
  GmmFlagsType flags = 0;
  for (const auto c : str) {
    switch (c) {
      case 'm':
        flags |= kGmmMeans;
        break;
      case 'v':
        flags |= kGmmVariances;
        break;
      case 'w':
        flags |= kGmmWeights;
        break;
      case 't':
        flags |= kGmmTransitions;
        break;
      case 'a':
        flags |= kGmmAll;
        break;
      default:
        KHG_ERR << "Invalid element " << CharToString(c)
                << " of GmmFlagsType option string " << str;
    }
  }
  return flags;
}

std::string GmmFlagsToString(GmmFlagsType flags) {
  std::string ans;
  if (flags & kGmmMeans) {
    ans += "m";
  }

  if (flags & kGmmVariances) {
    ans += "v";
  }

  if (flags & kGmmWeights) {
    ans += "w";
  }

  if (flags & kGmmTransitions) {
    ans += "t";
  }

  return ans;
}

}  // namespace khg
