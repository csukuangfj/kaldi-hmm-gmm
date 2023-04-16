// kaldi-hmm-gmm/csrc/model-common.cc
//
// Copyright 2009-2011  Microsoft Corporation

//                2023  Xiaomi Corporation
#include "kaldi-hmm-gmm/csrc/model-common.h"

#include <cmath>
#include <queue>

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

void GetSplitTargets(torch::Tensor state_occs,  // 1-D float tensor
                     int32_t target_components, float power, float min_count,
                     std::vector<int32_t> *targets) {
  std::priority_queue<CountStats> split_queue;
  int32_t num_pdfs = state_occs.size(0);

  auto state_occs_acc = state_occs.accessor<float, 1>();

  for (int32_t pdf_index = 0; pdf_index < num_pdfs; ++pdf_index) {
    float occ = pow(state_occs_acc[pdf_index], power);
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
    float orig_occ = state_occs_acc[state_to_split.pdf_index];

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

}  // namespace khg
