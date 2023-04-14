// kaldi-hmm-gmm/csrc/cluster-utils.h
//
// Copyright 2012   Arnab Ghoshal
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CLUSTER_UTILS_H_
#define KALDI_HMM_GMM_CSRC_CLUSTER_UTILS_H_

#include <stdint.h>

#include <vector>

#include "kaldi-hmm-gmm/csrc/clusterable-classes.h"

namespace khg {

struct RefineClustersOptions {
  int32_t num_iters = 100;  // must be >= 0.  If zero, does nothing.
  int32_t top_n = 5;        // must be >= 2.
  RefineClustersOptions() = default;
  RefineClustersOptions(int32_t num_iters_in, int32_t top_n_in)
      : num_iters(num_iters_in), top_n(top_n_in) {}
};

struct ClusterKMeansOptions {
  RefineClustersOptions refine_cfg;
  int32_t num_iters = 20;
  int32_t num_tries = 2;  // if >1, try whole procedure >once and pick best.
  bool verbose = true;
  ClusterKMeansOptions() = default;
  ClusterKMeansOptions(const RefineClustersOptions &refine_cfg,
                       int32_t num_iters, int32_t num_tries, bool verbose)
      : refine_cfg(refine_cfg),
        num_iters(num_iters),
        num_tries(num_tries),
        verbose(verbose) {}
};

/** ClusterKMeans is a K-means-like clustering algorithm. It starts with
 *  pseudo-random initialization of points to clusters and uses RefineClusters
 *  to iteratively improve the cluster assignments.  It does this for
 *  multiple iterations and picks the result with the best objective function.
 *
 *
 *  ClusterKMeans implicitly uses Rand(). It will not necessarily return
 *  the same value on different calls.  Use sRand() if you want consistent
 *  results.
 *  The algorithm used in ClusterKMeans is a "k-means-like" algorithm that tries
 *  to be as efficient as possible.  Firstly, since the algorithm it uses
 *  includes random initialization, it tries the whole thing cfg.num_tries times
 *  and picks the one with the best objective function.  Each try, it does as
 *  follows: it randomly initializes points to clusters, and then for
 *  cfg.num_iters iterations it calls RefineClusters().  The options to
 *  RefineClusters() are given by cfg.refine_cfg.  Calling RefineClusters once
 *  will always be at least as good as doing one iteration of reassigning points
 *  to clusters, but will generally be quite a bit better (without taking too
 *  much extra time).
 *
 *  @param points [in]  points to be clustered (must be all non-NULL).
 *  @param num_clust [in] number of clusters requested (it will always return
 * exactly this many, or will fail if num_clust > points.size()).
 *  @param clusters_out [out] may be NULL; if non-NULL, should be empty when
 * called. Will be set to a vector of statistics corresponding to the output
 * clusters.
 *  @param assignments_out [out] may be NULL; if non-NULL, will be set to a
 * vector of same size as "points", which says for each point which cluster it
 * is assigned to.
 *  @param cfg [in] configuration class specifying options to the algorithm.
 *  @return Returns the objective function improvement versus everything being
 *     in the same cluster.
 *
 */
float ClusterKMeans(const std::vector<Clusterable *> &points,
                    int32_t num_clust,  // exact number of clusters
                    std::vector<Clusterable *> *clusters_out,  // may be nullptr
                    std::vector<int32_t> *assignments_out,     // may be nullptr
                    const ClusterKMeansOptions &cfg = ClusterKMeansOptions());

/// Returns the total objective function after adding up all the
/// statistics in the vector (pointers may be NULL).
float SumClusterableObjf(const std::vector<Clusterable *> &vec);

/// Returns the total normalizer (usually count) of the cluster (pointers may be
/// NULL).
float SumClusterableNormalizer(const std::vector<Clusterable *> &vec);

/// Sums stats (ptrs may be NULL). Returns NULL if no non-NULL stats present.
Clusterable *SumClusterable(const std::vector<Clusterable *> &vec);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTER_UTILS_H_
