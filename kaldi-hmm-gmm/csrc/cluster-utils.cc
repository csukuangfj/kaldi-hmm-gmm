// kaldi-hmm-gmm/csrc/cluster-utils.cc
//
// Copyright 2012   Arnab Ghoshal
// Copyright 2009-2011  Microsoft Corporation;  Saarland University
//                2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/csrc/cluster-utils.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/log.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"

namespace khg {

typedef uint16_t uint_smaller;
// ============================================================================
// Some convenience functions used in the clustering routines
// ============================================================================

float SumClusterableObjf(const std::vector<Clusterable *> &vec) {
  float ans = 0.0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != NULL) {
      float objf = vec[i]->Objf();
      if (KALDI_ISNAN(objf)) {
        KHG_WARN << "SumClusterableObjf, NaN objf";
      } else {
        ans += objf;
      }
    }
  }
  return ans;
}
float SumClusterableNormalizer(const std::vector<Clusterable *> &vec) {
  float ans = 0.0;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] != nullptr) {
      float objf = vec[i]->Normalizer();
      if (KALDI_ISNAN(objf)) {
        KHG_WARN << "SumClusterableNormalizer, NaN objf";
      } else {
        ans += objf;
      }
    }
  }
  return ans;
}

Clusterable *SumClusterable(const std::vector<Clusterable *> &vec) {
  Clusterable *ans = nullptr;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] != nullptr) {
      if (ans == nullptr)
        ans = vec[i]->Copy();
      else
        ans->Add(*(vec[i]));
    }
  }
  return ans;
}

class RefineClusterer {
 public:
  // size used in point_info structure (we store a lot of these so don't want
  // to just make it int32_t). Also used as a time-id (cannot have more moves of
  // points, than can fit in this time). Must be big enough to store num-clust.
  typedef int32_t LocalInt;
  typedef uint_smaller ClustIndexInt;

  RefineClusterer(const std::vector<Clusterable *> &points,
                  std::vector<Clusterable *> *clusters,
                  std::vector<int32_t> *assignments,
                  const RefineClustersOptions &cfg)
      : points_(points),
        clusters_(clusters),
        assignments_(assignments),
        cfg_(cfg) {
    KHG_ASSERT(cfg_.top_n >= 2);
    num_points_ = points_.size();
    num_clust_ = static_cast<int32_t>(clusters->size());

    // so can fit clust-id in LocalInt
    if (cfg_.top_n > (int32_t)num_clust_)
      cfg_.top_n = static_cast<int32_t>(num_clust_);

    KHG_ASSERT(cfg_.top_n ==
               static_cast<int32_t>(static_cast<ClustIndexInt>(cfg_.top_n)));
    t_ = 0;
    my_clust_index_.resize(num_points_);
    // will set all PointInfo's to 0 too (they will be up-to-date).
    clust_time_.resize(num_clust_, 0);
    clust_objf_.resize(num_clust_);

    for (int32_t i = 0; i < num_clust_; ++i)
      clust_objf_[i] = (*clusters_)[i]->Objf();

    info_.resize(num_points_ * cfg_.top_n);
    ans_ = 0;
    InitPoints();
  }

  float Refine() {
    if (cfg_.top_n <= 1) return 0.0;  // nothing to do.
    Iterate();
    return ans_;
  }
  // at some point check cfg_.top_n > 1 after maxing to num_clust_.
 private:
  void InitPoint(int32_t point) {
    // Find closest clusters to this point.
    // distances are really negated objf changes, ignoring terms that don't vary
    // with the "other" cluster.

    std::vector<std::pair<float, LocalInt>> distances;
    distances.reserve(num_clust_ - 1);

    int32_t my_clust = (*assignments_)[point];
    Clusterable *point_cl = points_[point];

    for (int32_t clust = 0; clust < num_clust_; ++clust) {
      if (clust != my_clust) {
        Clusterable *tmp = (*clusters_)[clust]->Copy();
        tmp->Add(*point_cl);

        float other_clust_objf = clust_objf_[clust];
        float other_clust_plus_me_objf =
            (*clusters_)[clust]->ObjfPlus(*(points_[point]));

        float distance = other_clust_objf -
                         other_clust_plus_me_objf;  // negated delta-objf, with
                                                    // only "varying" terms.
        distances.push_back(std::make_pair(distance, (LocalInt)clust));
        delete tmp;
      }
    }

    if ((cfg_.top_n - 1 - 1) >= 0) {
      std::nth_element(distances.begin(),
                       distances.begin() + (cfg_.top_n - 1 - 1),
                       distances.end());
    }
    // top_n-1 is the # of elements we want to retain.  -1 because we need the
    // iterator that points to the end of that range (i.e. not potentially off
    // the end of the array).

    for (int32_t index = 0; index < cfg_.top_n - 1; index++) {
      point_info &info = GetInfo(point, index);
      int32_t clust = distances[index].second;
      info.clust = clust;
      float distance = distances[index].first;
      float other_clust_objf = clust_objf_[clust];
      float other_clust_plus_me_objf = -(distance - other_clust_objf);
      info.objf = other_clust_plus_me_objf;
      info.time = 0;
    }
    // now put the last element in, which is my current cluster.
    point_info &info = GetInfo(point, cfg_.top_n - 1);
    info.clust = my_clust;
    info.time = 0;
    info.objf = (*clusters_)[my_clust]->ObjfMinus(*(points_[point]));
    my_clust_index_[point] = cfg_.top_n - 1;
  }

  void InitPoints() {
    // finds, for each point, the closest cfg_.top_n clusters (including its own
    // cluster). this may be the most time-consuming step of the algorithm.
    for (int32_t p = 0; p < num_points_; ++p) InitPoint(p);
  }
  void Iterate() {
    int32_t iter, num_iters = cfg_.num_iters;
    for (iter = 0; iter < num_iters; ++iter) {
      int32_t cur_t = t_;
      for (int32_t point = 0; point < num_points_; point++) {
        if (t_ + 1 == 0) {
          KHG_WARN << "Stopping iterating at int32_t moves";
          return;  // once we use up all time points, must return-- this
                   // should rarely happen as int32_t is large.
        }
        ProcessPoint(point);
      }
      if (t_ == cur_t) break;  // nothing changed so we converged.
    }
  }

  void MovePoint(int32_t point, int32_t new_index) {
    // move point to a different cluster.
    t_++;
    int32_t old_index = my_clust_index_[point];  // index into info
    // array corresponding to current cluster.
    KHG_ASSERT(new_index < cfg_.top_n && new_index != old_index);
    point_info &old_info = GetInfo(point, old_index),
               &new_info = GetInfo(point, new_index);
    my_clust_index_[point] = new_index;  // update to new index.

    int32_t old_clust = old_info.clust, new_clust = new_info.clust;
    KHG_ASSERT((*assignments_)[point] == old_clust);
    (*assignments_)[point] = new_clust;
    (*clusters_)[old_clust]->Sub(*(points_[point]));
    (*clusters_)[new_clust]->Add(*(points_[point]));
    UpdateClust(old_clust);
    UpdateClust(new_clust);
  }

  void UpdateClust(int32_t clust) {
    KHG_ASSERT(clust < num_clust_);
    clust_objf_[clust] = (*clusters_)[clust]->Objf();
    clust_time_[clust] = t_;
  }

  void ProcessPoint(int32_t point) {
    // note: calling code uses the fact
    // that it only ever increases t_ by one.
    KHG_ASSERT(point < num_points_);
    // (1) Make sure own-cluster like is updated.
    int32_t self_index =
        my_clust_index_[point];  // index <cfg_.top_n of own cluster.
    point_info &self_info = GetInfo(point, self_index);
    int32_t self_clust = self_info.clust;  // cluster index of own cluster.
    KHG_ASSERT(self_index < cfg_.top_n);
    UpdateInfo(point, self_index);

    float own_clust_objf = clust_objf_[self_clust];
    float own_clust_minus_me_objf =
        self_info.objf;  // objf of own cluster minus self.
    // Now check the other "close" clusters and see if we want to move there.
    for (int32_t index = 0; index < cfg_.top_n; index++) {
      if (index != self_index) {
        UpdateInfo(point, index);
        point_info &other_info = GetInfo(point, index);
        float other_clust_objf = clust_objf_[other_info.clust];
        float other_clust_plus_me_objf = other_info.objf;
        float impr = other_clust_plus_me_objf + own_clust_minus_me_objf -
                     other_clust_objf - own_clust_objf;
        if (impr > 0) {  // better to switch...
          ans_ += impr;
          MovePoint(point, index);
          return;  // the stuff we precomputed at the top is invalidated now,
                   // and it's
          // easiest just to wait till next time we visit this point.
        }
      }
    }
  }

  void UpdateInfo(int32_t point, int32_t idx) {
    point_info &pinfo = GetInfo(point, idx);
    if (pinfo.time < clust_time_[pinfo.clust]) {  // it's not up-to-date...
      Clusterable *tmp_cl = (*clusters_)[pinfo.clust]->Copy();
      if (idx == my_clust_index_[point]) {
        tmp_cl->Sub(*(points_[point]));
      } else {
        tmp_cl->Add(*(points_[point]));
      }
      pinfo.time = t_;
      pinfo.objf = tmp_cl->Objf();
      delete tmp_cl;
    }
  }

  typedef struct {
    LocalInt clust;
    LocalInt time;
    float objf;  // Objf of this cluster plus this point (or minus, if own
                 // cluster).
  } point_info;

  point_info &GetInfo(int32_t point, int32_t idx) {
    KHG_ASSERT(point < num_points_ && idx < cfg_.top_n);
    int32_t i = point * cfg_.top_n + idx;
    KHG_ASSERT(i < static_cast<int32_t>(info_.size()));
    return info_[i];
  }

  const std::vector<Clusterable *> &points_;
  std::vector<Clusterable *> *clusters_;
  std::vector<int32_t> *assignments_;

  std::vector<point_info> info_;  // size is [num_points_ * cfg_.top_n].
  std::vector<ClustIndexInt>
      my_clust_index_;  // says for each point, which index 0...cfg_.top_n-1
                        // currently corresponds to its own cluster.

  std::vector<LocalInt> clust_time_;  // Modification time of cluster.
  std::vector<float> clust_objf_;     // [clust], objf for cluster.

  float ans_;  // objf improvement.

  int32_t num_clust_;
  int32_t num_points_;
  int32_t t_;
  RefineClustersOptions cfg_;  // note, we change top_n in config; don't make
                               // this member a reference member.
};

static float RefineClusters(const std::vector<Clusterable *> &points,
                            std::vector<Clusterable *> *clusters,
                            std::vector<int32_t> *assignments,
                            const RefineClustersOptions &cfg) {
  if (cfg.num_iters <= 0) {
    return 0.0;
  }  // nothing to do.

  KHG_ASSERT(clusters != nullptr && assignments != nullptr);

  KHG_ASSERT(!ContainsNullPointers(points) && !ContainsNullPointers(*clusters));

  RefineClusterer rc(points, clusters, assignments, cfg);
  KHG_LOG << "refine started";

  float ans = rc.Refine();
  KHG_LOG << "refine done";
  KHG_ASSERT(!ContainsNullPointers(*clusters));

  return ans;
}

// ============================================================================
// K-means like clustering routines
// ============================================================================

/// ClusterKMeansOnce is called internally by ClusterKMeans; it is equivalent
/// to calling ClusterKMeans with cfg.num_tries == 1.  It returns the objective
/// function improvement versus everything being in one cluster.

float ClusterKMeansOnce(const std::vector<Clusterable *> &points,
                        int32_t num_clust,
                        std::vector<Clusterable *> *clusters_out,
                        std::vector<int32_t> *assignments_out,
                        const ClusterKMeansOptions &cfg) {
  std::vector<int32_t> my_assignments;
  int32_t num_points = points.size();
  KHG_ASSERT(clusters_out != nullptr);
  KHG_ASSERT(num_points != 0);
  KHG_ASSERT(num_clust <= num_points);

  // we wouldn't know what to do with pointers in there if it is
  // not empty on entry
  KHG_ASSERT(clusters_out->empty());

  clusters_out->resize(num_clust, nullptr);
  assignments_out->resize(num_points);

  {  // This block assigns points to clusters.
    // This is done pseudo-randomly using Rand() so that
    // if we call ClusterKMeans multiple times we get different answers (so we
    // can choose the best if we want).
    int32_t skip;  // randomly choose a "skip" that's coprime to num_points.
    if (num_points == 1) {
      skip = 1;
    } else {
      // a number between 1 and num_points-1.
      skip = 1 + (Rand() % (num_points - 1));

      while (Gcd(skip, num_points) != 1) {
        // while skip is not coprime to num_points...
        if (skip == num_points - 1) skip = 0;

        skip++;  // skip is now still between 1 and num_points-1.  will cycle
                 // through
        // all of 1...num_points-1.
      }
    }
    int32_t i, j, count = 0;
    for (i = 0, j = 0; count != num_points;
         i = (i + skip) % num_points, j = (j + 1) % num_clust, count++) {
      // i cycles pseudo-randomly through all points; j skips ahead by 1 each
      // time modulo num_points. assign point i to cluster j.
      if ((*clusters_out)[j] == nullptr)
        (*clusters_out)[j] = points[i]->Copy();
      else
        (*clusters_out)[j]->Add(*(points[i]));

      (*assignments_out)[i] = j;
    }
  }

  float normalizer = SumClusterableNormalizer(*clusters_out);
  float ans;
  {  // work out initial value of "ans" (objective function improvement).
    Clusterable *all_stats = SumClusterable(*clusters_out);
    ans = SumClusterableObjf(*clusters_out) -
          all_stats->Objf();  // improvement just from the random
    // initialization.
    if (ans < -0.01 &&
        ans < -0.01 * fabs(all_stats->Objf())) {  // something bad happend.
      KHG_WARN
          << "ClusterKMeansOnce: objective function after random assignment "
             "to clusters is worse than in single cluster: "
          << (all_stats->Objf()) << " changed by " << ans
          << ".  Perhaps your stats class has the wrong properties?";
    }
    delete all_stats;
  }
  for (int32_t iter = 0; iter < cfg.num_iters; ++iter) {
    KHG_LOG << "iter: " << iter;
    // Keep refining clusters by reassigning points.
    float objf_before;
    if (cfg.verbose) objf_before = SumClusterableObjf(*clusters_out);

    float impr =
        RefineClusters(points, clusters_out, assignments_out, cfg.refine_cfg);

    float objf_after;
    if (cfg.verbose) objf_after = SumClusterableObjf(*clusters_out);

    ans += impr;
    if (cfg.verbose)
      KHG_LOG << "ClusterKMeansonce: on iteration " << (iter)
              << ", objf before = " << (objf_before) << ", impr = " << (impr)
              << ", objf after = " << (objf_after) << ", normalized by "
              << (normalizer) << " = " << (objf_after / normalizer);
    if (impr == 0) break;
  }
  return ans;
}

float ClusterKMeans(
    const std::vector<Clusterable *> &points, int32_t num_clust,
    std::vector<Clusterable *> *clusters_out,
    std::vector<int32_t> *assignments_out,
    const ClusterKMeansOptions &cfg /*= ClusterKMeansOptions()*/) {
  if (points.size() == 0) {
    // we wouldn't know whether to free the pointers, so it MUST be empty
    // on entry
    if (clusters_out) KHG_ASSERT(clusters_out->empty());

    if (assignments_out) assignments_out->clear();
    return 0.0;
  }
  KHG_ASSERT(cfg.num_tries >= 1 && cfg.num_iters >= 1);

  // we wouldn't know whether to deallocate if it is not empty on entry
  if (clusters_out) KHG_ASSERT(clusters_out->empty());

  if (cfg.num_tries == 1) {
    std::vector<int32_t> assignments;
    return ClusterKMeansOnce(
        points, num_clust, clusters_out,
        (assignments_out != nullptr ? assignments_out : &assignments), cfg);
  } else {  // multiple tries.
    if (clusters_out) {
      KHG_ASSERT(clusters_out->empty());  // we don't know the ownership of
                                          // any pointers in there, otherwise.
    }
    float best_ans = 0.0;
    for (int32_t i = 0; i < cfg.num_tries; i++) {
      std::vector<Clusterable *> clusters_tmp;
      std::vector<int32_t> assignments_tmp;
      float ans = ClusterKMeansOnce(points, num_clust, &clusters_tmp,
                                    &assignments_tmp, cfg);
      KHG_ASSERT(!ContainsNullPointers(clusters_tmp));
      if (i == 0 || ans > best_ans) {
        best_ans = ans;
        if (clusters_out) {
          if (clusters_out->size()) DeletePointers(clusters_out);
          *clusters_out = clusters_tmp;
          clusters_tmp.clear();  // suppress deletion of pointers.
        }
        if (assignments_out) *assignments_out = assignments_tmp;
      }
      // delete anything remaining in clusters_tmp (we cleared it if we used
      // the pointers.
      DeletePointers(&clusters_tmp);
    }
    return best_ans;
  }
}

}  // namespace khg
