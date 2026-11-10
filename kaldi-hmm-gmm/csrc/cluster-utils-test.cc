// kaldi-hmm-gmm/csrc/cluster-utils-test.cc
//
// Copyright 2009-2011  Microsoft Corporation
// Copyright (c)  2026  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/tree/cluster-utils-test.cc (gtest-ified).  It
// covers GaussClusterable/ScalarClusterable basics (kaldi's TestClusterUtils),
// ObjfPlus/ObjfMinus/Distance, SumClusterableObjf/SumClusterableNormalizer/
// SumClusterable, and ClusterKMeans.  kaldi's TestRefineClusters calls
// RefineClusters directly; in this repo RefineClusters is file-local in
// cluster-utils.cc, so its checks are exercised through the public
// ClusterKMeans with a large refine top_n (see the RefineClusters test below).
// kaldi's VectorClusterable tests (TestClusterUtilsVector,
// TestClusterKMeansVector), TestEnsureClusterableVectorNotNull,
// TestAddToClusters, TestAddToClustersOptimized, TestClusterBottomUp,
// TestTreeCluster and TestClusterTopDown are not applicable here: those APIs
// are not part of this repo.

#include "kaldi-hmm-gmm/csrc/cluster-utils.h"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/clusterable-classes.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/stl-utils.h"

namespace khg {

namespace {

// Relative comparison.  kaldi AssertEqual/ApproxEqual is
// |a-b| <= tol * (|a| + |b|) (../kaldi/src/base/kaldi-math.h);
// we use the stricter |a-b| <= 0.5 * tol * (|a| + |b|).
void ExpectEqual(double a, double b, double tol = 1.0e-6) {
  EXPECT_LE(std::fabs(a - b), 0.5 * tol * (std::fabs(a) + std::fabs(b)))
      << "a = " << a << ", b = " << b;
}

// kaldi's TestClusterUtils: very basic tests of the GaussClusterable class.
void TestGaussClusterableBasics() {
  float var_floor = 0.1f;
  int32_t dim = 1 + Rand() % 10;
  int32_t n_gauss = 1 + Rand() % 10;
  std::vector<GaussClusterable *> v(n_gauss);
  for (int32_t i = 0; i < n_gauss; ++i) {
    v[i] = new GaussClusterable(dim, var_floor);
  }
  for (int32_t i = 0; i < n_gauss; ++i) {
    int32_t n_points = 1 + Rand() % 30;
    for (int32_t j = 0; j < n_points; ++j) {
      float post = 0.5f * (Rand() % 3);
      DoubleVector vec(dim);
      for (int32_t k = 0; k < dim; ++k) {
        vec(k) = Randn(0.0f, 1.0f);
      }
      v[i]->AddStats(vec, post);
    }
  }
  for (int32_t i = 0; i + 1 < n_gauss; ++i) {
    float like_before =
        (v[i]->Objf() + v[i + 1]->Objf()) /
        (v[i]->Normalizer() + v[i + 1]->Normalizer());
    Clusterable *tmp = v[i]->Copy();
    tmp->Add(*(v[i + 1]));
    float like_after = tmp->Objf() / tmp->Normalizer();
    if (tmp->Normalizer() > 0.1) {
      // Likelihood should get worse (per frame) after combining stats.
      EXPECT_LE(like_after, like_before);
    }
    delete tmp;
  }
  for (int32_t i = 0; i < n_gauss; ++i) {
    delete v[i];
  }
}

}  // namespace

TEST(ClusterUtils, GaussClusterableBasics) {
  srand(1);
  for (int32_t rep = 0; rep < 2; ++rep) {
    TestGaussClusterableBasics();
  }
}

TEST(ClusterUtils, ObjfPlus) {
  srand(1);
  ScalarClusterable a(1.0f);
  ScalarClusterable b(2.5f);
  ExpectEqual(a.Objf(), 0.0);
  ExpectEqual(b.Objf(), 0.0);
  // 0.5 because half-distance, squared = 1/4, times two points...
  ExpectEqual(a.ObjfPlus(b), -0.5 * (1.0 - 2.5) * (1.0 - 2.5));
  // kaldi also dumps a.Write(std::cerr, ...) here; ScalarClusterable has no
  // Write() in this repo, so that part is skipped.
}

TEST(ClusterUtils, ObjfMinus) {
  srand(1);
  ScalarClusterable a(1.0f);
  ScalarClusterable b(2.5f);
  ExpectEqual(a.Objf(), 0.0);
  ExpectEqual(b.Objf(), 0.0);
  a.Add(b);
  ExpectEqual(a.ObjfMinus(b), 0.0);
  a.Add(b);
  ExpectEqual(a.ObjfMinus(b), -0.5 * (1.0 - 2.5) * (1.0 - 2.5));
}

TEST(ClusterUtils, Distance) {
  srand(1);
  ScalarClusterable a(1.0f);
  ScalarClusterable b(2.5f);
  ExpectEqual(a.Objf(), 0.0);
  ExpectEqual(b.Objf(), 0.0);
  // Since distance is negated objf-change, and original objfs were zero.
  ExpectEqual(a.ObjfPlus(b), -a.Distance(b));
}

TEST(ClusterUtils, SumObjfAndSumNormalizer) {
  srand(1);
  ScalarClusterable a(1.0f);
  ScalarClusterable b(2.5f);
  ExpectEqual(a.Objf(), 0.0);
  ExpectEqual(b.Objf(), 0.0);
  a.Add(b);
  std::vector<Clusterable *> vec;
  vec.push_back(&a);
  vec.push_back(&a);
  ExpectEqual(SumClusterableObjf(vec), 2 * vec[0]->Objf());
  ExpectEqual(SumClusterableNormalizer(vec), 2 * vec[0]->Normalizer());
}

TEST(ClusterUtils, Sum) {
  srand(1);
  ScalarClusterable a(1.0f);
  ScalarClusterable b(2.5f);
  std::vector<Clusterable *> vec;
  vec.push_back(&a);
  vec.push_back(&b);
  Clusterable *sum = SumClusterable(vec);
  ExpectEqual(a.ObjfPlus(b), sum->Objf());
  delete sum;
}

TEST(ClusterUtils, RefineClusters) {
  srand(1);
  // kaldi's TestRefineClusters verifies that RefineClusters does not make the
  // clustering worse and, with optimal parameters, makes it exact.  Here
  // RefineClusters is file-local in cluster-utils.cc, so we run the same
  // refinement (huge num_iters, top_n >= number of clusters) through the
  // public ClusterKMeans and check the analogous properties of its result and
  // its returned objective-function improvement.
  for (int32_t n = 0; n < 4; ++n) {
    int32_t n_clust = Rand() % 10;
    std::vector<Clusterable *> points;
    for (int32_t j = 0; j < n_clust; ++j) {
      int32_t n_points = 1 + Rand() % 5;
      float clust_center = static_cast<float>(j);
      for (int32_t k = 0; k < n_points; ++k) {
        points.push_back(
            new ScalarClusterable(clust_center + RandUniform() * 0.01f));
      }
    }
    if (points.empty()) continue;  // n_clust == 0: nothing to cluster.

    float points_objf = SumClusterableObjf(points);
    Clusterable *one_cluster = SumClusterable(points);
    float one_cluster_objf = one_cluster->Objf();
    delete one_cluster;
    // Summing the individual point objfs can never be worse than merging
    // everything into one cluster.
    EXPECT_GE(points_objf,
              one_cluster_objf -
                  (std::fabs(points_objf) + std::fabs(one_cluster_objf)) *
                      0.001);

    std::vector<Clusterable *> clusters;
    std::vector<int32_t> assignments;
    ClusterKMeansOptions kcfg;
    kcfg.refine_cfg.num_iters = 10000;  // very large.
    kcfg.refine_cfg.top_n = 2 + (Rand() % 20);
    kcfg.num_tries = 1;  // a single run keeps the check deterministic.
    kcfg.verbose = false;

    float impr = ClusterKMeans(points, n_clust, &clusters, &assignments, kcfg);

    float clust_objf_after = SumClusterableObjf(clusters);
    EXPECT_EQ(assignments.size(), points.size());
    EXPECT_GE(impr, -1e-4);  // never worse than everything in one cluster.
    // The returned improvement must match the change of the objective
    // function versus everything being in the same cluster.
    ExpectEqual(clust_objf_after - one_cluster_objf, impr, 1e-3);
    if (kcfg.refine_cfg.top_n >= n_clust) {  // check exact (kaldi's assertion).
      // Note: this is kaldi's literal assertion ("check exact"); it holds
      // trivially since the objf is always <= 0.  The refinement is a
      // k-means-like single-point-move method and can stop at local optima
      // with significantly negative objf, so no stronger exactness claim is
      // made here (kaldi's own test exhibits residual objfs around -1).
      EXPECT_LE(clust_objf_after, 0.01 * points.size());
    }
    DeletePointers(&clusters);
    DeletePointers(&points);
  }
}

TEST(ClusterUtils, ClusterKMeans) {
  srand(1);
  int32_t n_points_tot = 0;
  int32_t n_wrong_tot = 0;
  for (int32_t n = 0; n < 3; ++n) {
    int32_t n_clust = Rand() % 10;
    std::vector<Clusterable *> points;
    std::vector<int32_t> assignments_ref;
    for (int32_t j = 0; j < n_clust; ++j) {
      int32_t n_points = 1 + Rand() % 5;
      float clust_center = static_cast<float>(j);
      for (int32_t k = 0; k < n_points; ++k) {
        points.push_back(
            new ScalarClusterable(clust_center + RandUniform() * 0.01f));
        assignments_ref.push_back(j);
      }
    }
    std::vector<Clusterable *> clusters;
    std::vector<int32_t> assignments;
    ClusterKMeansOptions kcfg;
    kcfg.verbose = false;

    float ans = ClusterKMeans(points, n_clust, &clusters, &assignments, kcfg);

    if (n < 3) {
      // Make sure it does not crash with NULL outputs.
      ClusterKMeans(points, n_clust, nullptr, nullptr, kcfg);
    }

    float clust_objf = SumClusterableObjf(clusters);
    KHG_LOG << "TestClusterKMeans: objf after clustering is: " << clust_objf
            << ", impr is: " << ans;
    if (clust_objf < -1.0 * points.size()) {
      // A bit high... kaldi only logs a warning here.
      KHG_WARN << "ClusterKMeans did not work quite as well as expected";
    }
    EXPECT_EQ(assignments.size(), points.size());

    int32_t num_wrong = 0;
    for (int32_t i = 0; i < static_cast<int32_t>(points.size()); ++i) {
      int32_t j = Rand() % points.size();
      if (assignments_ref[i] == assignments_ref[j]) {
        if (assignments[i] != assignments[j]) num_wrong++;
      } else {
        if (assignments[i] == assignments[j]) num_wrong++;
      }
    }
    n_points_tot += points.size();
    n_wrong_tot += num_wrong;

    DeletePointers(&clusters);
    DeletePointers(&points);
  }
  // Got too many wrong in k-means test [may not be fatal, but check it out].
  EXPECT_LE(n_wrong_tot * 4, n_points_tot);
}

}  // namespace khg
