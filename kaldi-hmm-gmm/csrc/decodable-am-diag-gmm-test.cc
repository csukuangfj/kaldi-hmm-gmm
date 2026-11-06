// kaldi-hmm-gmm/csrc/decodable-am-diag-gmm-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Regression/smoke tests for DecodableAmDiagGmmUnmapped (1-based indices,
// likelihood cache, log_sum_exp_prune passthrough) and DecodableAmDiagGmmScaled.

#include "kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h"

#include <cmath>
#include <sstream>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/kaldi-math.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"

namespace khg {

namespace {

// Relative comparison.  kaldi AssertEqual/ApproxEqual is
// |a-b| <= tol * (|a| + |b|) (../kaldi/src/base/kaldi-math.h);
// we use the stricter |a-b| <= 0.5 * tol * (|a| + |b|).
void ExpectEqual(double a, double b, double tol = 1.0e-6) {
  EXPECT_LE(std::fabs(a - b), 0.5 * tol * (std::fabs(a) + std::fabs(b)))
      << "a = " << a << ", b = " << b;
}

// Random well-conditioned diagonal GMM (kaldi unittest::InitRandDiagGmm).
void InitRandDiagGmm(int32_t dim, int32_t num_comp, DiagGmm *gmm) {
  gmm->Resize(num_comp, dim);
  FloatVector weights(num_comp);
  FloatMatrix means(num_comp, dim);
  FloatMatrix invvars(num_comp, dim);
  for (int32_t m = 0; m != num_comp; ++m) {
    weights(m) = std::exp(Randn(0.0f, 1.0f));
    for (int32_t d = 0; d != dim; ++d) {
      means(m, d) = Randn(0.0f, 1.0f) * 10.0f;
      invvars(m, d) = std::exp(-Randn(0.0f, 1.0f)) * 0.1f;
    }
  }
  weights = (weights / weights.sum()).eval();
  gmm->SetWeights(weights);
  gmm->SetInvVarsAndMeans(invvars, means);
  gmm->ComputeGconsts();
}

// Reference implementation of kaldi's VectorBase::LogSumExp(prune)
// (../kaldi/src/matrix/kaldi-vector.cc): terms below max + kMinLogDiffFloat
// are dropped; when prune > 0, terms below max - prune are dropped instead if
// that is stricter.  The max term is always kept.  Result:
// max + log(sum(exp(f - max))) over the kept terms.
float RefLogSumExp(const std::vector<float> &terms, float prune) {
  float max_v = terms[0];
  for (size_t i = 1; i != terms.size(); ++i) {
    if (terms[i] > max_v) max_v = terms[i];
  }
  float cutoff = max_v + kMinLogDiffFloat;
  if (prune > 0.0f && max_v - prune > cutoff) {
    cutoff = max_v - prune;
  }
  double sum_relto_max = 0.0;
  for (size_t i = 0; i != terms.size(); ++i) {
    if (terms[i] >= cutoff) {
      sum_relto_max += std::exp(terms[i] - max_v);
    }
  }
  return max_v + std::log(sum_relto_max);
}

std::vector<float> ComponentLogLikes(const DiagGmm &gmm, const FloatVector &feat) {
  FloatVector loglikes;
  gmm.LogLikelihoods(feat, &loglikes);
  return std::vector<float>(loglikes.data(), loglikes.data() + loglikes.size());
}

// Two 3-emitting-state phones (1 and 2) as in hmm-utils-test.cc.
static const char kTopoStr[] = R"(
<Topology>
<TopologyEntry>
<ForPhones> 1 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones> 2 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
</Topology>
)";

static void BuildTransitionModel(HmmTopology *topo, ContextDependency **dep,
                                 TransitionModel **trans_model) {
  std::istringstream is(kTopoStr);
  bool binary = false;
  topo->Read(is, binary);

  std::vector<int32_t> phone2num_pdf_classes;
  topo->GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

  *dep = MonophoneContextDependency(topo->GetPhones(), phone2num_pdf_classes);
  *trans_model = new TransitionModel(**dep, *topo);
}

}  // namespace

// Basic behavior of DecodableAmDiagGmmUnmapped: 1-based state indices, the
// frame bookkeeping, the per-frame/per-pdf likelihoods, and the cache.
TEST(DecodableAmDiagGmm, UnmappedBasics) {
  srand(1);
  const int32_t dim = 4;
  const int32_t num_frames = 5;
  const int32_t num_pdfs = 2;

  AmDiagGmm am;
  for (int32_t i = 0; i != num_pdfs; ++i) {
    DiagGmm gmm;
    InitRandDiagGmm(dim, 3, &gmm);
    am.AddPdf(gmm);
  }

  FloatMatrix feats(num_frames, dim);
  for (int32_t r = 0; r != num_frames; ++r) {
    for (int32_t c = 0; c != dim; ++c) {
      feats(r, c) = Randn(0.0f, 1.0f);
    }
  }

  DecodableAmDiagGmmUnmapped dec(am, feats);  // default: no pruning

  EXPECT_EQ(dec.NumFramesReady(), num_frames);
  EXPECT_EQ(dec.NumIndices(), num_pdfs);
  EXPECT_FALSE(dec.IsLastFrame(-1));
  EXPECT_FALSE(dec.IsLastFrame(num_frames - 2));
  EXPECT_TRUE(dec.IsLastFrame(num_frames - 1));

  for (int32_t f = 0; f != num_frames; ++f) {
    FloatVector feat = feats.row(f).transpose();
    for (int32_t pdf = 0; pdf != num_pdfs; ++pdf) {
      std::vector<float> terms = ComponentLogLikes(am.GetPdf(pdf), feat);
      float expected = RefLogSumExp(terms, -1.0f);

      // state_index is 1-based (pdf-id plus one, OpenFst compatibility).
      float actual = dec.LogLikelihood(f, pdf + 1);
      ExpectEqual(actual, expected, 1.0e-5);

      // A repeated query hits the cache and returns the same value.
      EXPECT_FLOAT_EQ(dec.LogLikelihood(f, pdf + 1), actual);

      // It agrees with the model's own scoring (LogSumExp without pruning).
      EXPECT_FLOAT_EQ(am.LogLikelihood(pdf, feat), actual);
    }
  }
}

// Regression test: log_sum_exp_prune must be passed to LogSumExp().  The
// GMM below is engineered so that at every frame the per-component
// log-likes differ by 0, -0.3, -1.5, -2.5 from the best component.
TEST(DecodableAmDiagGmm, UnmappedLogSumExpPrune) {
  srand(1);
  const int32_t num_comp = 4;

  // Per-component log-weight offsets; since means are 0 and variances 1,
  // the per-component log-likes at any feature x differ by exactly these
  // offsets (the x-dependent terms are identical across components).
  std::vector<float> diffs = {0.0f, -0.3f, -1.5f, -2.5f};

  DiagGmm gmm;
  gmm.Resize(num_comp, 1);
  FloatVector weights(num_comp);
  float z = 0.0f;
  for (int32_t i = 0; i != num_comp; ++i) {
    weights(i) = std::exp(diffs[i]);
    z += weights(i);
  }
  weights = (weights / z).eval();
  FloatMatrix means = FloatMatrix::Zero(num_comp, 1);
  FloatMatrix invvars = FloatMatrix::Ones(num_comp, 1);
  gmm.SetWeights(weights);
  gmm.SetInvVarsAndMeans(invvars, means);
  gmm.ComputeGconsts();

  AmDiagGmm am;
  am.AddPdf(gmm);

  FloatMatrix feats(2, 1);
  feats(0, 0) = 0.0f;
  feats(1, 0) = 0.3f;

  // Sanity: the constructed spread is really {0, -0.3, -1.5, -2.5}, so the
  // pruning cutoffs below are discriminating.
  for (int32_t f = 0; f != 2; ++f) {
    FloatVector feat = feats.row(f).transpose();
    std::vector<float> terms = ComponentLogLikes(am.GetPdf(0), feat);
    for (int32_t i = 1; i != num_comp; ++i) {
      ExpectEqual(terms[i] - terms[0], diffs[i], 1.0e-4);
    }
  }

  std::vector<float> prune_values = {-1.0f, 100.0f, 2.0f, 0.5f};
  for (float prune : prune_values) {
    DecodableAmDiagGmmUnmapped dec(am, feats, prune);
    for (int32_t f = 0; f != 2; ++f) {
      FloatVector feat = feats.row(f).transpose();
      std::vector<float> terms = ComponentLogLikes(am.GetPdf(0), feat);
      float expected = RefLogSumExp(terms, prune);
      ExpectEqual(dec.LogLikelihood(f, 1), expected, 1.0e-5);
    }
  }

  // Discriminating check: with this spread, pruning at 2 must change the
  // answer (it drops the -2.5 term).  If LogLikelihood() ignored its
  // log_sum_exp_prune argument these two would be equal.
  DecodableAmDiagGmmUnmapped dec_unpruned(am, feats, -1.0f);
  DecodableAmDiagGmmUnmapped dec_pruned(am, feats, 2.0f);
  float ll_unpruned = dec_unpruned.LogLikelihood(0, 1);
  float ll_pruned = dec_pruned.LogLikelihood(0, 1);
  EXPECT_GT(std::fabs(ll_unpruned - ll_pruned), 0.01);

  // A prune that is not stricter than the float-epsilon cutoff changes
  // nothing: max - 100 is below max + kMinLogDiffFloat.
  DecodableAmDiagGmmUnmapped dec_loose(am, feats, 100.0f);
  EXPECT_FLOAT_EQ(dec_loose.LogLikelihood(0, 1), ll_unpruned);
}

// Smoke test for DecodableAmDiagGmmScaled: transition-id to pdf mapping and
// the acoustic scale.
TEST(DecodableAmDiagGmm, ScaledSmoke) {
  srand(1);
  HmmTopology topo;
  ContextDependency *dep = nullptr;
  TransitionModel *tm = nullptr;
  BuildTransitionModel(&topo, &dep, &tm);

  AmDiagGmm am;
  for (int32_t i = 0; i != tm->NumPdfs(); ++i) {
    DiagGmm gmm;
    InitRandDiagGmm(4, 2, &gmm);
    am.AddPdf(gmm);
  }

  const int32_t num_frames = 3;
  FloatMatrix feats(num_frames, 4);
  for (int32_t r = 0; r != num_frames; ++r) {
    for (int32_t c = 0; c != 4; ++c) {
      feats(r, c) = Randn(0.0f, 1.0f);
    }
  }

  const float scale = 0.8f;
  DecodableAmDiagGmmUnmapped unmapped(am, feats);
  DecodableAmDiagGmmScaled scaled(am, *tm, feats, scale);

  EXPECT_EQ(scaled.NumIndices(), tm->NumTransitionIds());
  for (int32_t tid = 1; tid <= tm->NumTransitionIds(); tid += 3) {
    int32_t pdf = tm->TransitionIdToPdf(tid);
    for (int32_t f = 0; f != num_frames; ++f) {
      float expected = scale * unmapped.LogLikelihood(f, pdf + 1);
      ExpectEqual(scaled.LogLikelihood(f, tid), expected);
    }
  }

  delete tm;
  delete dep;
}

}  // namespace khg
