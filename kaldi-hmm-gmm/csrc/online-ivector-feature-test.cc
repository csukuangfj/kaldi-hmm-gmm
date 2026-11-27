// kaldi-hmm-gmm/csrc/online-ivector-feature-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Unit tests for OnlineIvectorFeature and OnlineSilenceWeighting
// (online-ivector-feature.{h,cc}).  Parity with kaldi's
// ivector-extract-online2 and with the vosk runtime lives in
// tests/parity/parity_online_ivector_feature.py.

#include "kaldi-hmm-gmm/csrc/online-ivector-feature.h"

#include <cmath>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/online-ivector-extractor.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "kaldi_native_io/csrc/io-funcs.h"
#include "kaldi_native_io/csrc/kaldi-matrix.h"
#include "kaldi_native_io/csrc/kaldi-vector.h"

namespace khg {

namespace {

// Deterministic pseudo-random values in a realistic range.
double Frand(int32_t *seed) {
  *seed = *seed * 1103515245 + 12345;
  return static_cast<double>((*seed >> 8) % 2000 - 1000) / 100.0;
}

// A tiny extractor (D=3, S=2, I=4), written in kaldi's final.ie binary
// format and read back through the real reader.
IvectorExtractor MakeExtractor() {
  const int32_t D = 3;
  const int32_t S = 2;
  const int32_t I = 4;
  std::ostringstream os(std::ios::binary);
  kaldiio::WriteToken(os, true, "<IvectorExtractor>");
  kaldiio::WriteToken(os, true, "<w>");
  kaldiio::Matrix<double> w(0, 0);
  w.Write(os, true);
  kaldiio::WriteToken(os, true, "<w_vec>");
  kaldiio::Vector<double> w_vec(I);
  for (int32_t i = 0; i < I; i++) w_vec(i) = 1.0 / I;
  w_vec.Write(os, true);
  kaldiio::WriteToken(os, true, "<M>");
  kaldiio::WriteBasicType(os, true, I);
  int32_t seed = 7;
  for (int32_t i = 0; i < I; i++) {
    kaldiio::Matrix<double> m(D, S);
    for (int32_t d = 0; d < D; d++) {
      for (int32_t s = 0; s < S; s++) m(d, s) = Frand(&seed) / 10.0;
    }
    m.Write(os, true);
  }
  kaldiio::WriteToken(os, true, "<SigmaInv>");
  for (int32_t i = 0; i < I; i++) {
    kaldiio::WriteToken(os, true, "DP");
    kaldiio::WriteBasicType(os, true, D);
    for (int32_t r = 0; r < D; r++) {
      for (int32_t c = 0; c <= r; c++) {
        double v = (r == c ? 1.0 + std::fabs(Frand(&seed)) / 5.0 : 0.05);
        os.write(reinterpret_cast<const char *>(&v), sizeof(double));
      }
    }
  }
  kaldiio::WriteToken(os, true, "<IvectorOffset>");
  kaldiio::WriteBasicType(os, true, 1.5);
  kaldiio::WriteToken(os, true, "</IvectorExtractor>");

  std::istringstream is(os.str(), std::ios::binary);
  IvectorExtractor extractor;
  extractor.Read(is, true);
  return extractor;
}

struct IvectorSetup {
  IvectorExtractor extractor;
  std::unique_ptr<DiagGmm> ubm;
  OnlineIvectorExtractorConfig config;
  FloatMatrix feats;
};

// Raw dim 3 (= extractor dim), splice L=R=1, lda = center frame plus a
// small mix of the neighbours.
IvectorSetup MakeSetup(int32_t num_frames) {
  IvectorSetup s;
  s.extractor = MakeExtractor();
  const int32_t D = s.extractor.FeatDim();
  int32_t seed = 11;
  FloatMatrix lda = FloatMatrix::Zero(D, D * 3);
  for (int32_t d = 0; d < D; d++) {
    lda(d, D + d) = 1.0f;
    lda(d, d) = 0.1f;
    lda(d, 2 * D + d) = -0.1f;
  }
  DoubleMatrix glob = DoubleMatrix::Zero(2, D + 1);
  for (int32_t d = 0; d < D; d++) glob(0, d) = 30.0 * d;
  glob(0, D) = 100.0;

  const int32_t I = s.extractor.NumGauss();
  FloatVector weights(I);
  FloatMatrix inv_vars(I, D);
  FloatMatrix means_invvars(I, D);
  for (int32_t i = 0; i < I; i++) {
    weights(i) = 1.0f / I;
    for (int32_t d = 0; d < D; d++) {
      inv_vars(i, d) = 0.1f;
      means_invvars(i, d) = 0.1f * Frand(&seed);
    }
  }
  s.ubm.reset(new DiagGmm(weights, inv_vars, means_invvars));

  s.config.left_context = 1;
  s.config.right_context = 1;
  s.config.lda_mat = lda;
  s.config.global_cmvn_stats = glob;
  s.config.max_count = 20.0;
  s.config.num_gselect = 2;  // prune, so weights interact with gselect

  s.feats = FloatMatrix(num_frames, D);
  for (int32_t t = 0; t < num_frames; t++) {
    for (int32_t d = 0; d < D; d++) s.feats(t, d) = Frand(&seed);
  }
  return s;
}

void ExpectNear(const FloatVector &a, const FloatVector &b, float tol) {
  ASSERT_EQ(a.size(), b.size());
  for (int32_t i = 0; i < a.size(); i++) {
    EXPECT_NEAR(a(i), b(i), tol) << "dim " << i;
  }
}

}  // namespace

TEST(OnlineIvectorFeature, NumFramesReady) {
  IvectorSetup s = MakeSetup(10);
  OnlineIvectorFeature f(s.config, s.extractor, *s.ubm);
  EXPECT_EQ(f.NumFramesReady(), 0);
  f.AcceptFeatures(s.feats.topRows(1));
  EXPECT_EQ(f.NumFramesReady(), 0);  // right context 1
  f.AcceptFeatures(s.feats.middleRows(1, 4));
  EXPECT_EQ(f.NumFramesReady(), 4);
  EXPECT_FALSE(f.IsLastFrame(4));
  f.InputFinished();
  EXPECT_EQ(f.NumFramesReady(), 5);
  EXPECT_TRUE(f.IsLastFrame(4));
  EXPECT_EQ(f.Dim(), s.extractor.IvectorDim());
}

// Without weights and with use_most_recent_ivector == false, the i-vector
// history must equal the (eager) OnlineIvectorExtractor's: both are
// kaldi's period-based estimates.
TEST(OnlineIvectorFeature, PeriodModeMatchesOnlineIvectorExtractor) {
  IvectorSetup s = MakeSetup(57);
  s.config.use_most_recent_ivector = false;
  s.config.ivector_period = 10;

  OnlineIvectorExtractor ref(s.config, s.extractor, *s.ubm);
  ref.AcceptFeatures(s.feats);
  ref.Flush();

  OnlineIvectorFeature f(s.config, s.extractor, *s.ubm);
  f.AcceptFeatures(s.feats);
  f.InputFinished();
  ASSERT_EQ(f.NumFramesReady(), 57);
  ASSERT_EQ(ref.NumIvectorsReady(), 6);
  for (int32_t t = 0; t < 57; t++) {
    FloatVector iv;
    f.GetFrame(t, &iv);
    FloatVector expected = ref.GetIvector(t / 10).cast<float>();
    ExpectNear(iv, expected, 1e-4f);
  }
}

// use_most_recent_ivector: GetFrame(t) is the estimate from frames [0, t].
TEST(OnlineIvectorFeature, MostRecentIsEstimateAtFrame) {
  IvectorSetup s = MakeSetup(40);
  for (int32_t t : {0, 7, 23, 39}) {
    OnlineIvectorFeature f(s.config, s.extractor, *s.ubm);
    f.AcceptFeatures(s.feats);
    f.InputFinished();
    FloatVector iv;
    f.GetFrame(t, &iv);

    // Reference: the eager extractor with period t estimates at frames 0
    // and t (index 1).  (The i-vector dim is 2, so 15 CG iterations
    // converge regardless of the warm start.)
    OnlineIvectorExtractorConfig cfg = s.config;
    cfg.ivector_period = (t == 0 ? 1 : t);
    OnlineIvectorExtractor ref(cfg, s.extractor, *s.ubm);
    ref.AcceptFeatures(s.feats);
    ref.Flush();
    FloatVector expected = ref.GetIvector(t == 0 ? 0 : 1).cast<float>();
    ExpectNear(iv, expected, 1e-4f);
  }
}

// Weights of 1 for every frame equal the unweighted estimate.
TEST(OnlineIvectorFeature, UnitWeightsEqualUnweighted) {
  IvectorSetup s = MakeSetup(30);
  OnlineIvectorFeature plain(s.config, s.extractor, *s.ubm);
  plain.AcceptFeatures(s.feats);
  plain.InputFinished();

  OnlineIvectorFeature weighted(s.config, s.extractor, *s.ubm);
  weighted.AcceptFeatures(s.feats);
  weighted.InputFinished();
  std::vector<std::pair<int32_t, float>> dw;
  for (int32_t t = 0; t < 30; t++) dw.emplace_back(t, 1.0f);
  weighted.UpdateFrameWeights(dw);

  for (int32_t t : {5, 17, 29}) {
    FloatVector a;
    FloatVector b;
    plain.GetFrame(t, &a);
    weighted.GetFrame(t, &b);
    ExpectNear(a, b, 1e-5f);
  }
  EXPECT_NEAR(plain.NumFrames(), weighted.NumFrames(), 1e-3f);
}

// A weight that is changed after the frame was used must end up as if the
// final weight had been given from the start (kaldi's delta-weight
// correction).  Weight 0.5 is used, then corrected by +0.5 to 1.
TEST(OnlineIvectorFeature, DeltaWeightCorrection) {
  IvectorSetup s = MakeSetup(30);
  // gselect pruning depends on the weight (GetMinPost); disable pruning so
  // the correction is exact up to rounding.
  s.config.num_gselect = 0;
  s.config.min_post = 0.0;

  OnlineIvectorFeature direct(s.config, s.extractor, *s.ubm);
  direct.AcceptFeatures(s.feats);
  direct.InputFinished();
  std::vector<std::pair<int32_t, float>> dw;
  for (int32_t t = 0; t < 30; t++) dw.emplace_back(t, 1.0f);
  direct.UpdateFrameWeights(dw);
  FloatVector expected;
  direct.GetFrame(29, &expected);

  OnlineIvectorFeature corrected(s.config, s.extractor, *s.ubm);
  corrected.AcceptFeatures(s.feats);
  corrected.InputFinished();
  dw.clear();
  for (int32_t t = 0; t < 30; t++) dw.emplace_back(t, 0.5f);
  corrected.UpdateFrameWeights(dw);
  FloatVector tmp;
  corrected.GetFrame(20, &tmp);  // frames 0..20 used with weight 0.5
  dw.clear();
  for (int32_t t = 0; t < 30; t++) dw.emplace_back(t, 0.5f);
  corrected.UpdateFrameWeights(dw);
  FloatVector iv;
  corrected.GetFrame(29, &iv);
  ExpectNear(iv, expected, 1e-4f);
  EXPECT_NEAR(corrected.NumFrames(), direct.NumFrames(), 1e-3f);
}

// Silence weights down-weight frames: weight 0 for every frame keeps the
// prior (zero after removing the offset).
TEST(OnlineIvectorFeature, ZeroWeightsKeepPrior) {
  IvectorSetup s = MakeSetup(20);
  OnlineIvectorFeature f(s.config, s.extractor, *s.ubm);
  f.AcceptFeatures(s.feats);
  f.InputFinished();
  std::vector<std::pair<int32_t, float>> dw;
  for (int32_t t = 0; t < 20; t++) dw.emplace_back(t, 0.0f);
  f.UpdateFrameWeights(dw);
  FloatVector iv;
  f.GetFrame(19, &iv);
  for (int32_t i = 0; i < iv.size(); i++) EXPECT_NEAR(iv(i), 0.0f, 1e-6f);
  EXPECT_EQ(f.NumFrames(), 0.0f);
}

namespace {

// Two phones; phone 1 is silence.  3-state left-to-right topology.
struct SilenceSetup {
  HmmTopology topo;
  std::unique_ptr<ContextDependency> ctx_dep;
  std::unique_ptr<TransitionModel> tm;
  int32_t sil_tid = 0;     // a transition-id of phone 1
  int32_t speech_tid = 0;  // a transition-id of phone 2
};

SilenceSetup MakeSilenceSetup() {
  SilenceSetup s;
  std::istringstream iss(
      "<Topology>\n<TopologyEntry>\n<ForPhones> 1 2 </ForPhones>\n"
      "<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 "
      "</State>\n<State> 1 </State>\n</TopologyEntry>\n</Topology>\n");
  s.topo.Read(iss, false);
  std::vector<int32_t> phone2num_pdf_classes;
  s.topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);
  s.ctx_dep.reset(
      MonophoneContextDependency(s.topo.GetPhones(), phone2num_pdf_classes));
  s.tm.reset(new TransitionModel(*s.ctx_dep, s.topo));
  for (int32_t tid = 1; tid <= s.tm->NumTransitionIds(); tid++) {
    if (s.tm->TransitionIdToPhone(tid) == 1 && s.sil_tid == 0) s.sil_tid = tid;
    if (s.tm->TransitionIdToPhone(tid) == 2 && s.speech_tid == 0)
      s.speech_tid = tid;
  }
  return s;
}

}  // namespace

TEST(OnlineSilenceWeighting, Inactive) {
  SilenceSetup s = MakeSilenceSetup();
  OnlineSilenceWeightingConfig config;
  EXPECT_FALSE(config.Active());
  config.silence_phones_str = "1";
  EXPECT_FALSE(config.Active());  // weight 1.0
  config.silence_weight = 0.001f;
  EXPECT_TRUE(config.Active());
  OnlineSilenceWeighting sw(*s.tm, config, 3);
  EXPECT_TRUE(sw.Active());
}

TEST(OnlineSilenceWeighting, DeltaWeights) {
  SilenceSetup s = MakeSilenceSetup();
  ASSERT_GT(s.sil_tid, 0);
  ASSERT_GT(s.speech_tid, 0);
  OnlineSilenceWeightingConfig config;
  config.silence_phones_str = "1";
  config.silence_weight = 0.001f;
  OnlineSilenceWeighting sw(*s.tm, config, 3);

  // No traceback yet: everything gets the silence weight.  6 input frames
  // -> 2 decoder frames -> input frames 0..5.
  std::vector<std::pair<int32_t, float>> dw;
  sw.ComputeCurrentTraceback({});
  sw.GetDeltaWeights(6, 0, &dw);
  ASSERT_EQ(dw.size(), 6u);
  for (int32_t i = 0; i < 6; i++) {
    EXPECT_EQ(dw[i].first, i);
    EXPECT_FLOAT_EQ(dw[i].second, 0.001f);
  }

  // Traceback: decoder frame 0 silence, frame 1 speech.  Frame 0 keeps its
  // weight (only the last frame is output with a zero delta); frame 1
  // changes 0.001 -> 1; new decoder frame 2 has no traceback and copies
  // frame 1 (speech).
  sw.ComputeCurrentTraceback({s.sil_tid, s.speech_tid});
  sw.GetDeltaWeights(9, 0, &dw);
  std::vector<std::pair<int32_t, float>> expected = {
      {3, 0.999f}, {4, 0.999f}, {5, 0.999f},
      {6, 1.0f},   {7, 1.0f},   {8, 1.0f}};
  ASSERT_EQ(dw.size(), expected.size());
  for (size_t i = 0; i < dw.size(); i++) {
    EXPECT_EQ(dw[i].first, expected[i].first);
    EXPECT_NEAR(dw[i].second, expected[i].second, 1e-6f);
  }

  // Revised traceback: decoder frame 1 is silence after all.
  sw.ComputeCurrentTraceback({s.sil_tid, s.sil_tid, s.speech_tid});
  sw.GetDeltaWeights(9, 0, &dw);
  expected = {{3, -0.999f}, {4, -0.999f}, {5, -0.999f},
              {6, 0.0f},    {7, 0.0f},    {8, 0.0f}};
  ASSERT_EQ(dw.size(), expected.size());
  for (size_t i = 0; i < dw.size(); i++) {
    EXPECT_EQ(dw[i].first, expected[i].first);
    EXPECT_NEAR(dw[i].second, expected[i].second, 1e-6f);
  }

  std::vector<int32_t> nonsil;
  sw.GetNonsilenceFrames(9, 0, &nonsil);
  EXPECT_EQ(nonsil, std::vector<int32_t>({2}));
}

TEST(OnlineSilenceWeighting, BadSilencePhones) {
  SilenceSetup s = MakeSilenceSetup();
  OnlineSilenceWeightingConfig config;
  config.silence_phones_str = "1::2";
  config.silence_weight = 0.001f;
  EXPECT_ANY_THROW(OnlineSilenceWeighting(*s.tm, config, 1));
}

}  // namespace khg
