// kaldi-hmm-gmm/csrc/lattice-simple-decoder-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Regression/smoke tests for LatticeSimpleDecoder (kaldi has no decoder unit
// tests).  In particular they guard that ProcessNonemitting() treats an empty
// token queue (no active state has input-epsilon arcs) as a benign no-op.

#include "kaldi-hmm-gmm/csrc/lattice-simple-decoder.h"

#include <utility>
#include <vector>

#include "fst/fstlib.h"
#include "gtest/gtest.h"
#include "kaldi-hmm-gmm/csrc/decodable-itf.h"
#include "kaldifst/csrc/lattice-weight.h"

namespace khg {

namespace {

// Deterministic DecodableInterface for the decoder tests: frame t gives
// log-likelihood 1.0 to label targets[t] and -1.0 to every other label.
class TestDecodable : public DecodableInterface {
 public:
  explicit TestDecodable(std::vector<int32_t> targets)
      : targets_(std::move(targets)) {}

  float LogLikelihood(int32_t frame, int32_t index) override {
    KHG_ASSERT(frame >= 0 && frame < static_cast<int32_t>(targets_.size()));
    return (index == targets_[frame]) ? 1.0f : -1.0f;
  }

  bool IsLastFrame(int32_t frame) const override {
    return frame == static_cast<int32_t>(targets_.size()) - 1;
  }

  int32_t NumFramesReady() const override {
    return static_cast<int32_t>(targets_.size());
  }

  // Labels used by the test graphs are 1 and 2.
  int32_t NumIndices() const override { return 3; }

 private:
  std::vector<int32_t> targets_;
};

// state 0 -ilabel 1-> state 1 -ilabel 2-> state 2 (final): a linear acceptor
// where NO state has input-epsilon arcs.
fst::VectorFst<fst::StdArc> MakeLinearChainFst() {
  fst::VectorFst<fst::StdArc> fst;
  fst.AddState();  // state 0
  fst.AddState();  // state 1
  fst.AddState();  // state 2
  fst.SetStart(0);
  fst.SetFinal(2, fst::StdArc::Weight::One());
  fst.AddArc(0, fst::StdArc(1, 1, fst::StdArc::Weight::One(), 1));
  fst.AddArc(1, fst::StdArc(2, 2, fst::StdArc::Weight::One(), 2));
  return fst;
}

// Walks the (linear) best-path FST from its start state and returns the
// (ilabel, olabel) sequence; also sums Value1 (graph cost) + Value2
// (acoustic cost) over the arcs into *total_cost.
std::vector<std::pair<int32_t, int32_t>> GetLinearPath(const fst::Lattice &lat,
                                                       float *total_cost) {
  std::vector<std::pair<int32_t, int32_t>> path;
  *total_cost = 0.0f;
  fst::Lattice::StateId s = lat.Start();
  EXPECT_NE(s, fst::kNoStateId);
  while (lat.Final(s) == fst::LatticeWeight::Zero()) {
    EXPECT_EQ(lat.NumArcs(s), 1);
    if (lat.NumArcs(s) != 1) break;
    fst::ArcIterator<fst::Fst<fst::LatticeArc>> aiter(lat, s);
    const fst::LatticeArc &arc = aiter.Value();
    path.emplace_back(arc.ilabel, arc.olabel);
    *total_cost += arc.weight.Value1() + arc.weight.Value2();
    s = arc.nextstate;
    EXPECT_LT(static_cast<int32_t>(path.size()), 100);  // cycle guard
  }
  return path;
}

}  // namespace

// Regression test: graphs whose active states have NO input-epsilon arcs make
// the ProcessNonemitting() queue empty, which is a benign no-op (kaldi has no
// check here); the old implementation threw "no surviving tokens".
TEST(LatticeSimpleDecoder, DecodeWithoutInputEpsilons) {
  srand(1);
  fst::VectorFst<fst::StdArc> fst = MakeLinearChainFst();

  LatticeSimpleDecoderConfig config(16.0f, 10.0f, 25);
  LatticeSimpleDecoder decoder(fst, config);

  TestDecodable decodable(std::vector<int32_t>{1, 2});
  bool ok = false;
  EXPECT_NO_THROW(ok = decoder.Decode(&decodable));
  EXPECT_TRUE(ok);

  EXPECT_TRUE(decoder.ReachedFinal());
  EXPECT_EQ(decoder.NumFramesDecoded(), 2);

  fst::Lattice lat;
  ASSERT_TRUE(decoder.GetBestPath(&lat, true));
  float total_cost = 0.0f;
  std::vector<std::pair<int32_t, int32_t>> path = GetLinearPath(lat,
                                                               &total_cost);
  // The best path consumes labels 1 then 2 (olabel == ilabel in this graph);
  // each frame has log-likelihood 1, so the acoustic cost is -2.
  std::vector<std::pair<int32_t, int32_t>> expected = {{1, 1}, {2, 2}};
  EXPECT_EQ(path, expected);
  EXPECT_FLOAT_EQ(total_cost, -2.0f);

  fst::Lattice raw;
  EXPECT_TRUE(decoder.GetRawLattice(&raw, true));
  EXPECT_GT(raw.NumStates(), 0);
}

// Sanity case for a graph WITH input-epsilon arcs: ProcessNonemitting() must
// propagate over them both before the first frame (frame -1) and between
// emitting frames.
TEST(LatticeSimpleDecoder, DecodeWithInputEpsilons) {
  srand(1);
  // s0 -0/7 (0.5)-> s1 -1/1-> s2 -0/8 (0.25)-> s3 -2/2-> s4 (final).
  fst::VectorFst<fst::StdArc> fst;
  for (int32_t i = 0; i != 5; ++i) fst.AddState();
  fst.SetStart(0);
  fst.SetFinal(4, fst::StdArc::Weight::One());
  fst.AddArc(0, fst::StdArc(0, 7, fst::StdArc::Weight(0.5f), 1));
  fst.AddArc(1, fst::StdArc(1, 1, fst::StdArc::Weight::One(), 2));
  fst.AddArc(2, fst::StdArc(0, 8, fst::StdArc::Weight(0.25f), 3));
  fst.AddArc(3, fst::StdArc(2, 2, fst::StdArc::Weight::One(), 4));

  LatticeSimpleDecoderConfig config(16.0f, 10.0f, 25);
  LatticeSimpleDecoder decoder(fst, config);

  TestDecodable decodable(std::vector<int32_t>{1, 2});
  bool ok = false;
  EXPECT_NO_THROW(ok = decoder.Decode(&decodable));
  EXPECT_TRUE(ok);

  EXPECT_TRUE(decoder.ReachedFinal());
  EXPECT_EQ(decoder.NumFramesDecoded(), 2);

  fst::Lattice lat;
  ASSERT_TRUE(decoder.GetBestPath(&lat, true));
  float total_cost = 0.0f;
  std::vector<std::pair<int32_t, int32_t>> path = GetLinearPath(lat,
                                                               &total_cost);
  // The two input-epsilon arcs appear in the best path with their olabels 7
  // and 8.  Total cost = graph (0.5 + 0.25) + acoustic (-1 -1) = -1.25.
  std::vector<std::pair<int32_t, int32_t>> expected = {
      {0, 7}, {1, 1}, {0, 8}, {2, 2}};
  EXPECT_EQ(path, expected);
  EXPECT_FLOAT_EQ(total_cost, -1.25f);
}

// Two competing paths; the best path must follow the acoustic scores.
TEST(LatticeSimpleDecoder, BestPathFollowsAcousticScores) {
  srand(1);
  // s0 -1/1-> s1 -2/2-> s3 (final); s0 -3/3-> s2 -4/4-> s3 (final).
  fst::VectorFst<fst::StdArc> fst;
  for (int32_t i = 0; i != 4; ++i) fst.AddState();
  fst.SetStart(0);
  fst.SetFinal(3, fst::StdArc::Weight::One());
  fst.AddArc(0, fst::StdArc(1, 1, fst::StdArc::Weight::One(), 1));
  fst.AddArc(1, fst::StdArc(2, 2, fst::StdArc::Weight::One(), 3));
  fst.AddArc(0, fst::StdArc(3, 3, fst::StdArc::Weight::One(), 2));
  fst.AddArc(2, fst::StdArc(4, 4, fst::StdArc::Weight::One(), 3));

  LatticeSimpleDecoderConfig config(16.0f, 10.0f, 25);
  LatticeSimpleDecoder decoder(fst, config);

  // Labels 1 and 2 score 1 per frame; labels 3 and 4 score -1 per frame.
  TestDecodable decodable(std::vector<int32_t>{1, 2});
  bool ok = false;
  EXPECT_NO_THROW(ok = decoder.Decode(&decodable));
  EXPECT_TRUE(ok);

  EXPECT_TRUE(decoder.ReachedFinal());

  fst::Lattice lat;
  ASSERT_TRUE(decoder.GetBestPath(&lat, true));
  float total_cost = 0.0f;
  std::vector<std::pair<int32_t, int32_t>> path = GetLinearPath(lat,
                                                               &total_cost);
  // The 3/4 path costs +2; the best path is 1/2 with acoustic cost -2.
  std::vector<std::pair<int32_t, int32_t>> expected = {{1, 1}, {2, 2}};
  EXPECT_EQ(path, expected);
  EXPECT_FLOAT_EQ(total_cost, -2.0f);
}

}  // namespace khg
