// sherpa-onnx-style consumer spike: one executable linking BOTH static cores
// (kaldi-hmm-gmm-core + kaldi-decoder-core), no Python.  Exercises the exact
// C++ surface sherpa-onnx will use:
//   khg::TransitionModel (implements kaldi_decoder::TransitionInformation)
//   khg::AddTransitionProbs (hmm-utils)
//   kaldi_decoder::LatticeFasterDecoder + DecodeUtteranceLatticeFaster
//   kaldi_decoder::LatticeBestPath
// Exit code 0 = the whole chain works from a single consumer.
#include <cstdio>
#include <memory>
#include <sstream>
#include <vector>

#include "fst/fstlib.h"
#include "kaldi-decoder/csrc/decoder-wrappers.h"
#include "kaldi-decoder/csrc/lattice-faster-decoder.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/hmm-topology.h"
#include "kaldi-hmm-gmm/csrc/hmm-utils.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"

namespace {

class TableDecodable : public kaldi_decoder::DecodableInterface {
 public:
  explicit TableDecodable(int32_t num_indices) : num_indices_(num_indices) {}
  float LogLikelihood(int32_t frame, int32_t index) override {
    return (index == frame + 1) ? 0.0f : -10.0f;
  }
  int32_t NumFramesReady() const override { return 2; }
  bool IsLastFrame(int32_t frame) const override { return frame == 1; }
  int32_t NumIndices() const override { return num_indices_; }

 private:
  int32_t num_indices_;
};

const char *kTopo =
    "<Topology>\n"
    "<TopologyEntry>\n"
    "<ForPhones> 1 </ForPhones>\n"
    "<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>\n"
    "<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>\n"
    "<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>\n"
    "<State> 3 </State>\n"
    "</TopologyEntry>\n"
    "</Topology>\n";

}  // namespace

int main() {
  // 1) A tiny transition model (the spike of the khg surface).
  khg::HmmTopology topo;
  std::istringstream topo_is(kTopo);
  topo.Read(topo_is, false);
  std::vector<int32_t> phones = topo.GetPhones();
  std::vector<int32_t> phone2num_pdf;
  topo.GetPhoneToNumPdfClasses(&phone2num_pdf);
  std::unique_ptr<khg::ContextDependency> tree(
      khg::MonophoneContextDependency(phones, phone2num_pdf));
  khg::TransitionModel tm(*tree, topo);

  // 2) Graph: s0 --tid 1/word 10--> s1 --tid 2/word 20--> s2 (final).
  fst::VectorFst<fst::StdArc> graph;
  auto s0 = graph.AddState();
  auto s1 = graph.AddState();
  auto s2 = graph.AddState();
  graph.SetStart(s0);
  graph.AddArc(s0, fst::StdArc(1, 10, fst::StdArc::Weight::One(), s1));
  graph.AddArc(s1, fst::StdArc(2, 20, fst::StdArc::Weight::One(), s2));
  graph.SetFinal(s2, fst::StdArc::Weight::One());

  // 3) khg: add transition probs (compile-train-graphs style, once).
  std::vector<int32_t> no_disambig;
  khg::AddTransitionProbs(tm, no_disambig, 1.0f, 1.0f, &graph);

  // 4) kd: decode + lattice best path.
  TableDecodable decodable(tm.NumTransitionIds());
  kaldi_decoder::LatticeFasterDecoderConfig config;
  kaldi_decoder::LatticeFasterDecoder decoder(graph, config);
  std::vector<int32_t> ali;
  std::vector<int32_t> words;
  double like = 0;
  fst::VectorFst<fst::LatticeArc> lat;
  bool ok = kaldi_decoder::DecodeUtteranceLatticeFaster(
      decoder, decodable, tm, "utt", /*allow_partial=*/true,
      /*acoustic_scale=*/1.0f, &ali, &words, &like, &lat);

  std::vector<int32_t> ali2;
  std::vector<int32_t> words2;
  double like2 = 0;
  bool ok2 = kaldi_decoder::LatticeBestPath(lat, 1.0f, 1.0f, &ali2, &words2,
                                            &like2);

  std::printf("consumer spike: decode_ok=%d ali=%zu words=%zu like=%.3f | "
              "best_path_ok=%d words=%zu like2=%.3f\n",
              ok, ali.size(), words.size(), like, ok2, words2.size(), like2);
  return (ok && ok2 && !words.empty() && words == words2) ? 0 : 1;
}
