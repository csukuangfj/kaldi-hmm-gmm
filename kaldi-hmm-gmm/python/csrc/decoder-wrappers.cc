// kaldi-hmm-gmm/python/csrc/decoder-wrappers.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// Thin convenience wrappers over kaldi-decoder-core.  The decoder
// implementation lives in ../kaldi-decoder; this file only glues it to
// khg's decodables/transition models so the recipe hot path stays in C++.
//
// This file also registers the kaldi-decoder option structs used in the
// signatures below (AlignConfig, LatticeFasterDecoderConfig,
// LatticeSimpleDecoderConfig) and LatticeBestPath -- all module_local:
// kaldi-hmm-gmm is self-contained at the Python level.

#include "kaldi-hmm-gmm/python/csrc/decoder-wrappers.h"

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kaldi-decoder/csrc/decoder-wrappers.h"
#include "kaldi-decoder/csrc/lattice-faster-decoder.h"
#include "kaldi-decoder/csrc/lattice-simple-decoder.h"
#include "kaldi-hmm-gmm/csrc/hmm-utils.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "kaldi-hmm-gmm/python/csrc/fst-caster.h"

namespace khg {

static void PybindAlignConfig(py::module *m) {
  using PyClass = kaldi_decoder::AlignConfig;
  auto cls =
      py::class_<PyClass>(
          *m, "AlignConfig", py::module_local(),
          R"doc(Alignment beam settings (kaldi's ``gmm-align-compiled`` flags).)doc")
          .def(py::init<float, float, bool>(), py::arg("beam") = 200.0,
               py::arg("retry_beam") = 0.0, py::arg("careful") = false)
          .def_readwrite(
              "beam", &PyClass::beam,
              R"doc(Wide alignment beam (kaldi ``--beam``, default 200).)doc")
          .def_readwrite(
              "retry_beam", &PyClass::retry_beam,
              R"doc(Retry beam on failure; 0 disables (kaldi ``--retry-beam``).)doc")
          .def_readwrite("careful", &PyClass::careful,
                         R"doc(Careful alignment: detects "blind alley" failures
(kaldi ``--careful``); MODIFIES the graph.)doc");
}

static void PybindLatticeFasterDecoderConfig(py::module *m) {
  using PyClass = kaldi_decoder::LatticeFasterDecoderConfig;
  auto cls =
      py::class_<PyClass>(
          *m, "LatticeFasterDecoderConfig", py::module_local(),
          R"doc(Options for the lattice-faster decoder (kaldi's
`LatticeFasterDecoderConfig
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/lattice-faster-decoder.h>`_).
All fields writable; defaults match kaldi's exactly.)doc")
          .def(py::init<float, int32_t, int32_t, float, int32_t, bool, float,
                        float, float, int32_t, int32_t>(),
               py::arg("beam") = 16.0,
               py::arg("max_active") = std::numeric_limits<int32_t>::max(),
               py::arg("min_active") = 200, py::arg("lattice_beam") = 10.0,
               py::arg("prune_interval") = 25,
               py::arg("determinize_lattice") = true,
               py::arg("beam_delta") = 0.5, py::arg("hash_ratio") = 2.0,
               py::arg("prune_scale") = 0.1,
               py::arg("memory_pool_tokens_block_size") = (1 << 8),
               py::arg("memory_pool_links_block_size") = (1 << 8))
          .def_readwrite("beam", &PyClass::beam,
                         R"doc(Decoder beam (kaldi ``--beam``): controls which
paths stay alive.)doc")
          .def_readwrite(
              "max_active", &PyClass::max_active,
              R"doc(Max active tokens per frame (kaldi ``--max-active``).)doc")
          .def_readwrite(
              "min_active", &PyClass::min_active,
              R"doc(Don't prune below this many tokens (kaldi ``--min-active``).)doc")
          .def_readwrite("lattice_beam", &PyClass::lattice_beam,
                         R"doc(Lattice beam (kaldi ``--lattice-beam``): how much
worse than the best path an alternative may be to stay in the lattice.)doc")
          .def_readwrite(
              "prune_interval", &PyClass::prune_interval,
              R"doc(Prune every this many frames (kaldi ``--prune-interval``).)doc")
          .def_readwrite("determinize_lattice", &PyClass::determinize_lattice,
                         R"doc(Whether the CALLER will determinize the output
lattice (kaldi ``--determinize-lattice``); not inspected by the decoder.)doc")
          .def_readwrite(
              "beam_delta", &PyClass::beam_delta,
              R"doc(Adaptive beam increment (kaldi ``--beam-delta``).)doc")
          .def_readwrite(
              "hash_ratio", &PyClass::hash_ratio,
              R"doc(Hash-table sizing ratio (kaldi ``--hash-ratio``).)doc")
          .def_readwrite(
              "prune_scale", &PyClass::prune_scale,
              R"doc(Internal token-pruning scale (rarely changed).)doc")
          .def_readwrite("memory_pool_tokens_block_size",
                         &PyClass::memory_pool_tokens_block_size,
                         R"doc(Token memory-pool block size.)doc")
          .def_readwrite("memory_pool_links_block_size",
                         &PyClass::memory_pool_links_block_size,
                         R"doc(Forward-link memory-pool block size.)doc")
          .def(
              "check", &PyClass::Check,
              R"doc(Validate option ranges; raises on bad values (kaldi ``Check()``).)doc")
          .def("__str__", &PyClass::ToString);
}

static void PybindLatticeSimpleDecoderConfig(py::module *m) {
  using PyClass = kaldi_decoder::LatticeSimpleDecoderConfig;
  auto cls =
      py::class_<PyClass>(
          *m, "LatticeSimpleDecoderConfig", py::module_local(),
          R"doc(Options for the lattice-simple decoder (kaldi's
`LatticeSimpleDecoderConfig
<https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/lattice-simple-decoder.h>`_).
All fields writable; defaults match kaldi's.)doc")
          .def(py::init<float, float, int32_t, bool, bool, float, float>(),
               py::arg("beam") = 16.0, py::arg("lattice_beam") = 10.0,
               py::arg("prune_interval") = 25,
               py::arg("determinize_lattice") = true,
               py::arg("prune_lattice") = true, py::arg("beam_ratio") = 0.9,
               py::arg("prune_scale") = 0.1)
          .def_readwrite("beam", &PyClass::beam)
          .def_readwrite("lattice_beam", &PyClass::lattice_beam)
          .def_readwrite("prune_interval", &PyClass::prune_interval)
          .def_readwrite("determinize_lattice", &PyClass::determinize_lattice)
          .def_readwrite("prune_lattice", &PyClass::prune_lattice)
          .def_readwrite("beam_ratio", &PyClass::beam_ratio)
          .def_readwrite("prune_scale", &PyClass::prune_scale)
          .def("__str__", &PyClass::ToString);
}

static void PybindLatticeBestPath(py::module *m) {
  m->def(
      "lattice_best_path",
      [](const fst::VectorFst<fst::LatticeArc> &lat, float lm_scale,
         float acoustic_scale) -> std::tuple<bool, std::vector<int32_t>,
                                             std::vector<int32_t>, double> {
        std::vector<int32_t> alignments;
        std::vector<int32_t> words;
        double log_like = 0;
        bool ok = kaldi_decoder::LatticeBestPath(lat, lm_scale, acoustic_scale,
                                                 &alignments, &words,
                                                 &log_like);
        return std::make_tuple(ok, alignments, words, log_like);
      },
      py::arg("lat"), py::arg("lm_scale") = 1.0,
      py::arg("acoustic_scale") = 1.0, py::call_guard<py::gil_scoped_release>(),
      R"doc(1-best of a lattice under lm/acoustic scaling (kaldi's
`lattice-best-path <https://github.com/kaldi-asr/kaldi/blob/master/src/latbin/lattice-best-path.cc>`_).

Minimizes ``lm_scale * graph_cost + acoustic_scale * acoustic_cost``.
On the lattices returned by :func:`decode_utterance_lattice_faster` (acoustic
costs already divided by the decode acoustic scale), use
``acoustic_scale=1.0`` and sweep ``lm_scale`` for lmwt scoring.

:param lat: Input lattice (``kaldifst.Lattice``).
:param lm_scale: Weight of the graph cost (kaldi ``--lm-scale``).
:param acoustic_scale: Weight of the acoustic cost
   (kaldi ``--acoustic-scale``).
:return: ``(ok, alignment, words, like)``; ``ok`` is False when the
   lattice has no complete path (outputs empty).)doc");
}

void PybindDecoderWrappers(py::module *m) {
  PybindAlignConfig(m);
  PybindLatticeFasterDecoderConfig(m);
  PybindLatticeSimpleDecoderConfig(m);
  PybindLatticeBestPath(m);

  // Alignment: kaldi's gmm-align-compiled core (AddTransitionProbs here +
  // kaldi_decoder::AlignUtteranceWrapper).  The five counters are the
  // caller's running totals and are accumulated, as in kaldi's writers.
  m->def(
      "align_utterance",
      [](const kaldi_decoder::AlignConfig &config, const std::string &utt,
         float acoustic_scale, float transition_scale, float self_loop_scale,
         const TransitionModel &trans_model, fst::VectorFst<fst::StdArc> *fst,
         kaldi_decoder::DecodableInterface &decodable, int32_t num_done,
         int32_t num_error, int32_t num_retried, double tot_like,
         int64_t frame_count)
          -> std::tuple<int32_t, int32_t, int32_t, double, int64_t,
                        std::vector<int32_t>, std::vector<int32_t>> {
        AddTransitionProbs(trans_model, std::vector<int32_t>{},
                           transition_scale, self_loop_scale, fst);
        std::vector<int32_t> alignment;
        std::vector<int32_t> words;
        kaldi_decoder::AlignUtteranceWrapper(config, utt, acoustic_scale, fst,
                                             &decodable, &num_done,
                                             &num_error, &num_retried,
                                             &tot_like, &frame_count,
                                             &alignment, &words);
        return std::make_tuple(num_done, num_error, num_retried, tot_like,
                               frame_count, alignment, words);
      },
      py::arg("config"), py::arg("utt"), py::arg("acoustic_scale"),
      py::arg("transition_scale"), py::arg("self_loop_scale"),
      py::arg("trans_model"), py::arg("fst"), py::arg("decodable"),
      py::arg("num_done") = 0, py::arg("num_error") = 0,
      py::arg("num_retried") = 0, py::arg("tot_like") = 0.0,
      py::arg("frame_count") = 0,
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Force-align one utterance (kaldi's ``gmm-align-compiled`` core).

Adds the transition probabilities to the training graph (``AddTransitionProbs``
with ``transition_scale``/``self_loop_scale``) and runs the Viterbi alignment
in a single C++ call.  **Mutates** the graph in place (same as kaldi's
``gmm-align-compiled``); pass a copy to reuse it afterwards.

Returns ``(num_done, num_error, num_retried, tot_like, frame_count,
alignment, words)``: the five counters are INPUTS that get incremented
(kaldi's accumulating-table semantics) -- omit them to get this utterance's
contributions only, or thread running totals through a loop.  ``alignment``
is transition-ids.

:param config: Alignment beams (:class:`AlignConfig`).
:param utt: Utterance id (for log messages).
:param acoustic_scale: Acoustic scale used in the alignment.
:param transition_scale: Transition-probability scale.
:param self_loop_scale: Self-loop versus non-self-loop log-prob scale.
:param trans_model: Transition model (:class:`TransitionModel`; needed for
   ``AddTransitionProbs``).
:param fst: Per-utterance training graph; mutated in place.
:param decodable: Scores (:class:`DecodableInterface`).)doc");

  // Decoding: kaldi's gmm-latgen-simple/faster output block.  The decoder is
  // constructed internally (no cross-module decoder objects).
  // Returns (ok, alignment, words, like, lat); `lat` is the lattice exactly
  // as kaldi writes it (acoustic costs / acoustic_scale) -- feed it to
  // lattice_best_path for lmwt/wip scoring.
  m->def(
      "decode_utterance_lattice_simple",
      [](const kaldi_decoder::LatticeSimpleDecoderConfig &config,
         fst::VectorFst<fst::StdArc> *fst,
         kaldi_decoder::DecodableInterface &decodable,
         const kaldi_decoder::TransitionInformation &trans_model,
         const std::string &utt, bool allow_partial, float acoustic_scale)
          -> std::tuple<bool, std::vector<int32_t>, std::vector<int32_t>,
                        double, fst::VectorFst<fst::LatticeArc>> {
        kaldi_decoder::LatticeSimpleDecoder decoder(*fst, config);
        std::vector<int32_t> alignment;
        std::vector<int32_t> words;
        double log_like = 0;
        fst::VectorFst<fst::LatticeArc> lat;
        bool ok = kaldi_decoder::DecodeUtteranceLatticeSimple(
            decoder, decodable, trans_model, utt, allow_partial,
            acoustic_scale, &alignment, &words, &log_like, &lat);
        return std::make_tuple(ok, alignment, words, log_like, lat);
      },
      py::arg("config"), py::arg("fst"), py::arg("decodable"),
      py::arg("trans_model"), py::arg("utt"), py::arg("allow_partial"),
      py::arg("acoustic_scale"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Decode one utterance with the lattice-simple decoder (kaldi's
``gmm-latgen-simple`` output block).  The decoder is constructed inside this
call; the graph ``fst`` is only borrowed for its duration.

Returns ``(ok, alignment, words, like, lat)``; ``lat`` is the lattice exactly
as kaldi writes it (acoustic costs divided by ``acoustic_scale``) -- feed it
to :func:`lattice_best_path` for lmwt/wip scoring.

:param config: Decoder options (:class:`LatticeSimpleDecoderConfig`).
:param fst: Decoding graph (e.g. HCLG); borrowed, not mutated.
:param decodable: Scores (:class:`DecodableInterface`).
:param trans_model: Transition information (any
   :class:`TransitionInformation`, e.g. :class:`TransitionModel`).
:param utt: Utterance id (for log messages).
:param allow_partial: Output partial results if no final state was reached.
:param acoustic_scale: Acoustic scale used in the decode.)doc");

  m->def(
      "decode_utterance_lattice_faster",
      [](const kaldi_decoder::LatticeFasterDecoderConfig &config,
         fst::VectorFst<fst::StdArc> *fst,
         kaldi_decoder::DecodableInterface &decodable,
         const kaldi_decoder::TransitionInformation &trans_model,
         const std::string &utt, bool allow_partial, float acoustic_scale)
          -> std::tuple<bool, std::vector<int32_t>, std::vector<int32_t>,
                        double, fst::VectorFst<fst::LatticeArc>> {
        kaldi_decoder::LatticeFasterDecoder decoder(*fst, config);
        std::vector<int32_t> alignment;
        std::vector<int32_t> words;
        double log_like = 0;
        fst::VectorFst<fst::LatticeArc> lat;
        bool ok = kaldi_decoder::DecodeUtteranceLatticeFaster(
            decoder, decodable, trans_model, utt, allow_partial,
            acoustic_scale, &alignment, &words, &log_like, &lat);
        return std::make_tuple(ok, alignment, words, log_like, lat);
      },
      py::arg("config"), py::arg("fst"), py::arg("decodable"),
      py::arg("trans_model"), py::arg("utt"), py::arg("allow_partial"),
      py::arg("acoustic_scale"),
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Decode one utterance with the lattice-faster decoder (kaldi's
``gmm-latgen-faster`` output block).  The decoder is constructed inside this
call; the graph ``fst`` is only borrowed for its duration.

Returns ``(ok, alignment, words, like, lat)``; ``lat`` is the lattice exactly
as kaldi writes it (acoustic costs divided by ``acoustic_scale``) -- feed it
to :func:`lattice_best_path` for lmwt/wip scoring.

:param config: Decoder options (:class:`LatticeFasterDecoderConfig`).
:param fst: Decoding graph (e.g. HCLG); borrowed, not mutated.
:param decodable: Scores (:class:`DecodableInterface`).
:param trans_model: Transition information (any
   :class:`TransitionInformation`, e.g. :class:`TransitionModel`).
:param utt: Utterance id (for log messages).
:param allow_partial: Output partial results if no final state was reached.
:param acoustic_scale: Acoustic scale used in the decode.)doc");
}

}  // namespace khg
