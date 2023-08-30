// kaldi-hmm-gmm/python/csrc/decoder-wrappers.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/decoder-wrappers.h"

#include <string>
#include <tuple>
#include <vector>

#include "kaldi-hmm-gmm/csrc/decoder-wrappers.h"

namespace khg {

static void PybindAlignConfig(py::module *m) {
  using PyClass = AlignConfig;
  py::class_<PyClass>(*m, "AlignConfig")
      .def(py::init<float, float, bool>(), py::arg("beam") = 200.0,
           py::arg("retry_beam") = 0.0, py::arg("careful") = false)
      .def_readwrite("beam", &PyClass::beam)
      .def_readwrite("retry_beam", &PyClass::retry_beam)
      .def_readwrite("careful", &PyClass::careful);
}

static void PybindAlignUtteranceWrapper(py::module *m) {
  m->def(
      "align_utterance_wrapper",
      [](const AlignConfig &config, const std::string &utt,
         float acoustic_scale, fst::VectorFst<fst::StdArc> *fst,
         DecodableInterface *decodable, int32_t num_done, int32_t num_error,
         int32_t num_retried, double tot_like, int64_t frame_count)
          -> std::tuple<int32_t, int32_t, int32_t, double, int64_t,
                        std::vector<int32_t>, std::vector<int32_t>> {
        std::vector<int32_t> alignment;
        std::vector<int32_t> words;
        AlignUtteranceWrapper(config, utt, acoustic_scale, fst, decodable,
                              &num_done, &num_error, &num_retried, &tot_like,
                              &frame_count, &alignment, &words);

        return std::make_tuple(num_done, num_error, num_retried, tot_like,
                               frame_count, alignment, words);
      },
      py::arg("config"), py::arg("utt"), py::arg("acoustic_scale"),
      py::arg("fst"), py::arg("decodable"), py::arg("num_done"),
      py::arg("num_error"), py::arg("num_retried"), py::arg("tot_like"),
      py::arg("frame_count"));
}

static void PybindDecodeUtteranceLatticeSimple(py::module *m) {
  m->def(
      "decode_utterance_lattice_simple",
      [](LatticeSimpleDecoder &decoder,  // not const but is really an input.
         DecodableInterface &decodable,  // not const but is really an input.
         const TransitionInformation &trans_model, const std::string &utt,
         bool allow_partial) -> std::tuple<bool, std::vector<int32_t>,
                                           std::vector<int32_t>, double> {
        std::vector<int32_t> alignments;
        std::vector<int32_t> words;
        double log_like = 0;
        bool succeeded = DecodeUtteranceLatticeSimple(
            decoder, decodable, trans_model, utt, allow_partial, &alignments,
            &words, &log_like);

        return std::make_tuple(succeeded, alignments, words, log_like);
      },
      py::arg("decoder"), py::arg("decodable"), py::arg("trans_model"),
      py::arg("utt"), py::arg("allow_partial"));
}

template <typename PyClass>
static void PybindDecodeUtteranceLatticeFaster(py::module *m) {
  m->def(
      "decode_utterance_lattice_faster",
      [](PyClass &decoder,               // not const but is really an input.
         DecodableInterface &decodable,  // not const but is really an input.
         const TransitionInformation &trans_model, const std::string &utt,
         bool allow_partial) -> std::tuple<bool, std::vector<int32_t>,
                                           std::vector<int32_t>, double> {
        std::vector<int32_t> alignments;
        std::vector<int32_t> words;
        double log_like = 0;
        bool succeeded = DecodeUtteranceLatticeFaster(
            decoder, decodable, trans_model, utt, allow_partial, &alignments,
            &words, &log_like);

        return std::make_tuple(succeeded, alignments, words, log_like);
      },
      py::arg("decoder"), py::arg("decodable"), py::arg("trans_model"),
      py::arg("utt"), py::arg("allow_partial"));
}

void PybindDecoderWrappers(py::module *m) {
  PybindAlignConfig(m);
  PybindAlignUtteranceWrapper(m);
  PybindDecodeUtteranceLatticeSimple(m);

  PybindDecodeUtteranceLatticeFaster<
      LatticeFasterDecoderTpl<fst::Fst<fst::StdArc>>>(m);

  PybindDecodeUtteranceLatticeFaster<
      LatticeFasterDecoderTpl<fst::VectorFst<fst::StdArc>>>(m);

  PybindDecodeUtteranceLatticeFaster<
      LatticeFasterDecoderTpl<fst::ConstFst<fst::StdArc>>>(m);
}

}  // namespace khg
