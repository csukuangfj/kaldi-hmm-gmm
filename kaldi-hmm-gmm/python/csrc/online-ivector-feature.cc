// kaldi-hmm-gmm/kaldi-hmm-gmm/python/csrc/online-ivector-feature.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/online-ivector-feature.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/online-ivector-feature.h"

namespace khg {

static void PybindOnlineSilenceWeighting(py::module *m) {
  using PyConfig = OnlineSilenceWeightingConfig;
  py::class_<PyConfig>(*m, "OnlineSilenceWeightingConfig",
                       R"doc(kaldi's ``OnlineSilenceWeightingConfig``
(online2/online-ivector-feature.h).

vosk sets ``silence_phones_str`` from ``--endpoint.silence-phones`` in
``conf/model.conf`` and ``silence_weight=0.001``.)doc")
      .def(py::init([](const std::string &silence_phones_str,
                       float silence_weight, float max_state_duration) {
             PyConfig c;
             c.silence_phones_str = silence_phones_str;
             c.silence_weight = silence_weight;
             c.max_state_duration = max_state_duration;
             return c;
           }),
           py::arg("silence_phones_str") = "", py::arg("silence_weight") = 1.0,
           py::arg("max_state_duration") = -1.0,
           R"doc(Construct (defaults match kaldi: weighting off).

:param silence_phones_str: Silence phone ids separated by ``:`` or ``,``.
:param silence_weight: Weight of silence frames (1.0 = no weighting).
:param max_state_duration: Runs of one transition-id at least this long are
   treated as silence (<= 0: off).)doc")
      .def_readwrite("silence_phones_str", &PyConfig::silence_phones_str)
      .def_readwrite("silence_weight", &PyConfig::silence_weight)
      .def_readwrite("max_state_duration", &PyConfig::max_state_duration)
      .def("active", &PyConfig::Active,
           "True if silence weighting is in effect.");

  using PyClass = OnlineSilenceWeighting;
  py::class_<PyClass>(*m, "OnlineSilenceWeighting",
                      R"doc(kaldi's ``OnlineSilenceWeighting``: turns the
decoder's best path into per-frame weights for
:meth:`OnlineIvectorFeature.update_frame_weights`, down-weighting silence.

Unlike kaldi, :meth:`compute_current_traceback` takes the best path as
transition-ids (one per decoder frame), e.g. from kaldi-decoder's
``LatticeFasterOnlineDecoder.get_best_path_transition_ids(False)``.

Example (vosk's ``Recognizer::UpdateSilenceWeights``)::

    sw = khg.OnlineSilenceWeighting(tm, cfg, frame_subsampling_factor=3)
    if sw.active() and ivector.num_frames_ready > 0:
        sw.compute_current_traceback(
            decoder.get_best_path_transition_ids(False))
        ivector.update_frame_weights(
            sw.get_delta_weights(ivector.num_frames_ready))

**Lifetime:** keeps a reference to the transition model.)doc")
      .def(py::init<const TransitionModel &,
                    const OnlineSilenceWeightingConfig &, int32_t>(),
           py::arg("trans_model"), py::arg("config"),
           py::arg("frame_subsampling_factor") = 1, py::keep_alive<1, 2>())
      .def("active", &PyClass::Active)
      .def("compute_current_traceback", &PyClass::ComputeCurrentTraceback,
           py::arg("transition_ids"),
           R"doc(Store the current best path: one transition-id per decoder
frame, all frames decoded so far.)doc")
      .def(
          "get_delta_weights",
          [](PyClass &self, int32_t num_frames_ready,
             int32_t first_decoder_frame) {
            std::vector<std::pair<int32_t, float>> delta_weights;
            self.GetDeltaWeights(num_frames_ready, first_decoder_frame,
                                 &delta_weights);
            return delta_weights;
          },
          py::arg("num_frames_ready"), py::arg("first_decoder_frame") = 0,
          R"doc(``(input_frame, delta_weight)`` pairs for
:meth:`OnlineIvectorFeature.update_frame_weights`.

:param num_frames_ready: Input frames ready (the i-vector feature's
   ``num_frames_ready``).
:param first_decoder_frame: Input-frame offset of decoder frame 0.)doc")
      .def(
          "get_nonsilence_frames",
          [](PyClass &self, int32_t num_frames_ready,
             int32_t first_decoder_frame) {
            std::vector<int32_t> frames;
            self.GetNonsilenceFrames(num_frames_ready, first_decoder_frame,
                                     &frames);
            return frames;
          },
          py::arg("num_frames_ready"), py::arg("first_decoder_frame") = 0,
          "Decoder frames of the traceback that are not silence.");
}

void PybindOnlineIvectorFeature(py::module *m) {
  using PyClass = OnlineIvectorFeature;
  py::class_<PyClass>(*m, "OnlineIvectorFeature",
                      R"doc(kaldi's ``OnlineIvectorFeature``
(online2/online-ivector-feature.h): the i-vector kaldi's nnet3 online
decoders and vosk feed to the acoustic model.

Raw features (e.g. MFCC) are pushed with :meth:`accept_features`; stats are
accumulated lazily up to the frame asked for in :meth:`get_frame`.  With
``config.use_most_recent_ivector`` (default) that returns the estimate at the
requested frame.  Frame weights (silence weighting, see
:class:`OnlineSilenceWeighting`) are optional.

Example::

    cfg = khg.OnlineIvectorExtractorConfig(lda_mat=..., global_cmvn_stats=...,
                                           max_count=100)  # vosk
    feat = khg.OnlineIvectorFeature(cfg, extractor, ubm)
    feat.accept_features(mfcc)          # [T, F] float32
    iv = feat.get_frame(feat.num_frames_ready - 1)

**Lifetime:** keeps references to the extractor and the UBM.)doc")
      .def(py::init<const OnlineIvectorExtractorConfig &,
                    const IvectorExtractor &, const DiagGmm &>(),
           py::arg("config"), py::arg("extractor"), py::arg("ubm"),
           py::keep_alive<1, 3>(), py::keep_alive<1, 4>(),
           py::call_guard<py::gil_scoped_release>())
      .def("accept_features", &PyClass::AcceptFeatures, py::arg("feats"),
           py::call_guard<py::gil_scoped_release>(),
           "Append raw feature frames ``[T, F]`` (rows = frames).")
      .def("input_finished", &PyClass::InputFinished,
           "No more frames; the right-context frames become ready.")
      .def_property_readonly("dim", &PyClass::Dim, "The i-vector dimension.")
      .def_property_readonly("num_frames_ready", &PyClass::NumFramesReady,
                             R"doc(Frames whose i-vector can be requested
(raw frames minus the splice right context until :meth:`input_finished`).)doc")
      .def("is_last_frame", &PyClass::IsLastFrame, py::arg("frame"))
      .def(
          "get_frame",
          [](PyClass &self, int32_t frame) {
            if (frame < 0 || frame >= self.NumFramesReady()) {
              throw py::index_error("frame out of range [0, num_frames_ready)");
            }
            FloatVector feat;
            self.GetFrame(frame, &feat);
            return feat;
          },
          py::arg("frame"), py::call_guard<py::gil_scoped_release>(),
          R"doc(The i-vector for ``frame`` (float32 ``[dim]``, prior offset
removed from dimension 0, as in kaldi).)doc")
      .def("update_frame_weights", &PyClass::UpdateFrameWeights,
           py::arg("delta_weights"),
           R"doc(Add ``(frame, delta_weight)`` pairs (from
:meth:`OnlineSilenceWeighting.get_delta_weights`).)doc")
      .def_property_readonly("num_frames", &PyClass::NumFrames,
                             "Weighted frame count (kaldi's NumFrames()).")
      .def("objf_impr_per_frame", &PyClass::ObjfImprPerFrame,
           "Objective-function improvement per frame (diagnostic).");

  PybindOnlineSilenceWeighting(m);
}

}  // namespace khg
