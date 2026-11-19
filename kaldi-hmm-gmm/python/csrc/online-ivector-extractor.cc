// kaldi-hmm-gmm/kaldi-hmm-gmm/python/csrc/online-ivector-extractor.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/online-ivector-extractor.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <string>

#include "kaldi-hmm-gmm/csrc/online-ivector-extractor.h"

namespace khg {

void PybindOnlineIvectorExtractor(py::module *m) {
  using PyConfig = OnlineIvectorExtractorConfig;
  py::class_<PyConfig> config(
      *m, "OnlineIvectorExtractorConfig",
      R"doc(Options for :class:`OnlineIvectorExtractor`, mirroring kaldi's
``OnlineIvectorExtractionConfig`` (online2/online-ivector-feature.h) plus the
running-CMVN options of the posterior chain.

Files of a model's ``ivector/`` directory map to the fields as follows:
``splice.conf`` -> ``left_context``/``right_context``,
``final.mat`` -> ``lda_mat``, ``global_cmvn.stats`` -> ``global_cmvn_stats``,
``online_cmvn.conf`` -> the ``cmn_*``/``normalize_*``/``skip_dims`` fields.

.. seealso:: :class:`OnlineIvectorExtractor`)doc");
  config
      .def(py::init([](int32_t left_context, int32_t right_context,
                       const FloatMatrix &lda_mat,
                       const DoubleMatrix &global_cmvn_stats, int32_t cmn_window,
                       int32_t speaker_frames, int32_t global_frames,
                       bool normalize_mean, bool normalize_variance,
                       const std::string &skip_dims,
                       bool online_cmvn_iextractor, double max_count,
                       int32_t num_cg_iters, int32_t ivector_period,
                       int32_t num_gselect, double min_post,
                       double posterior_scale) {
             PyConfig c;
             c.left_context = left_context;
             c.right_context = right_context;
             c.lda_mat = lda_mat;
             c.global_cmvn_stats = global_cmvn_stats;
             c.cmvn_opts.cmn_window = cmn_window;
             c.cmvn_opts.speaker_frames = speaker_frames;
             c.cmvn_opts.global_frames = global_frames;
             c.cmvn_opts.normalize_mean = normalize_mean;
             c.cmvn_opts.normalize_variance = normalize_variance;
             c.cmvn_opts.skip_dims = skip_dims;
             c.online_cmvn_iextractor = online_cmvn_iextractor;
             c.max_count = max_count;
             c.num_cg_iters = num_cg_iters;
             c.ivector_period = ivector_period;
             c.num_gselect = num_gselect;
             c.min_post = min_post;
             c.posterior_scale = posterior_scale;
             return c;
           }),
           py::arg("left_context") = 3, py::arg("right_context") = 3,
           py::arg("lda_mat") = FloatMatrix(),
           py::arg("global_cmvn_stats") = DoubleMatrix(),
           py::arg("cmn_window") = 600, py::arg("speaker_frames") = 600,
           py::arg("global_frames") = 200, py::arg("normalize_mean") = true,
           py::arg("normalize_variance") = false, py::arg("skip_dims") = "",
           py::arg("online_cmvn_iextractor") = false,
           py::arg("max_count") = 0.0, py::arg("num_cg_iters") = 15,
           py::arg("ivector_period") = 10, py::arg("num_gselect") = 0,
           py::arg("min_post") = 0.0, py::arg("posterior_scale") = 1.0,
           R"doc(Construct with the given options (defaults match kaldi).

:param left_context: Splice left context (``splice.conf``).
:param right_context: Splice right context (``splice.conf``).
:param lda_mat: Linear transform ``[lda_dim][spliced_dim]`` or
   ``[lda_dim][spliced_dim + 1]`` with an affine offset (``final.mat``).
:param global_cmvn_stats: Global stats ``[2, feat_dim + 1]``
   (``global_cmvn.stats``).
:param cmn_window: Running-CMVN window (``online_cmvn.conf``).
:param speaker_frames: Running-CMVN speaker frames.
:param global_frames: Running-CMVN global frames.
:param normalize_mean: Running-CMVN mean normalization.
:param normalize_variance: Running-CMVN variance normalization.
:param skip_dims: Colon-separated dims to skip, e.g. ``"13:14"``.
:param online_cmvn_iextractor: If ``True``, the stats chain is also
   CMVN-normalized (kaldi's flag; ``False`` in vosk's models).
:param max_count: Prior scaling above this frame count (vosk: 100).
:param num_cg_iters: CG iterations per estimate (kaldi: 15).
:param ivector_period: Re-estimate every N frames (kaldi: 10).
:param num_gselect: Keep top-N UBM Gaussians (0 = all).
:param min_post: UBM posterior pruning threshold (0 = off).
:param posterior_scale: Scale of UBM posteriors (0 < s <= 1).)doc")
      .def_readwrite("left_context", &PyConfig::left_context)
      .def_readwrite("right_context", &PyConfig::right_context)
      .def_property(
          "lda_mat", [](const PyConfig &c) { return c.lda_mat; },
          [](PyConfig &c, const FloatMatrix &m) { c.lda_mat = m; },
          "Linear transform [lda_dim][spliced_dim (+1)].")
      .def_property(
          "global_cmvn_stats",
          [](const PyConfig &c) { return c.global_cmvn_stats; },
          [](PyConfig &c, const DoubleMatrix &m) { c.global_cmvn_stats = m; },
          "Global CMVN stats [2][feat_dim + 1].")
      .def_property(
          "cmn_window",
          [](const PyConfig &c) { return c.cmvn_opts.cmn_window; },
          [](PyConfig &c, int32_t v) { c.cmvn_opts.cmn_window = v; },
          "Running-CMVN window.")
      .def_property(
          "speaker_frames",
          [](const PyConfig &c) { return c.cmvn_opts.speaker_frames; },
          [](PyConfig &c, int32_t v) { c.cmvn_opts.speaker_frames = v; },
          "Running-CMVN speaker frames.")
      .def_property(
          "global_frames",
          [](const PyConfig &c) { return c.cmvn_opts.global_frames; },
          [](PyConfig &c, int32_t v) { c.cmvn_opts.global_frames = v; },
          "Running-CMVN global frames.")
      .def_property(
          "normalize_mean",
          [](const PyConfig &c) { return c.cmvn_opts.normalize_mean; },
          [](PyConfig &c, bool v) { c.cmvn_opts.normalize_mean = v; },
          "Running-CMVN mean normalization.")
      .def_property(
          "normalize_variance",
          [](const PyConfig &c) { return c.cmvn_opts.normalize_variance; },
          [](PyConfig &c, bool v) { c.cmvn_opts.normalize_variance = v; },
          "Running-CMVN variance normalization.")
      .def_property(
          "skip_dims",
          [](const PyConfig &c) -> std::string { return c.cmvn_opts.skip_dims; },
          [](PyConfig &c, const std::string &v) { c.cmvn_opts.skip_dims = v; },
          "Colon-separated dims to skip in CMVN.")
      .def_readwrite("online_cmvn_iextractor",
                     &PyConfig::online_cmvn_iextractor)
      .def_readwrite("max_count", &PyConfig::max_count)
      .def_readwrite("num_cg_iters", &PyConfig::num_cg_iters)
      .def_readwrite("ivector_period", &PyConfig::ivector_period)
      .def_readwrite("num_gselect", &PyConfig::num_gselect)
      .def_readwrite("min_post", &PyConfig::min_post)
      .def_readwrite("posterior_scale", &PyConfig::posterior_scale);

  using PyDriver = OnlineIvectorExtractor;
  py::class_<PyDriver> driver(
      *m, "OnlineIvectorExtractor",
      R"doc(Streaming i-vector extractor (kaldi's ``OnlineIvectorFeature``
dataflow): running-CMVN -> splice -> lda -> UBM posteriors on one chain, raw
splice -> lda on the stats chain (unless ``online_cmvn_iextractor``), with
iVector re-estimation every ``ivector_period`` frames.

Example:

    cfg = khg.OnlineIvectorExtractorConfig(lda_mat=..., global_cmvn_stats=...)
    eng = khg.OnlineIvectorExtractor(cfg, extractor, ubm)
    eng.accept_features(mfcc)   # [T, 40] float32, int16-scale input audio
    eng.flush()
    ivs = eng.get_ivectors()    # [ceil(T/period), ivector_dim]

.. seealso:: :class:`IvectorExtractor`, :class:`OnlineIvectorExtractorConfig`)doc");
  driver
      .def(py::init<const OnlineIvectorExtractorConfig &,
                    const IvectorExtractor &, const DiagGmm &>(),
           py::arg("config"), py::arg("extractor"), py::arg("ubm"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Construct for one utterance stream.

:param config: The :class:`OnlineIvectorExtractorConfig`.
:param extractor: The trained :class:`IvectorExtractor` (``final.ie``).
:param ubm: The UBM :class:`DiagGmm` (``final.dubm``).)doc")
      .def("accept_features", &PyDriver::AcceptFeatures, py::arg("feats"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Feed raw feature frames (e.g. MFCC), rows = frames.
Frames are processed with the right-context lookahead of kaldi; call
:meth:`flush` at the end of the utterance.)doc")
      .def("flush", &PyDriver::Flush, py::call_guard<py::gil_scoped_release>(),
           "Process the tail of the utterance (call at utterance end).")
      .def("reset", &PyDriver::Reset, py::call_guard<py::gil_scoped_release>(),
           "Start a new utterance (running CMVN restarts from global stats).")
      .def(
          "get_ivector",
          [](const PyDriver &self, int32_t index) -> DoubleVector {
            return self.GetIvector(index);
          },
          py::arg("index"), R"doc(The iVector estimated after
``index * ivector_period`` frames.

:param index: Row index in ``[0, num_ivectors_ready)``.)doc")
      .def(
          "get_ivectors",
          [](const PyDriver &self) -> DoubleMatrix {
            return self.GetIvectors();
          },
          "All iVectors as rows ``[num_ivectors_ready, ivector_dim]``.")
      .def_property_readonly(
          "latest_ivector",
          [](const PyDriver &self) -> DoubleVector {
            return self.LatestIvector();
          },
          "The most recently estimated iVector (for streaming decoding).")
      .def_property_readonly("num_frames", &PyDriver::NumFrames,
                             "Number of frames fed.")
      .def_property_readonly("num_ivectors_ready",
                             &PyDriver::NumIvectorsReady,
                             "Number of iVectors estimated so far.")
      .def_property_readonly("feat_dim", &PyDriver::FeatDim,
                             "Input feature dimension.")
      .def_property_readonly("count", &PyDriver::Count,
                             "Weighted frame count in the stats.");
}

}  // namespace khg
