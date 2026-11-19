// kaldi-hmm-gmm/kaldi-hmm-gmm/python/csrc/ivector-extractor.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "kaldi-hmm-gmm/python/csrc/ivector-extractor.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/ivector-extractor.h"
#include "kaldi_native_io/csrc/kaldi-io.h"

namespace khg {

void PybindIvectorExtractor(py::module *m) {
  m->def(
      "read_diag_gmm",
      [](const std::string &filename) -> std::unique_ptr<DiagGmm> {
        bool binary = true;
        kaldiio::Input ki(filename, &binary);
        auto gmm = std::make_unique<DiagGmm>();
        gmm->Read(ki.Stream(), binary);
        return gmm;
      },
      py::arg("filename"), py::call_guard<py::gil_scoped_release>(),
      R"doc(Read a kaldi DiagGmm from a file, e.g. ``ivector/final.dubm``.

:param filename: Path of the model file.
:return: The :class:`DiagGmm`.)doc");

  m->def(
      "vector_to_posterior_entry",
      [](const FloatVector &log_likes, int32_t num_gselect,
         double min_post) -> GaussPost {
        return VectorToPosteriorEntry(log_likes, num_gselect, min_post);
      },
      py::arg("log_likes"), py::arg("num_gselect") = 0,
      py::arg("min_post") = 0.0, py::call_guard<py::gil_scoped_release>(),
      R"doc(Convert per-Gaussian log-likes to pruned, normalized posteriors
(kaldi's ``VectorToPosteriorEntry``).

:param log_likes: Log-likelihoods of shape ``[num_gauss]``.
:param num_gselect: Keep only the top-N Gaussians; ``0`` keeps all.
:param min_post: Drop trailing posteriors below ``min_post * total`` before
   renormalizing; ``0.0`` keeps whatever survives ``num_gselect``.
:return: Sparse posteriors as a list of ``(gaussian_index, weight)`` pairs.)doc");

  using PyStats = IvectorExtractorUtteranceStats;
  py::class_<PyStats> stats(*m, "IvectorExtractorUtteranceStats",
                            R"doc(Sufficient statistics for estimating the
iVector of one utterance (kaldi's ``IvectorExtractorUtteranceStats``, 1st-order
stats only).

.. seealso:: :meth:`IvectorExtractor.compute_ivector`)doc");
  stats
      .def(py::init<int32_t, int32_t>(), py::arg("num_gauss"),
           py::arg("feat_dim"),
           R"doc(Construct zeroed stats.

:param num_gauss: Number of UBM Gaussians.
:param feat_dim: Feature dimension (after lda transform).)doc")
      .def("acc_stats", &PyStats::AccStats, py::arg("feats"), py::arg("post"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Accumulate stats for a sequence of frames.

:param feats: Features of shape ``[T, D]`` (float32).
:param post: Per-frame sparse posteriors, ``T`` lists of
   ``(gaussian_index, weight)``.)doc")
      .def("scale", &PyStats::Scale, py::arg("scale"),
           py::call_guard<py::gil_scoped_release>(),
           R"doc(Scale the accumulated stats by ``0 <= scale <= 1``.)doc")
      .def_property_readonly("num_frames", &PyStats::NumFrames,
                             "Weighted frame count.");

  using PyExtractor = IvectorExtractor;
  py::class_<PyExtractor> extractor(
      *m, "IvectorExtractor",
      R"doc(A trained i-vector extractor (kaldi's ``final.ie``).

An i-vector is a low-dimensional (e.g. 100-400 dim) vector summarizing the
speaker (and channel) of an utterance.  See ``docs/ivector.md`` for the math.
Estimation needs the extractor plus Gaussian posteriors from a UBM.

Example:

    extractor = khg.IvectorExtractor.read("ivector/final.ie")
    ubm = khg.read_diag_gmm("ivector/final.dubm")
    # ... accumulate stats (or use OnlineIvectorExtractor) ...
    stats = khg.IvectorExtractorUtteranceStats(ubm.num_gauss, extractor.feat_dim)
    stats.acc_stats(feats, post)
    iv = extractor.compute_ivector(stats)  # np [ivector_dim]

.. seealso:: :class:`OnlineIvectorExtractor` for the streaming driver.)doc");
  extractor
      .def(py::init<>(), "Construct an empty extractor; use :meth:`read`.")
      .def_static(
          "read",
          [](const std::string &filename) -> PyExtractor {
            bool binary = true;
            kaldiio::Input ki(filename, &binary);
            PyExtractor e;
            e.Read(ki.Stream(), binary);
            return e;
          },
          py::arg("filename"), py::call_guard<py::gil_scoped_release>(),
          R"doc(Read an extractor from a file, e.g. ``ivector/final.ie``.

:param filename: Path of the model file.
:return: The :class:`IvectorExtractor`.)doc")
      .def(
          "compute_ivector",
          [](const PyExtractor &self, const PyStats &utt_stats) -> DoubleVector {
            DoubleVector mean(self.IvectorDim());
            self.GetIvectorDistribution(utt_stats, &mean, nullptr);
            return mean;
          },
          py::arg("utt_stats"), py::call_guard<py::gil_scoped_release>(),
          R"doc(Compute the iVector (posterior mean) from utterance stats.

:param utt_stats: Accumulated :class:`IvectorExtractorUtteranceStats`.
:return: The iVector of shape ``[ivector_dim]`` (float64); note the first
   dimension carries :attr:`prior_offset`.)doc")
      .def(
          "get_ivector_distribution",
          [](const PyExtractor &self, const PyStats &utt_stats)
              -> std::pair<DoubleVector, DoubleMatrix> {
            DoubleVector mean(self.IvectorDim());
            DoubleMatrix var(self.IvectorDim(), self.IvectorDim());
            self.GetIvectorDistribution(utt_stats, &mean, &var);
            return std::make_pair(mean, var);
          },
          py::arg("utt_stats"), py::call_guard<py::gil_scoped_release>(),
          R"doc(Gaussian approximation of the iVector distribution.

:param utt_stats: Accumulated :class:`IvectorExtractorUtteranceStats`.
:return: ``(mean, var)`` with shapes ``[S]`` and ``[S, S]``.)doc")
      .def_property_readonly("ivector_dim", &PyExtractor::IvectorDim,
                             "iVector dimension S.")
      .def_property_readonly("feat_dim", &PyExtractor::FeatDim,
                             "Feature dimension D (after lda transform).")
      .def_property_readonly("num_gauss", &PyExtractor::NumGauss,
                             "Number of UBM Gaussians I.")
      .def_property_readonly("prior_offset", &PyExtractor::PriorOffset,
                             R"doc(The prior over iVectors is not centered at
zero; its first dimension has this offset.)doc")
      .def_property_readonly("ivector_dependent_weights",
                             &PyExtractor::IvectorDependentWeights,
                             "Whether weight projection vectors w are used.");

  using PyOnline = OnlineIvectorEstimationStats;
  py::class_<PyOnline> online(
      *m, "OnlineIvectorEstimationStats",
      R"doc(Incremental i-vector estimation as frames arrive (kaldi's
``OnlineIvectorEstimationStats``).  Maintains the linear/quadratic terms of
the iVector posterior directly, applies the ``max_count`` prior scaling, and
estimates the iVector with warm-started conjugate gradient.

.. seealso:: :class:`OnlineIvectorExtractor`, which drives this class.)doc");
  online
      .def(py::init<int32_t, double, double>(), py::arg("ivector_dim"),
           py::arg("prior_offset"), py::arg("max_count") = 0.0,
           R"doc(Construct with a prior only (no data).

:param ivector_dim: iVector dimension S.
:param prior_offset: :attr:`IvectorExtractor.prior_offset` of the extractor.
:param max_count: If > 0, the prior is weighted more once the frame count
   exceeds this value (vosk uses 100).)doc")
      .def(
          "acc_stats",
          [](PyOnline &self, const IvectorExtractor &extractor,
             const FloatVector &feature, const GaussPost &gauss_post) {
            self.AccStats(extractor, feature, gauss_post);
          },
          py::arg("extractor"), py::arg("feature"), py::arg("gauss_post"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Accumulate stats for one frame.

:param extractor: The extractor providing M/Sigma^-1.
:param feature: One feature frame of shape ``[D]`` (float32).
:param gauss_post: Sparse ``(gaussian_index, weight)`` posteriors.)doc")
      .def(
          "acc_stats_sequence",
          [](PyOnline &self, const IvectorExtractor &extractor,
             const FloatMatrix &features,
             const std::vector<GaussPost> &gauss_post) {
            self.AccStats(extractor, features, gauss_post);
          },
          py::arg("extractor"), py::arg("feats"), py::arg("post"),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Accumulate stats for a sequence of frames.

:param feats: Features of shape ``[T, D]`` (float32).
:param post: ``T`` lists of ``(gaussian_index, weight)``.)doc")
      .def(
          "get_ivector",
          [](const PyOnline &self, int32_t num_cg_iters,
             const DoubleVector &ivector) -> DoubleVector {
            DoubleVector iv = ivector.size() == self.IvectorDim()
                                  ? ivector
                                  : DoubleVector::Zero(self.IvectorDim());
            self.GetIvector(num_cg_iters, &iv);
            return iv;
          },
          py::arg("num_cg_iters") = 15, py::arg("ivector") = DoubleVector(),
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Current iVector estimate.

:param num_cg_iters: Conjugate-gradient iterations (warm-started); pass a
   negative value to solve exactly.
:param ivector: Optional warm start of shape ``[S]`` (e.g. the previous
   estimate); defaults to the prior mean.
:return: The iVector of shape ``[S]`` (float64).)doc")
      .def("scale", &PyOnline::Scale, py::arg("scale"),
           py::call_guard<py::gil_scoped_release>(),
           "Scale the data stats by 0 <= scale <= 1 (the prior is restored).")
      .def("objf_change", &PyOnline::ObjfChange, py::arg("ivector"),
           py::call_guard<py::gil_scoped_release>(),
           "Objective improvement per frame vs the default iVector (diagnostic).")
      .def_property_readonly("count", &PyOnline::Count,
                             "Weighted frame count.")
      .def_property_readonly("num_frames", &PyOnline::NumFrames,
                             "Weighted frame count.")
      .def_property_readonly("ivector_dim", &PyOnline::IvectorDim,
                             "iVector dimension S.")
      .def_property_readonly("prior_offset", &PyOnline::PriorOffset,
                             "Prior offset of the extractor.")
      .def_property_readonly(
          "linear_term", [](const PyOnline &self) { return self.LinearTerm(); },
          "Linear term of the posterior [S].")
      .def_property_readonly(
          "quadratic_term",
          [](const PyOnline &self) { return self.QuadraticTerm(); },
          "Quadratic term of the posterior [S, S].");
}

}  // namespace khg
