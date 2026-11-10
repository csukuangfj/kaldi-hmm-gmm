// kaldi-hmm-gmm/csrc/diag-gmm.h
//
// Copyright 2009-2011  Microsoft Corporation;
//                      Saarland University (Author: Arnab Ghoshal);
//                      Georg Stemmer;  Jan Silovsky
//           2012       Arnab Ghoshal
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_DIAG_GMM_H_

/**
 * \file diag-gmm.h
 * \brief Diagonal-covariance Gaussian mixture models (\c DiagGmm) stored in
 *  exponential form.
 *
 * This file is copied/modified from ../kaldi/src/gmm/diag-gmm.h.
 *
 * It provides \c khg::DiagGmm, the Gaussian mixture model with diagonal
 * covariances that is used as the emission density of every pdf of the
 * acoustic model (see am-diag-gmm.h).  A model of \f$M\f$ components in
 * dimension \f$D\f$ defines the density
 * \f[ p(x) = \sum_{j=1}^{M} w_j \, \mathcal{N}(x; \mu_j,
 *     \mathrm{diag}(\sigma_j^2)), \f]
 * where the mixture weights \f$w_j\f$ are non-negative and should sum to one.
 *
 * As in kaldi, the parameters are kept in exponential (natural) form: the
 * linear weights \f$w_j\f$, the inverse variances \f$\sigma_j^{-2}\f$ and the
 * means times inverse variances \f$\mu_j \odot \sigma_j^{-2}\f$.  This is the
 * representation the statistics accumulators and the update formulas of
 * mle-diag-gmm.h are written against.  For each component a constant
 * "gconst" \f$g_j\f$ collecting the data-independent terms of the
 * log-likelihood is cached (see ComputeGconsts()), so that per-component
 * evaluation of
 * \f[ \ell_j(x) = g_j + (\mu_j \odot \sigma_j^{-2})^{\top} x
 *      - \tfrac{1}{2} (\sigma_j^{-2})^{\top} (x \odot x)
 *      = \log w_j + \log \mathcal{N}(x; \mu_j, \mathrm{diag}(\sigma_j^2)) \f]
 * is a single inner product per component, and the full log-likelihood is
 * \f[ \log p(x) = \log \sum_j w_j \, \mathcal{N}(x; \mu_j,
 *     \mathrm{diag}(\sigma_j^2)) = \mathrm{logsumexp}_j \, \ell_j(x). \f]
 *
 * The API falls into the following groups:
 *
 * - Construction and sizing: the default, copy and sizing constructors,
 *   the constructor from explicit parameters, the merge constructor from
 *   several weighted GMMs, Resize() and CopyFromDiagGmm().
 * - Parameter access and mutation: weights(), means_invvars(), inv_vars(),
 *   gconsts() and valid_gconsts(); SetWeights()/SetMeans()/SetInvVars()/
 *   SetInvVarsAndMeans() for whole matrices, the per-component setters and
 *   getters (SetComponentWeight(), SetComponentMean(), SetComponentInvVar(),
 *   GetComponentMean(), GetComponentVariance()), and GetMeans()/GetVars()
 *   which expose the parameters in normal (mean/variance) form.
 * - Likelihood evaluation: LogLikelihood(), LogLikelihoods(),
 *   LogLikelihoodsMatrix(), LogLikelihoodsPreselect(),
 *   ComponentPosteriors() and ComponentLogLikelihood().
 * - Gaussian selection for pruning: GaussianSelection() (single frame and
 *   matrix versions) and GaussianSelectionPreselect().
 * - Model manipulation: Split()/Merge()/MergeKmeans() change the number of
 *   components; RemoveComponent()/RemoveComponents() drop components;
 *   Interpolate() and Perturb() modify the parameters; Generate() samples a
 *   feature vector from the model.
 * - I/O: Read()/Write() use kaldi's on-disk format (the implementation is
 *   ported from ../kaldi/src/gmm/diag-gmm.cc and the file format is
 *   identical), so models are interchangeable with kaldi.
 *
 * Usage notes:
 *
 * - The weights must be positive (checked by the merge constructor) and
 *   should sum to one for \f$p\f$ to be a properly normalized density; the
 *   variances must be positive.  ComputeGconsts() treats zero weights or
 *   variances as "invalid" components and counts them.
 * - ComputeGconsts() must be called after any parameter change and before
 *   likelihood evaluation or Write().  Every mutator marks the cached
 *   gconsts as stale (\c valid_gconsts() returns false) until the next
 *   ComputeGconsts().
 * - If you need plain means and variances (e.g. for the MLE/MAP update
 *   formulas or for smoothing), use DiagGmmNormal from diag-gmm-normal.h,
 *   a friend class that converts to and from this exponential form.
 */

#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/cluster-utils.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/model-common.h"

namespace khg {

/**
 * \brief Definition for Gaussian Mixture Model with diagonal covariances.
 *
 * Stores \f$M\f$ diagonal-covariance Gaussian components in exponential
 * form: \c weights_ holds the linear mixture weights \f$w_j\f$,
 * \c inv_vars_ holds the inverse variances \f$\sigma_j^{-2}\f$ (one row per
 * component), and \c means_invvars_ holds the means times inverse variances
 * \f$\mu_j \odot \sigma_j^{-2}\f$.  The cached per-component gconsts are
 * \f[ g_j = \log w_j - \frac{1}{2} \Big( D \log 2\pi
 *      + \sum_{d} \log \sigma_{jd}^{2}
 *      + \sum_{d} \frac{\mu_{jd}^{2}}{\sigma_{jd}^{2}} \Big)
 *      = \log w_j + \log \mathcal{N}(0; \mu_j, \mathrm{diag}(\sigma_j^2)), \f]
 * so that \f$\ell_j(x) = g_j + (\mu_j \odot \sigma_j^{-2})^{\top} x
 * - \tfrac{1}{2} (\sigma_j^{-2})^{\top} (x \odot x)\f$.
 */
class DiagGmm {
 public:
  /// this makes it a little easier to modify the internals
  friend class DiagGmmNormal;

  /// Empty constructor.
  DiagGmm() : valid_gconsts_(false) {}

  /// Copy constructor; copies the parameters and the cached gconsts (the
  /// copy keeps the validity state of the source's gconsts).
  explicit DiagGmm(const DiagGmm &gmm) : valid_gconsts_(false) {
    CopyFromDiagGmm(gmm);
  }

  /// Constructor from parameters in exponential form.  Computes the gconsts
  /// immediately.
  /// \param weights       1-D tensor of shape (nmix,); the mixture weights
  ///                      \f$w_j\f$ (linear, not log).
  /// \param inv_vars      2-D tensor of shape (nmix, dim); the inverse
  ///                      variances \f$\sigma_j^{-2}\f$.
  /// \param means_invvars 2-D tensor of shape (nmix, dim); the means times
  ///                      inverse variances \f$\mu_j \odot \sigma_j^{-2}\f$.
  DiagGmm(const FloatVector &weights, const FloatMatrix &inv_vars,
          const FloatMatrix &means_invvars)
      : valid_gconsts_(false),
        weights_(weights),
        inv_vars_(inv_vars),
        means_invvars_(means_invvars) {
    ComputeGconsts();
  }

  DiagGmm &operator=(const DiagGmm &other) = delete;  ///< Disallow assignment.

  /// Resizes arrays to this dim. Does not initialize data.
  /// \param nMix  Number of mixture components.
  /// \param dim   Dimensionality of the feature vectors.
  void Resize(int32_t nMix, int32_t dim);

  /// Returns the number of mixture components in the GMM
  /// \return The number of components \f$M\f$.
  int32_t NumGauss() const { return weights_.size(); }

  /// Returns the dimensionality of the Gaussian mean vectors
  /// \return The feature dimension \f$D\f$.
  int32_t Dim() const { return means_invvars_.cols(); }

  /// Copies from given DiagGmm
  /// \param diaggmm  The model to copy from, including its cached gconsts
  ///                 and their validity state.
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

  /// Constructor that only sizes the model; the parameters are left
  /// uninitialized (see Resize()).
  /// \param nMix  Number of mixture components.
  /// \param dim   Dimensionality of the feature vectors.
  DiagGmm(int32_t nMix, int32_t dim) : valid_gconsts_(false) {
    Resize(nMix, dim);
  }

  /// Constructor that allows us to merge GMMs with weights.  Weights must sum
  /// to one, or this GMM will not be properly normalized (we don't check this).
  /// Weights must be positive (we check this).
  /// \param gmms  Vector of (weight, GMM) pairs; the resulting model contains
  ///              the components of all input GMMs, each scaled by its weight.
  explicit DiagGmm(const std::vector<std::pair<float, const DiagGmm *>> &gmms);

  /// Sets the gconsts.  Returns the number that are "invalid" e.g. because of
  /// zero weights or variances.
  /// \return The number of components whose gconst is \f$\pm\infty\f$.
  int32_t ComputeGconsts();

  /// Returns the log-likelihood of a data point (vector) given the GMM
  /// \f[ \log p(x) = \log \sum_j w_j \, \mathcal{N}(x; \mu_j,
  ///     \mathrm{diag}(\sigma_j^2)). \f]
  /// Requires valid gconsts (ComputeGconsts() must have been called after the
  /// last parameter change).
  /// \param data  A 1-D tensor of shape (dim,)
  /// \return \f$\log p(\mathrm{data})\f$.
  float LogLikelihood(const FloatVector &data) const;

  /// Outputs the per-component log-likelihoods
  /// \f$\ell_j(x) = \log w_j + \log \mathcal{N}(x; \mu_j,
  /// \mathrm{diag}(\sigma_j^2))\f$, i.e. each value includes the mixture
  /// weight of its component.  Requires valid gconsts.
  /// @param data  1-D tensor of shape (dim,)
  /// @param loglikes  On return, 1-D tensor of shape (nmix,)
  void LogLikelihoods(const FloatVector &data, FloatVector *loglikes) const;

  /// Const accessor for the mixture weights.
  /// \return 1-D tensor of shape (nmix,); linear weights (not log).
  const FloatVector &weights() const { return weights_; }

  /// Mutable accessor for the mixture weights.  Note: changing the weights
  /// invalidates the gconsts; call ComputeGconsts() afterwards.
  /// \return Reference to the 1-D weight tensor of shape (nmix,).
  FloatVector &weights() { return weights_; }

  /// Tells whether the cached gconsts are up to date.
  /// \return true if ComputeGconsts() has been called after the last
  ///         parameter change.
  bool valid_gconsts() const { return valid_gconsts_; }

  /// Const accessors
  /// \return 1-D tensor of shape (nmix,) with the per-component gconsts
  ///         \f$g_j\f$ (see the class docs).  Aborts if the gconsts are not
  ///         valid.
  const FloatVector &gconsts() const {
    KHG_ASSERT(valid_gconsts_);
    return gconsts_;
  }

  /// Const accessor for the means times inverse variances.
  /// \return 2-D tensor of shape (nmix, dim), \f$\mu_j \odot
  ///         \sigma_j^{-2}\f$.
  const FloatMatrix &means_invvars() const { return means_invvars_; }

  /// Const accessor for the inverse variances.
  /// \return 2-D tensor of shape (nmix, dim), \f$\sigma_j^{-2}\f$.
  const FloatMatrix &inv_vars() const { return inv_vars_; }

  /// This version of the LogLikelihoods function operates on
  /// a sequence of frames simultaneously; the row index of both "data" and
  /// "loglikes" is the frame index.
  /// @param data 2-D matrix of (num-frames, dim)
  /// @param loglikes On ouput, it contains a  2-D matrix of (num_frames, nmix)
  void LogLikelihoodsMatrix(const FloatMatrix &data,
                            FloatMatrix *loglikes) const;

  /// Outputs the per-component log-likelihoods of a subset of mixture
  /// components.  Note: at output, loglikes->Dim() will equal indices.size().
  /// loglikes[i] will correspond to the log-likelihood of the Gaussian
  /// indexed indices[i], including the mixture weight.
  ///
  /// @param data 1-D tensor of shape (dim,)
  /// @param indices  The indices of the selected mixture components.
  /// @param loglikes  On return, 1-D tensor of shape (indices.size(),)
  void LogLikelihoodsPreselect(const FloatVector &data,
                               const std::vector<int32_t> &indices,
                               FloatVector *loglikes) const;

  /// Get gaussian selection information for one frame.  Returns log-like for
  /// this frame.  Output is the best "num_gselect" indices, sorted from best to
  /// worst likelihood.  If "num_gselect" > NumGauss(), sets it to NumGauss().
  /// The returned value is the total log-likelihood of the *selected*
  /// components (their log-likes combined with LogAdd).
  ///
  /// @param data 1-D tensor of shape (dim,)
  /// @param num_gselect  How many Gaussians to select; clamped to NumGauss().
  /// @param output  On return, the selected component indices, sorted from
  ///                best to worst likelihood.
  /// @return Return the total loglike of the selected gaussian
  float GaussianSelection(const FloatVector &data, int32_t num_gselect,
                          std::vector<int32_t> *output) const;

  /// This version of the Gaussian selection function works for a sequence
  /// of frames rather than just a single frame.  Returns sum of the log-likes
  /// over all frames.
  /// @param data 2-D tensor of shape (num_frames, dim)
  /// @param num_gselect  How many Gaussians to select per frame.
  /// @param output  On return, one vector of selected component indices per
  ///                frame (sorted from best to worst likelihood).
  /// @return The sum over frames of the per-frame selected log-likes.
  float GaussianSelection(const FloatMatrix &data, int32_t num_gselect,
                          std::vector<std::vector<int32_t>> *output) const;

  /// Get gaussian selection information for one frame.  Returns log-like for
  /// this frame.  Output is the best "num_gselect" indices that were
  /// preselected, sorted from best to worst likelihood.  If "num_gselect" >
  /// NumGauss(), sets it to NumGauss().
  ///
  /// @param data 1-D tensor of shape (dim,)
  /// @param preselect  The candidate component indices to select from.
  /// @param num_gselect  How many components to select.
  /// @param output  On return, the selected component indices, sorted from
  ///                best to worst likelihood.
  /// @return The total log-likelihood of the selected components.
  float GaussianSelectionPreselect(const FloatVector &data,
                                   const std::vector<int32_t> &preselect,
                                   int32_t num_gselect,
                                   std::vector<int32_t> *output) const;

  /// Computes the posterior probabilities of all Gaussian components given
  /// a data point. Returns the log-likehood of the data given the GMM.
  /// The posteriors are
  /// \f$ \gamma_j = w_j \mathcal{N}(x; \mu_j, \mathrm{diag}(\sigma_j^2)) /
  /// p(x) \f$.
  ///
  /// @param data 1-D tensor of shape (dim,)
  /// @param posteriors On return, it is a 1-D tensor of shape (nmix,)
  /// @return \f$\log p(\mathrm{data})\f$.
  float ComponentPosteriors(const FloatVector &data,
                            FloatVector *posteriors) const;

  /// Computes the log-likelihood of a data point given a single Gaussian
  /// component. NOTE: Currently we make no guarantees about what happens if
  /// one of the variances is zero.
  /// @param data 1-D tensor of shape (dim,)
  /// @param comp_id  Index of the mixture component (0-based).
  /// @return \f$\log w_{\mathrm{comp\_id}} +
  ///         \log \mathcal{N}(\mathrm{data}; \mu, \mathrm{diag}(\sigma^2))\f$.
  float ComponentLogLikelihood(const FloatVector &data, int32_t comp_id) const;

  /// Generates a random data-point from this distribution.
  /// @param output 1-D tensor of shape (dim,). Must be pre-allocated
  void Generate(FloatVector *output) const;

  /// Perturbs the component means with a random vector multiplied by the
  /// perturb factor.
  /// \param perturb_factor  Scale of the random perturbation.
  void Perturb(float perturb_factor);

  /// this = rho x source + (1-rho) x this
  /// \param rho  Interpolation weight of \p source, in \f$[0, 1]\f$.
  /// \param source  The model to interpolate with; must have the same number
  ///                of components and the same dimension.
  /// \param flags  Which parameter groups to interpolate (default: all).
  ///               Weights are re-normalized after interpolation.
  void Interpolate(float rho, const DiagGmm &source,
                   GmmFlagsType flags = kGmmAll);

  /// Merge the components to a specified target \c components: this
  /// version uses a different approach based on K-means.
  /// \param target_components  Desired number of components after merging.
  /// \param cfg  K-means configuration.
  void MergeKmeans(int32_t target_components,
                   const ClusterKMeansOptions &cfg = ClusterKMeansOptions());

  /// Merge the components and remember the order in which the components were
  /// merged (flat list of pairs)
  /// \param target_components  Desired number of components after merging.
  /// \param history  If non-NULL, on return holds the merge history as a flat
  ///                 list of pairs.
  void Merge(int32_t target_components,
             std::vector<int32_t> *history = nullptr);

  /// Split the components and remember the order in which the components were
  /// split
  /// \param target_components  Desired number of components after splitting.
  /// \param perturb_factor  Scale of the random perturbation used to
  ///                        differentiate the two halves of a split component.
  /// \param history  If non-NULL, on return holds the split history as a flat
  ///                 list of pairs.
  void Split(int32_t target_components, float perturb_factor,
             std::vector<int32_t> *history = nullptr);

  /// Removes single component from model
  /// \param gauss  Index of the component to remove.
  /// \param renorm_weights  If true, re-normalize the remaining weights to
  ///                        sum to one.
  void RemoveComponent(int32_t gauss, bool renorm_weights);

  /// Removes multiple components from model; "gauss" must not have dups.
  /// \param gauss  Indices of the components to remove (no duplicates).
  /// \param renorm_weights  If true, re-normalize the remaining weights to
  ///                        sum to one.
  void RemoveComponents(const std::vector<int32_t> &gauss, bool renorm_weights);

  /// Set mixture weights.
  /// \param w  1-D tensor of shape (num_mix,)
  void SetWeights(const FloatVector &w);

  /// Use SetMeans to update only the Gaussian means (and not variances)
  /// \param m  2-D tensor of shape (num_mix, dim)
  void SetMeans(const FloatMatrix &m);

  /// Accessor for means.
  /// \return 2-D tensor of shape (num_mix, dim) with the plain means
  ///         \f$\mu_j\f$ (computed from means_invvars_ / inv_vars_).
  FloatMatrix GetMeans() const;

  /// Set the (inverse) variances and recompute means_invvars_
  /// \param v  2-D matrix of shape (num_mix, dim) with the inverse variances
  ///           \f$\sigma_j^{-2}\f$.
  void SetInvVars(const FloatMatrix &v);

  /// Accessor for covariances.
  /// \return 2-D tensor of shape (num_mix, dim) with the diagonal variances
  ///         \f$\sigma_j^2\f$.
  FloatMatrix GetVars() const;

  /// Set weight for single component.
  /// \param gauss  Component index.
  /// \param weight  The new (linear) mixture weight.
  void SetComponentWeight(int32_t gauss, float weight);

  /// Mutators for single component, supports float or double
  /// Set mean for a single component - internally multiplies with inv(var)
  /// \param gauss  Component index.
  /// \param in  1-D tensor of shape (dim,) with the plain mean.
  void SetComponentMean(int32_t gauss, const FloatVector &in);

  /// Use SetInvVarsAndMeans if updating both means and (inverse) variances
  /// \param invvars  2-D tensor of shape (num_mix, dim), inverse variances.
  /// \param means    2-D tensor of shape (num_mix, dim), plain means.
  void SetInvVarsAndMeans(const FloatMatrix &invvars, const FloatMatrix &means);

  /// Set inv-var for single component (recommend to do this before
  /// setting the mean, if doing both, for numerical reasons).
  /// \param gauss  Component index.
  /// \param in  1-D tensor of shape (dim,) with the inverse variances.
  void SetComponentInvVar(int32_t gauss, const FloatVector &in);

  /// Accessor for single component mean
  /// \param gauss  Component index.
  /// \return 1-D tensor of shape (dim,) with the plain mean.
  FloatVector GetComponentMean(int32_t gauss) const;

  /// Accessor for single component variance.
  /// \param gauss  Component index.
  /// \return 1-D tensor of shape (dim,) with the diagonal variance.
  FloatVector GetComponentVariance(int32_t gauss) const;

  /// Reads from a kaldi-format stream (ported from
  /// ../kaldi/src/gmm/diag-gmm.cc; file format is identical).
  /// \param is  The input stream to read from.
  /// \param binary  true for binary mode, false for text mode.
  void Read(std::istream &is, bool binary);

  /// Writes to a kaldi-format stream (ported from
  /// ../kaldi/src/gmm/diag-gmm.cc; file format is identical).
  /// \param os  The output stream to write to.
  /// \param binary  true for binary mode, false for text mode.
  void Write(std::ostream &os, bool binary) const;

 private:
  /// MergedComponentsLogdet computes logdet for merged components
  /// f1, f2 are first-order stats (normalized by zero-order stats)
  /// s1, s2 are second-order stats (normalized by zero-order stats)
  /// \param w1  Weight of the first component.
  /// \param w2  Weight of the second component.
  /// \param f1  1-D tensor; first-order stats of the first component.
  /// \param f2  1-D tensor; first-order stats of the second component.
  /// \param s1  1-D tensor; second-order stats of the first component.
  /// \param s2  1-D tensor; second-order stats of the second component.
  /// \return The log-determinant of the merged component.
  float MergedComponentsLogdet(float w1, float w2,
                               const FloatVector &f1,         // 1-D
                               const FloatVector &f2,         // 1-D
                               const FloatVector &s1,         // 1-D
                               const FloatVector &s2) const;  // 1-D

 private:
  /// Equals log(weight) - 0.5 * (log(2pi) + log det(var) + mean*mean*inv(var))
  FloatVector gconsts_;  // 1-d tensor, (nimx,)

  bool valid_gconsts_;  ///< Recompute gconsts_ if false

  /// 1-D, (nmix,) weights (not log)., kFloat
  FloatVector weights_;

  /// 2-D, (nmix, dim), Inverted (diagonal) variances, kFloat
  FloatMatrix inv_vars_;

  /// 2-D, (nmix, dim), Means times inverted variance, kFloat
  FloatMatrix means_invvars_;
};

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_DIAG_GMM_H_
