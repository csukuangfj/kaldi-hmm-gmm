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
// this if is copied and modified from
// kaldi/src/gmm/diag-gmm.h

#include <utility>
#include <vector>

#include "kaldi-hmm-gmm/csrc/cluster-utils.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/model-common.h"

namespace khg {

/// Definition for Gaussian Mixture Model with diagonal covariances
class DiagGmm {
 public:
  /// this makes it a little easier to modify the internals
  friend class DiagGmmNormal;

  /// Empty constructor.
  DiagGmm() : valid_gconsts_(false) {}

  explicit DiagGmm(const DiagGmm &gmm) : valid_gconsts_(false) {
    CopyFromDiagGmm(gmm);
  }

  DiagGmm(const FloatVector &weights, const FloatMatrix &inv_vars,
          const FloatMatrix &means_invvars)
      : valid_gconsts_(false),
        weights_(weights),
        inv_vars_(inv_vars),
        means_invvars_(means_invvars) {
    ComputeGconsts();
  }

  DiagGmm &operator=(const DiagGmm &other) = delete;  // Disallow assignment

  /// Resizes arrays to this dim. Does not initialize data.
  void Resize(int32_t nMix, int32_t dim);

  /// Returns the number of mixture components in the GMM
  int32_t NumGauss() const { return weights_.size(); }

  /// Returns the dimensionality of the Gaussian mean vectors
  int32_t Dim() const { return means_invvars_.cols(); }

  /// Copies from given DiagGmm
  void CopyFromDiagGmm(const DiagGmm &diaggmm);

  DiagGmm(int32_t nMix, int32_t dim) : valid_gconsts_(false) {
    Resize(nMix, dim);
  }

  /// Constructor that allows us to merge GMMs with weights.  Weights must sum
  /// to one, or this GMM will not be properly normalized (we don't check this).
  /// Weights must be positive (we check this).
  explicit DiagGmm(const std::vector<std::pair<float, const DiagGmm *>> &gmms);

  /// Sets the gconsts.  Returns the number that are "invalid" e.g. because of
  /// zero weights or variances.
  int32_t ComputeGconsts();

  /// Returns the log-likelihood of a data point (vector) given the GMM
  // @param data  A 1-D tensor
  float LogLikelihood(const FloatVector &data) const;

  /// Outputs the per-component log-likelihoods
  /// @param data  1-D tensor.
  /// @param loglikes  1-D tensor.
  void LogLikelihoods(const FloatVector &data, FloatVector *loglikes) const;

  const FloatVector &weights() const { return weights_; }
  FloatVector &weights() { return weights_; }

  bool valid_gconsts() const { return valid_gconsts_; }

  /// Const accessors
  const FloatVector &gconsts() const {
    KHG_ASSERT(valid_gconsts_);
    return gconsts_;
  }

  const FloatMatrix &means_invvars() const { return means_invvars_; }
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
  /// @param indices
  /// @param loglikes 1-D tensor of shape (indices.size(),)
  void LogLikelihoodsPreselect(const FloatVector &data,
                               const std::vector<int32_t> &indices,
                               FloatVector *loglikes) const;

  /// Get gaussian selection information for one frame.  Returns log-like for
  /// this frame.  Output is the best "num_gselect" indices, sorted from best to
  /// worst likelihood.  If "num_gselect" > NumGauss(), sets it to NumGauss().
  ///
  /// @param data 1-D tensor of shape (dim,)
  /// @param num_gselect
  /// @param output
  /// @return Return the total loglike of the selected gaussian
  float GaussianSelection(const FloatVector &data, int32_t num_gselect,
                          std::vector<int32_t> *output) const;

  /// This version of the Gaussian selection function works for a sequence
  /// of frames rather than just a single frame.  Returns sum of the log-likes
  /// over all frames.
  /// @param data 2-D tensor of shape (num_frames, dim)
  /// @param num_gselect
  /// @param output
  ///
  float GaussianSelection(const FloatMatrix &data, int32_t num_gselect,
                          std::vector<std::vector<int32_t>> *output) const;

  /// Get gaussian selection information for one frame.  Returns log-like for
  /// this frame.  Output is the best "num_gselect" indices that were
  /// preselected, sorted from best to worst likelihood.  If "num_gselect" >
  /// NumGauss(), sets it to NumGauss().
  ///
  /// @param data 1-D tensor of shape (dim,)
  float GaussianSelectionPreselect(const FloatVector &data,
                                   const std::vector<int32_t> &preselect,
                                   int32_t num_gselect,
                                   std::vector<int32_t> *output) const;

  /// Computes the posterior probabilities of all Gaussian components given
  /// a data point. Returns the log-likehood of the data given the GMM.
  ///
  /// @param data 1-D tensor of shape (dim,)
  /// @param posteriors On return, it is a 1-D tensor of shape (nmix,)
  float ComponentPosteriors(const FloatVector &data,
                            FloatVector *posteriors) const;

  /// Computes the log-likelihood of a data point given a single Gaussian
  /// component. NOTE: Currently we make no guarantees about what happens if
  /// one of the variances is zero.
  /// @param data 1-D tensor of shape (dim,)
  float ComponentLogLikelihood(const FloatVector &data, int32_t comp_id) const;

  /// Generates a random data-point from this distribution.
  /// @param output 1-D tensor of shape (dim,). Must be pre-allocated
  void Generate(FloatVector *output) const;

  /// Perturbs the component means with a random vector multiplied by the
  /// perturb factor.
  void Perturb(float perturb_factor);

  /// this = rho x source + (1-rho) x this
  void Interpolate(float rho, const DiagGmm &source,
                   GmmFlagsType flags = kGmmAll);

  // Merge the components to a specified target #components: this
  // version uses a different approach based on K-means.
  void MergeKmeans(int32_t target_components,
                   const ClusterKMeansOptions &cfg = ClusterKMeansOptions());

  /// Merge the components and remember the order in which the components were
  /// merged (flat list of pairs)
  void Merge(int32_t target_components,
             std::vector<int32_t> *history = nullptr);

  /// Split the components and remember the order in which the components were
  /// split
  void Split(int32_t target_components, float perturb_factor,
             std::vector<int32_t> *history = nullptr);

  /// Removes single component from model
  void RemoveComponent(int32_t gauss, bool renorm_weights);

  /// Removes multiple components from model; "gauss" must not have dups.
  void RemoveComponents(const std::vector<int32_t> &gauss, bool renorm_weights);

  // w is a 1-D tensor of shape (num_mix,)
  void SetWeights(const FloatVector &w);  ///< Set mixture weights
                                          ///
  /// Use SetMeans to update only the Gaussian means (and not variances)
  /// m is a 2-D tensor of shape (num_mix, dim)
  void SetMeans(const FloatMatrix &m);
  /// Accessor for means.
  FloatMatrix GetMeans() const;

  /// Set the (inverse) variances and recompute means_invvars_
  void SetInvVars(const FloatMatrix &v);  // v is a 2-D matrix
                                          //
  /// Accessor for covariances.
  FloatMatrix GetVars() const;

  /// Set weight for single component.
  void SetComponentWeight(int32_t gauss, float weight);

  /// Mutators for single component, supports float or double
  /// Set mean for a single component - internally multiplies with inv(var)
  /// in is a 1-D tensor
  void SetComponentMean(int32_t gauss, const FloatVector &in);

  /// Use SetInvVarsAndMeans if updating both means and (inverse) variances
  void SetInvVarsAndMeans(const FloatMatrix &invvars, const FloatMatrix &means);

  /// Set inv-var for single component (recommend to do this before
  /// setting the mean, if doing both, for numerical reasons).
  /// in is a 1-D tensor
  void SetComponentInvVar(int32_t gauss, const FloatVector &in);

  /// Accessor for single component mean
  /// Return a 1-D tensor
  FloatVector GetComponentMean(int32_t gauss) const;

  /// Accessor for single component variance.
  /// Return a 1-D tensor
  FloatVector GetComponentVariance(int32_t gauss) const;

 private:
  // MergedComponentsLogdet computes logdet for merged components
  // f1, f2 are first-order stats (normalized by zero-order stats)
  // s1, s2 are second-order stats (normalized by zero-order stats)
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
