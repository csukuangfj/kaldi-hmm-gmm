// kaldi-hmm-gmm/csrc/eigen.h

// Copyright (c)  2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_EIGEN_H_
#define KALDI_HMM_GMM_CSRC_EIGEN_H_

#include "Eigen/Dense"

/**
 * \file eigen.h
 * \brief Eigen typedefs matching kaldi's Matrix layout, plus small vector
 *        math and random helpers.
 *
 * This file defines the fixed-layout matrix/vector typedefs used throughout
 * the library: dynamic-size matrices are row-major (like kaldi's
 * \c Matrix<double>/<tt>Matrix<float></tt>) and vectors are column vectors.
 * It also provides a few small utilities: \c LogSumExp and \c Softmax over
 * \c FloatVector, and normally distributed random number generators
 * (\c Randn, \c RandnVector, \c RandnMatrix) analogous to the helpers in
 * kaldi/src/base/kaldi-math.h.
 */

namespace khg {
using FloatMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using DoubleMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using FloatVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using FloatRowVector = Eigen::Matrix<float, 1, Eigen::Dynamic>;

using DoubleRowVector = Eigen::Matrix<double, 1, Eigen::Dynamic>;

/// Returns \f$\log \sum_i \exp(v_i)\f$; entries below \c prune are dropped.
float LogSumExp(const FloatVector &v, float prune = -1.0f);

/// Returns the softmax of \c v; optionally outputs the log-sum-exp.
FloatVector Softmax(const FloatVector &v, float *log_sum_exp = nullptr);

// A vector of normal distribution
/// Returns a vector of \c n samples from \f$\mathcal{N}(\mathrm{mean},
/// \mathrm{stddev}^2)\f$.
FloatVector RandnVector(int32_t n, float mean = 0, float stddev = 1);

/// Returns a rows-by-cols matrix of samples from
/// \f$\mathcal{N}(\mathrm{mean}, \mathrm{stddev}^2)\f$.
FloatMatrix RandnMatrix(int32_t rows, int32_t cols, float mean = 0,
                        float stddev = 1);

/// Returns one sample from \f$\mathcal{N}(\mathrm{mean},
/// \mathrm{stddev}^2)\f$.
float Randn(float mean = 0, float stddev = 1);

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_EIGEN_H_
