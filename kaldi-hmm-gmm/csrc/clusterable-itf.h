// kaldi-hmm-gmm/csrc/clusterable-itf.h
//
// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CLUSTERABLE_ITF_H_
#define KALDI_HMM_GMM_CSRC_CLUSTERABLE_ITF_H_
#include <string>

#include "kaldi-hmm-gmm/csrc/log.h"

/**
 * \file clusterable-itf.h
 * \brief Abstract interface for statistics that can be clustered.
 *
 * This file is copied/modified from kaldi/src/itf/clusterable-itf.h.
 *
 * It declares the abstract class \c Clusterable: a set of statistics (e.g.
 * Gaussian sufficient statistics) that can be added, subtracted and scaled,
 * and that reports an objective function \f$\mathrm{Objf}\f$ and a
 * normalizer (typically a count).  Concrete implementations live in
 * clusterable-classes.h (\c ScalarClusterable, \c GaussClusterable); the
 * clustering algorithms that consume this interface live in cluster-utils.h.
 * See kaldi's \c clustering documentation for an explanation of its function.
 */

namespace khg {

/** \addtogroup clustering_group
 @{
  A virtual class for clusterable objects; see kaldi's \c clustering
  documentation for an explanation of its function.
*/

/**
 * \brief A virtual class for clusterable objects.
 *
 * An instance represents sufficient statistics for one cluster.  The
 * operations \c Add, \c Sub and \c Scale update the statistics in place;
 * \c Objf reports the objective function (to be maximized) of fitting the
 * statistics with a single cluster, and \c ObjfPlus, \c ObjfMinus and
 * \c Distance give the objective-function changes implied by merging or
 * splitting clusters.
 */
class Clusterable {
 public:
  /// \name Functions that must be overridden
  /// @{
  virtual ~Clusterable() = default;

  /// Return a copy of this object.
  virtual Clusterable *Copy() const = 0;
  /**
   * \brief Return the objective function associated with the stats.
   *
   * Return the objective function associated with the stats [assuming ML
   * estimation].  The higher, the better; a larger value means the single
   * cluster explains the accumulated statistics better.
   *
   * \return The objective function of this cluster.
   */
  virtual float Objf() const = 0;
  /// Return the normalizer (typically, count) associated with the stats
  virtual float Normalizer() const = 0;
  /// Set stats to empty.
  virtual void SetZero() = 0;
  /**
   * \brief Add other stats.
   *
   * \param other  Stats to add to this object; must be of the same derived
   *               type.
   */
  virtual void Add(const Clusterable &other) = 0;
  /**
   * \brief Subtract other stats.
   *
   * \param other  Stats to subtract from this object; must be of the same
   *               derived type.
   */
  virtual void Sub(const Clusterable &other) = 0;
  /**
   * \brief Scale the stats by a positive number f [not mandatory to supply
   *        this].
   *
   * \param f  The (positive) scale factor.
   */
  virtual void Scale(float f) {
    KHG_ERR << "This Clusterable object does not implement Scale().";
  }

  /// Return a string that describes the inherited type.
  virtual std::string Type() const = 0;

  /// @}

  /// \name Functions that have default implementations
  /// @{

  // These functions have default implementations (but may be overridden for
  // speed). Implementations in clusterable-classes.cc

  /**
   * \brief Return the objective function of the combined object this + other.
   *
   * \param other  The stats to combine with this object.
   * \return \f$\mathrm{Objf}(\mathrm{this} + \mathrm{other})\f$.
   */
  virtual float ObjfPlus(const Clusterable &other) const;
  /**
   * \brief Return the objective function of the subtracted object this -
   *        other.
   *
   * \param other  The stats to subtract from this object.
   * \return \f$\mathrm{Objf}(\mathrm{this} - \mathrm{other})\f$.
   */
  virtual float ObjfMinus(const Clusterable &other) const;
  /**
   * \brief Return the objective function decrease from merging the two
   *        clusters, negated to be a positive number (or zero).
   *
   * \param other  The stats of the other cluster.
   * \return The objective-function loss
   *         \f$\max(0,\ \mathrm{Objf}(\mathrm{this}) +
   *         \mathrm{Objf}(\mathrm{other}) -
   *         \mathrm{Objf}(\mathrm{this} + \mathrm{other}))\f$
   *         from merging the two clusters; zero if merging does not hurt.
   */
  virtual float Distance(const Clusterable &other) const;
  /// @}
};
/// @} end of "ingroup clustering_group"
}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTERABLE_ITF_H_
