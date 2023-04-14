// kaldi-hmm-gmm/csrc/clusterable-itf.h
//
// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_CLUSTERABLE_ITF_H_
#define KALDI_HMM_GMM_CSRC_CLUSTERABLE_ITF_H_
#include <string>

#include "kaldi-hmm-gmm/csrc/log.h"

namespace khg {

/** \addtogroup clustering_group
 @{
  A virtual class for clusterable objects; see \ref clustering for an
  explanation if its function.
*/

class Clusterable {
 public:
  /// \name Functions that must be overridden
  /// @{
  virtual ~Clusterable() = default;

  /// Return a copy of this object.
  virtual Clusterable *Copy() const = 0;
  /// Return the objective function associated with the stats
  /// [assuming ML estimation]
  virtual float Objf() const = 0;
  /// Return the normalizer (typically, count) associated with the stats
  virtual float Normalizer() const = 0;
  /// Set stats to empty.
  virtual void SetZero() = 0;
  /// Add other stats.
  virtual void Add(const Clusterable &other) = 0;
  /// Subtract other stats.
  virtual void Sub(const Clusterable &other) = 0;
  /// Scale the stats by a positive number f [not mandatory to supply this].
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

  /// Return the objective function of the combined object this + other.
  virtual float ObjfPlus(const Clusterable &other) const;
  /// Return the objective function of the subtracted object this - other.
  virtual float ObjfMinus(const Clusterable &other) const;
  /// Return the objective function decrease from merging the two
  /// clusters, negated to be a positive number (or zero).
  virtual float Distance(const Clusterable &other) const;
  /// @}
};
/// @} end of "ingroup clustering_group"
}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_CLUSTERABLE_ITF_H_
