// kaldi-hmm-gmm/csrc/kaldi-math.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Yanmin Qian;
//                      Jan Silovsky;  Saarland University
//                2023  Xiaomi Corporation

// this file is copied and modified from
// kaldi/src/base/kaldi-math.h
#ifndef KALDI_HMM_GMM_CSRC_KALDI_MATH_H_
#define KALDI_HMM_GMM_CSRC_KALDI_MATH_H_

#include <cmath>

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-7f
#endif

// M_LOG_2PI =  log(2*pi)
#ifndef M_LOG_2PI
#define M_LOG_2PI 1.8378770664093454835606594728112
#endif

#define KALDI_ISINF std::isinf
#define KALDI_ISNAN std::isnan

namespace khg {

static const float kMinLogDiffFloat = std::log(FLT_EPSILON);  // negative!

#if !defined(_MSC_VER) || (_MSC_VER >= 1700)
inline double Log1p(double x) { return std::log1p(x); }
inline float Log1p(float x) { return std::log1pf(x); }
#else
inline double Log1p(double x) {
  const double cutoff = 1.0e-08;
  if (x < cutoff)
    return x - 0.5 * x * x;
  else
    return std::log(1.0 + x);
}

inline float Log1p(float x) {
  const float cutoff = 1.0e-07;
  if (x < cutoff)
    return x - 0.5 * x * x;
  else
    return std::log(1.0 + x);
}
#endif

// returns log(exp(x) + exp(y)).
inline float LogAdd(float x, float y) {
  float diff;

  if (x < y) {
    diff = x - y;
    x = y;
  } else {
    diff = y - x;
  }
  // diff is negative.  x is now the larger one.

  if (diff >= kMinLogDiffFloat) {
    float res;
    res = x + Log1p(std::exp(diff));
    return res;
  } else {
    return x;  // return the larger one.
  }
}

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_KALDI_MATH_H_
