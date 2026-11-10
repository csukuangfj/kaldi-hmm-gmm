// kaldi-hmm-gmm/csrc/log.h
//
// Copyright 2019 LAIX (Yi Sun)
// Copyright 2019 SmartAction LLC (kkm)
// Copyright 2016 Brno University of Technology (author: Karel Vesely)
// Copyright 2009-2011  Microsoft Corporation;  Ondrej Glembek;  Lukas Burget;
//                      Saarland University
// Copyright (c)  2022  Xiaomi Corporation
//
// This file is copied/modified from kaldi/src/base/kaldi-error.h
//

/**
 * \file log.h
 * \brief Logging and assertion macros used throughout the library.
 *
 * This file is copied/modified from kaldi/src/base/kaldi-error.h.
 *
 * It provides the \c Logger stream-style logging class and the macros
 * \c KHG_LOG (info), \c KHG_WARN (warning) and \c KHG_ERR (error; throws
 * std::runtime_error when the message is flushed), the assertion macros
 * \c KHG_ASSERT and \c KHG_PARANOID_ASSERT (the latter active only when
 * \c KHG_PARANOID is defined), and the convenience macro
 * \c KHG_DISALLOW_COPY_AND_ASSIGN.  Usage is like kaldi's
 * \c KALDI_LOG / \c KALDI_WARN / \c KALDI_ERR, e.g.
 * <tt>KHG_ASSERT(x); KHG_ERR &lt;&lt; "bad value: " &lt;&lt; v;</tt>.
 */

#ifndef KALDI_HMM_GMM_CSRC_LOG_H_
#define KALDI_HMM_GMM_CSRC_LOG_H_

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace khg {

enum class LogLevel {
  kInfo = 0,
  kWarn = 1,
  kError = 2,  // abort the program
};

class Logger {
 public:
  Logger(const char *filename, const char *func_name, uint32_t line_num,
         LogLevel level)
      : level_(level) {
    os_ << filename << ":" << func_name << ":" << line_num << "\n";
    switch (level_) {
      case LogLevel::kInfo:
        os_ << "[I] ";
        break;
      case LogLevel::kWarn:
        os_ << "[W] ";
        break;
      case LogLevel::kError:
        os_ << "[E] ";
        break;
    }
  }

  template <typename T>
  Logger &operator<<(const T &val) {
    os_ << val;
    return *this;
  }

  ~Logger() noexcept(false) {
    if (level_ == LogLevel::kError) {
      // throw std::runtime_error(os_.str());
      // abort();
      throw std::runtime_error(os_.str());
    }
    // fprintf(stderr, "%s\n", os_.str().c_str());
  }

 private:
  std::ostringstream os_;
  LogLevel level_;
};

class Voidifier {
 public:
  void operator&(const Logger &) const {}
};

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__PRETTY_FUNCTION__)
// for clang and GCC
#define KHG_FUNC __PRETTY_FUNCTION__
#else
// for other compilers
#define KHG_FUNC __func__
#endif

#define KHG_LOG khg::Logger(__FILE__, KHG_FUNC, __LINE__, khg::LogLevel::kInfo)

#define KHG_WARN khg::Logger(__FILE__, KHG_FUNC, __LINE__, khg::LogLevel::kWarn)

#define KHG_ERR khg::Logger(__FILE__, KHG_FUNC, __LINE__, khg::LogLevel::kError)

#define KHG_ASSERT(x)                                   \
  (x) ? (void)0                                         \
      : khg::Voidifier() & KHG_ERR << "Check failed!\n" \
                                   << "x: " << #x

#ifdef KHG_PARANOID
#define KHG_PARANOID_ASSERT(x) KHG_ASSERT(x)
#else
#define KHG_PARANOID_ASSERT(x) (void)0
#endif

#define KHG_DISALLOW_COPY_AND_ASSIGN(Class) \
 public:                                    \
  Class(const Class &) = delete;            \
  Class &operator=(const Class &) = delete;

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_LOG_H_
