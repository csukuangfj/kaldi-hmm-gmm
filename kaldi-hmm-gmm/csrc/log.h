// kaldi-hmm-gmm/csrc/log.h
//
// Copyright (c)  2022  Xiaomi Corporation

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

#define KHG_PARANOID_ASSERT KHG_ASSERT

#define KHG_DISALLOW_COPY_AND_ASSIGN(Class) \
 public:                                    \
  Class(const Class &) = delete;            \
  Class &operator=(const Class &) = delete;

}  // namespace khg

#endif  // KALDI_HMM_GMM_CSRC_LOG_H_
