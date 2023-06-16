// kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h
//
// Copyright 2009-2012  Saarland University (author: Arnab Ghoshal);
//                      Yanmin Qian; Johns Hopkins University (author: Daniel
//                      Povey) Cisco Systems (author: Neha Agrawal)
//                2023  Xiaomi Corporation
#ifndef KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_
#define KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_
namespace khg {

class AccumAmDiagGmm {
 public:
  AccumAmDiagGmm() : total_frames_(0.0), total_log_like_(0.0) {}
  ~AccumAmDiagGmm();
  AccumAmDiagGmm(const AccumAmDiagGmm &) = delete;
  AccumAmDiagGmm &operator=(const AccumAmDiagGmm &) = delete;

 private:
  /// MLE accumulators and update methods for the GMMs
  std::vector<AccumDiagGmm *> gmm_accumulators_;

  /// Total counts & likelihood (for diagnostics)
  double total_frames_, total_log_like_;
};

}  // namespace khg
#endif  // KALDI_HMM_GMM_CSRC_MLE_AM_DIAG_GMM_H_
