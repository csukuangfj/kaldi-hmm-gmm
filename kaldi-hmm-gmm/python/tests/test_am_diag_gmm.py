#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_am_diag_gmm_py
import unittest
import torch
import kaldi_hmm_gmm as khg


class TestAmDiagGmm(unittest.TestCase):
    def test(self):
        nmix = 10
        dim = 8

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)

        diag_gmm.compute_gconsts()

        am = khg.AmDiagGmm()
        am.add_pdf(diag_gmm)
        assert am.num_pdfs == 1
        assert am.num_gauss == nmix

        am.add_pdf(diag_gmm)
        assert am.num_pdfs == 2
        assert am.num_gauss == nmix * 2
        am.num_gauss_in_pdf(0) == nmix
        am.num_gauss_in_pdf(1) == nmix

        am.split_pdf(pdf_idx=1, target_components=nmix * 5, perturb_factor=0.01)
        assert am.num_gauss_in_pdf(1) == nmix * 5
        assert am.num_gauss == nmix + nmix * 5


if __name__ == "__main__":
    torch.manual_seed(20230416)
    unittest.main()
