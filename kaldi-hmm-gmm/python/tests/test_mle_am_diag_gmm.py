#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_mle_am_diag_gmm_py
import unittest
import torch
import kaldi_hmm_gmm as khg


def init_rand_diag_gmm(dim: int, num_comp: int):
    weights = torch.rand(num_comp, dtype=torch.float32)

    weights /= weights.sum()

    means = torch.rand(num_comp, dim)
    variances = torch.rand(num_comp, dim)

    diag_gmm = khg.DiagGmm(nmix=num_comp, dim=dim)

    diag_gmm.set_weights(weights)
    diag_gmm.set_means(means)
    diag_gmm.set_invvars(1 / variances)

    diag_gmm.compute_gconsts()

    return diag_gmm


class TestAccumAmDiagGmm(unittest.TestCase):
    def test(self):
        dim = torch.randint(low=2, high=10, size=(1,)).item()
        num_pdfs = torch.randint(low=3, high=15, size=(1,)).item()

        am = khg.AmDiagGmm()
        total_num_comp = 0
        for i in range(num_pdfs):
            num_comp = torch.randint(low=1, high=8, size=(1,)).item()
            total_num_comp += num_comp
            gmm = init_rand_diag_gmm(dim=dim, num_comp=num_comp)
            am.add_pdf(gmm)
        assert am.num_pdfs == num_pdfs, (am.num_pdfs, num_pdfs)
        assert am.dim == dim, (am.dim, dim)
        print(total_num_comp, num_pdfs, dim)


if __name__ == "__main__":
    torch.manual_seed(20230617)
    unittest.main()
