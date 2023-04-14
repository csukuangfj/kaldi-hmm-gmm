#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_diag_gmm_py


import math
import unittest

import torch

import kaldi_hmm_gmm as khg


class TestDiagGmm(unittest.TestCase):
    def test(self):
        nmix = 10
        dim = 20

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = torch.rand(nmix, dtype=torch.float32)
        weights /= weights.sum()

        mean = torch.rand(nmix, dim)
        var = torch.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_inv_vars(1 / var)

        assert torch.allclose(diag_gmm.weights, weights)
        assert torch.allclose(diag_gmm.means, mean)
        assert torch.allclose(diag_gmm.vars, var)

        assert diag_gmm.num_gauss == nmix, diag_gmm.num_gauss
        assert diag_gmm.dim == dim, diag_gmm.dim

        assert diag_gmm.valid_gconsts is False
        num_bad = 0
        assert diag_gmm.compute_gconsts() == num_bad
        assert diag_gmm.valid_gconsts is True

        expected_gconsts = weights.log() - 0.5 * (
            diag_gmm.dim * math.log(2 * math.pi)
            + var.log().sum(dim=1)
            + mean.square().div(var).sum(dim=1)
        )

        assert torch.allclose(diag_gmm.means_invvars, mean.div(var))
        assert torch.allclose(diag_gmm.inv_vars, 1 / var)

        assert torch.allclose(diag_gmm.gconsts, expected_gconsts)
        for i in range(nmix):
            assert torch.allclose(diag_gmm.get_component_mean(i), mean[i])
            assert torch.allclose(diag_gmm.get_component_variance(i), var[i])

        diag_gmm.set_component_weight(0, 0.2)
        diag_gmm.set_component_weight(1, 0.8)
        assert diag_gmm.valid_gconsts is False
        assert diag_gmm.compute_gconsts() == num_bad


if __name__ == "__main__":
    torch.manual_seed(20230414)
    unittest.main()
