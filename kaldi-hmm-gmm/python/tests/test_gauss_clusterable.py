#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_gauss_clusterable_py

import unittest

import kaldi_hmm_gmm as khg
import torch


class TestGaussClusterable(unittest.TestCase):
    def test(self):
        a = khg.GaussClusterable()
        assert a.count() == 0

    def test2(self):
        mean = torch.rand(3)
        var = torch.rand(3)
        a = khg.GaussClusterable(mean, var, var_floor=1e-10, count=0.5)
        print(var.dtype)

        mean2 = torch.rand(3)

        w = 10
        a.add_stats(mean2, weight=w)

        assert a.count() == 0.50 + w, a.count()

        assert torch.allclose(
            torch.from_numpy(a.x_stats()), (mean + mean2 * w).to(torch.double)
        ), (
            a.x_stats(),
            mean + mean2 * w,
        )
        assert torch.allclose(
            torch.from_numpy(a.x2_stats()), (var + mean2.square() * w).to(torch.double)
        ), (
            a.x2_stats(),
            (var + mean2.square() * w).to(torch.double),
        )


if __name__ == "__main__":
    unittest.main()
