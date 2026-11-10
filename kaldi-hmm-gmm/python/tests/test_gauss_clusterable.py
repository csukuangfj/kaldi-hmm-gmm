#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_gauss_clusterable_py

import unittest

import kaldi_hmm_gmm as khg
import numpy as np


class TestGaussClusterable(unittest.TestCase):
    def test(self):
        a = khg.GaussClusterable()
        assert a.count() == 0

    def test2(self):
        mean = np.random.rand(3)
        var = np.random.rand(3)
        a = khg.GaussClusterable(mean, var, var_floor=1e-10, count=0.5)
        print(var.dtype)

        mean2 = np.random.rand(3)

        w = 10
        a.add_stats(mean2, weight=w)

        assert a.count() == 0.50 + w, a.count()

        assert np.allclose(
            np.asarray(a.x_stats()), (mean + mean2 * w).astype(np.float64)
        ), (
            a.x_stats(),
            mean + mean2 * w,
        )
        assert np.allclose(
            np.asarray(a.x2_stats()), (var + mean2 ** 2 * w).astype(np.float64)
        ), (
            a.x2_stats(),
            (var + mean2 ** 2 * w).astype(np.float64),
        )


if __name__ == "__main__":
    unittest.main()
