#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_eigen_py


import unittest

import _kaldi_hmm_gmm as khg
import numpy as np
import torch


class TestEigen(unittest.TestCase):
    def test(self):
        foo = khg.TestEigen()
        m1 = foo.m1
        assert isinstance(m1, np.ndarray)
        assert m1.shape == (2, 3)
        np.testing.assert_array_equal(m1, np.ones((2, 3)))
        m1[0, 0] = 10
        assert foo.m1[0, 0] == 10, foo.m1

        self.m1 = np.array([1, 2])
        np.testing.assert_array_equal(self.m1, np.array([1, 2]))

        a = torch.tensor([1, 2])
        self.m1 = a  # data is not copied!
        np.testing.assert_array_equal(self.m1, a)

        a[0] = 10  # also changes self.m1

        np.testing.assert_array_equal(self.m1, a)


if __name__ == "__main__":
    unittest.main()
