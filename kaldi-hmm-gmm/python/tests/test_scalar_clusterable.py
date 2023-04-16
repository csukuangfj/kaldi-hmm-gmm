#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_scalar_clusterable_py

import unittest

import kaldi_hmm_gmm as khg


class TestScalarClusterable(unittest.TestCase):
    def test_objf_plus(self):
        a = khg.ScalarClusterable(1.0)
        assert a.objf() == 0, a.objf()

        b = khg.ScalarClusterable(2.5)
        assert b.objf() == 0, b.objf()
        # 0.5 because half-distance, squared = 1/4, times two points...
        assert a.objf_plus(b) == -0.5 * (1.0 - 2.5) * (1.0 - 2.5)

    def test_objf_minus(self):
        a = khg.ScalarClusterable(1.0)
        b = khg.ScalarClusterable(2.5)

        assert a.objf() == 0, a.objf()
        assert b.objf() == 0, b.objf()

        a.add(b)

        assert a.objf_minus(b) == 0
        a.add(b)
        assert a.objf_minus(b) == -0.5 * (1.0 - 2.5) * (1.0 - 2.5)

    def test_distance(self):
        a = khg.ScalarClusterable(1.0)
        b = khg.ScalarClusterable(2.5)

        assert a.objf() == 0, a.objf()
        assert b.objf() == 0, b.objf()

        assert a.objf_plus(b) == -a.distance(b)

    def test_sum_objf_and_normalizer(self):
        a = khg.ScalarClusterable(1.0)
        b = khg.ScalarClusterable(2.5)

        assert a.objf() == 0, a.objf()
        assert b.objf() == 0, b.objf()

        a.add(b)

        assert khg.sum_clusterable_objf([a, a, a]) == 3 * a.objf()
        assert khg.sum_clusterable_normalizer([a, a, a]) == 3 * a.normalizer()

    def test_sum(self):
        a = khg.ScalarClusterable(1.0)
        b = khg.ScalarClusterable(2.5)

        s = khg.sum_clusterable([a, b])
        assert a.objf_plus(b) == s.objf()


if __name__ == "__main__":
    unittest.main()
