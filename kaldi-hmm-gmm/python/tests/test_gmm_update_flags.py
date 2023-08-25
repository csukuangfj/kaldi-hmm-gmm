#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_gmm_update_flags.py
import unittest
import kaldi_hmm_gmm as khg


class TestGmmUpdateFlags(unittest.TestCase):
    def test1(self):
        means = khg.GmmUpdateFlags.kGmmMeans
        assert int(means) == 1, int(means)
        assert khg.gmm_flags_to_str(means) == "m", khg.gmm_flags_to_str(means)

        variances = khg.GmmUpdateFlags.kGmmVariances
        assert int(variances) == 2, int(variances)
        assert khg.gmm_flags_to_str(variances) == "v", khg.gmm_flags_to_str(variances)

        weights = khg.GmmUpdateFlags.kGmmWeights
        assert int(weights) == 4, int(weights)
        assert khg.gmm_flags_to_str(weights) == "w", khg.gmm_flags_to_str(weights)

        transitions = khg.GmmUpdateFlags.kGmmTransitions
        assert int(transitions) == 8, int(transitions)
        assert khg.gmm_flags_to_str(transitions) == "t", khg.gmm_flags_to_str(
            transitions
        )

        all = khg.GmmUpdateFlags.kGmmAll
        assert int(all) == 15, 1 | 2 | 4 | 8
        assert khg.gmm_flags_to_str(all) == "mvwt", khg.gmm_flags_to_str(all)

    def test2(self):
        khg.str_to_gmm_flags("m") == khg.GmmUpdateFlags.kGmmMeans
        khg.str_to_gmm_flags("v") == khg.GmmUpdateFlags.kGmmVariances
        khg.str_to_gmm_flags("w") == khg.GmmUpdateFlags.kGmmWeights
        khg.str_to_gmm_flags("t") == khg.GmmUpdateFlags.kGmmTransitions
        khg.str_to_gmm_flags("mvwt") == khg.GmmUpdateFlags.kGmmAll
        khg.str_to_gmm_flags("a") == khg.GmmUpdateFlags.kGmmAll


if __name__ == "__main__":
    unittest.main()
