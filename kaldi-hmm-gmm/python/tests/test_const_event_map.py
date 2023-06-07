#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_constant_event_map_py
import unittest
import torch
import kaldi_hmm_gmm as khg

# An event is a list of pairs. Each pair has two integers.
# The first int is the key, while the second int is the value
# The list must be sorted by keys in strictly ascending order.
# No duplicates are allowed for keys


class TestConstantEventMap(unittest.TestCase):
    def test(self):
        m = khg.ConstantEventMap(3)

        # __str__
        assert str(m) == "CE 3 ", str(m)

        # check
        # keys in the events must be sorted in strictly ascending order
        with self.assertRaises(RuntimeError):
            m.check([(1, 1), (1, 2)])  # duplicates

        with self.assertRaises(RuntimeError):
            m.check([(2, 1), (1, 2)])  # no in ascending order

        m.check([(1, 1), (2, 2)])  # ok

        # lookup
        # Return a tuple: (found, answer)
        # answer is valid only when found is true
        found, answer = m.lookup(event=[(1, 2), (2, 30), (3, 40)], key=2)
        assert found is True, found
        assert answer == 30, answer

        found, answer = m.lookup(event=[(1, 2), (2, 30), (3, 40)], key=4)
        assert found is False, found
        # don't read answer since it is invalid when found is False

        # map
        # For constant event map, it always returns the stored value
        found, answer = m.map(event=[(1, 200)])
        assert found is True, found
        assert answer == 3, answer

        found, answer = m.map(event=[(1, 30), (2, 400)])
        assert found is True, found
        assert answer == 3, answer

        # multimap
        answers = m.multimap(event=[(1, 20)])
        assert answers == [3], answers

        # It just append the stored value to the given answers
        answers = m.multimap(event=[(1, 20)], answers=[1, 5])
        assert answers == [1, 5, 3], answers


if __name__ == "__main__":
    torch.manual_seed(20230522)
    unittest.main()
