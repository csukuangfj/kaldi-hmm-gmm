#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_ali_to_phones_py
import unittest

import kaldi_hmm_gmm as khg
from ali_to_phones import ali_to_phones, int2sym, split_to_phones

TOPO_STR = """
<Topology>
<TopologyEntry>
<ForPhones> 1 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones> 2 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
</Topology>
"""


def build_transition_model():
    topo = khg.HmmTopology()
    topo.read(TOPO_STR)
    tree = khg.monophone_context_dependency(
        phones=topo.phones,
        phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
    )
    return khg.TransitionModel(ctx_dep=tree, hmm_topo=topo)


def phone_instance(tm, phone):
    """One instance of a 3-state phone: 3 states x (1 self-loop + 1 forward)."""
    ali = []
    states = sorted(
        ts
        for ts in range(1, tm.num_transition_states + 1)
        if tm.transition_state_to_phone(ts) == phone
    )
    for ts in states:
        ali.append(tm.pair_to_transition_id(ts, 0))  # self-loop
        ali.append(tm.pair_to_transition_id(ts, 1))  # forward (last is final)
    return ali


class TestAliToPhones(unittest.TestCase):
    def test_split_to_phones_separates_adjacent_same_phone(self):
        tm = build_transition_model()
        instance1 = phone_instance(tm, 1)
        instance2 = phone_instance(tm, 1)  # same phone again
        ali = instance1 + instance2

        split = split_to_phones(tm, ali)
        self.assertEqual(len(split), 2)
        self.assertEqual(split[0], instance1)
        self.assertEqual(split[1], instance2)

    def test_per_frame_length_matches_alignment(self):
        tm = build_transition_model()
        ali = phone_instance(tm, 1) + phone_instance(tm, 2)
        phones = ali_to_phones(tm, ali, per_frame=True)
        self.assertEqual(len(phones), len(ali))
        # phone 1 first, then phone 2
        self.assertTrue(all(p == 1 for p in phones[:6]))
        self.assertTrue(all(p == 2 for p in phones[6:]))

    def test_collapsed(self):
        tm = build_transition_model()
        ali = phone_instance(tm, 1) + phone_instance(tm, 1) + phone_instance(tm, 2)
        phones = ali_to_phones(tm, ali, per_frame=False)
        self.assertEqual(phones, [1, 1, 2])

    def test_write_lengths(self):
        tm = build_transition_model()
        ali = phone_instance(tm, 1) + phone_instance(tm, 2)
        pairs = ali_to_phones(tm, ali, per_frame=False, write_lengths=True)
        self.assertEqual(pairs, [(1, 6), (2, 6)])

    def test_per_frame_and_write_lengths_incompatible(self):
        tm = build_transition_model()
        ali = phone_instance(tm, 1)
        with self.assertRaises(AssertionError):
            ali_to_phones(tm, ali, per_frame=True, write_lengths=True)

    def test_int2sym(self):
        id2sym = {1: "SIL", 2: "N", 3: "Y"}
        self.assertEqual(int2sym([1, 3, 2], id2sym), ["SIL", "Y", "N"])


if __name__ == "__main__":
    unittest.main()
