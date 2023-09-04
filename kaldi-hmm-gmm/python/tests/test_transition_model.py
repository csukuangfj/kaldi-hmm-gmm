#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_trnasition_model_py
import pickle
import unittest

import kaldi_hmm_gmm as khg
import numpy as np
import torch


def get_hmm_topo():
    s = """
 <Topology>
 <TopologyEntry>
 <ForPhones> 1 </ForPhones>
 <State> 0 <PdfClass> 0
 <Transition> 0 0.5
 <Transition> 1 0.5
 </State>
 <State> 1 <PdfClass> 1
 <Transition> 1 0.5
 <Transition> 2 0.5
 </State>
 <State> 2 <PdfClass> 2
 <Transition> 2 0.5
 <Transition> 3 0.5
 </State>
 <State> 3 <PdfClass> 3
 <Transition> 3 0.5
 <Transition> 4 0.5
 </State>
 <State> 4 <PdfClass> 4
 <Transition> 4 0.5
 <Transition> 5 0.5
 </State>
 <State> 5
 </State>
 </TopologyEntry>
 <TopologyEntry>
 <ForPhones> 2 3 4 </ForPhones>
 <State> 0 <PdfClass> 0
 <Transition> 0 0.5
 <Transition> 1 0.5
 </State>
 <State> 1 <PdfClass> 1
 <Transition> 1 0.5
 <Transition> 2 0.5
 </State>
 <State> 2 <PdfClass> 2
 <Transition> 2 0.5
 <Transition> 3 0.5
 </State>
 <State> 3
 </State>
 </TopologyEntry>
 </Topology>
    """
    topo = khg.HmmTopology()
    topo.read(s)

    return topo


class TestTransitionModel(unittest.TestCase):
    def test_mono(self):
        topo = get_hmm_topo()
        tree = khg.monophone_context_dependency(
            phones=topo.phones,
            phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
        )

        transition_model = khg.TransitionModel(ctx_dep=tree, hmm_topo=topo)

        # Get the hmm topo of this transition_model
        assert str(topo) == str(transition_model.topo)

        # Get a list of phones of this transition_model
        assert transition_model.phones == topo.phones
        assert transition_model.phones == [1, 2, 3, 4], transition_model.phones
        transition_id_to_pdf_array = transition_model.transition_id_to_pdf_array()
        # [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
        # transition_id starts from 1
        # transition_id_to_pdf_array[0] is not valid and should never be accessed

        assert transition_model.is_self_loop(1) == True
        assert transition_model.is_self_loop(2) == False
        assert transition_model.is_self_loop(3) == True
        assert transition_model.is_self_loop(4) == False

        # transition_id 1 and 2 corresponds to the same transition state
        # which maps to the same tuple (phone=1, hmm_state=1, forward_pdf_id=1, self_loop_pdf_id=1)
        assert transition_model.transition_ids_equivalent(1, 2) is True
        assert transition_model.transition_ids_equivalent(1, 3) is False

        # If it is state 0 of a phone, then it is the start state.
        #
        # transition id 1 and 2 corresponds to the hmm state 0 for phone 1
        # transition id 3 corresponds to state 1 for phone 1, which is not the
        # start state.
        assert transition_model.transition_ids_is_start_of_phone(1) is True
        assert transition_model.transition_ids_is_start_of_phone(2) is True
        assert transition_model.transition_ids_is_start_of_phone(3) is False

        assert transition_model.transition_id_to_phone(1) == 1
        assert transition_model.transition_id_to_phone(2) == 1
        assert transition_model.transition_id_to_phone(10) == 1
        assert transition_model.transition_id_to_phone(11) == 2
        assert transition_model.transition_id_to_phone(16) == 2
        assert transition_model.transition_id_to_phone(17) == 3

        # the transition of transition id 1 does not go to the final hmm stage
        assert transition_model.is_final(1) is False

        # the transition of transition id 10 goes to the final hmm stage of phone 1
        assert transition_model.is_final(10) is True

        # 5: phone 1 has 5 pdf ids
        # 3: phone 1 to phone 3, each has 3 pdf ids
        assert transition_model.num_pdfs == 5 + 3 * 3, transition_model.num_pdfs

        # test stats
        stats = transition_model.init_stats()
        assert stats.ndim == 1, stats.ndim
        assert stats.dtype == np.float64, stats.dtype

        # transition_id starts from 1, so stats[0] is never used
        assert stats.shape[0] == transition_model.num_transition_ids + 1, (
            stats.shape[0],
            transition_model.num_transition_ids + 1,
        )
        assert torch.from_numpy(stats).abs().sum().item() == 0, stats

        stats = transition_model.accumulate(prob=0.25, trans_id=1, stats=stats)
        assert stats[1].item() == 0.25, stats[1]

        stats = transition_model.accumulate(prob=0.25, trans_id=1, stats=stats)
        assert stats[1].item() == 0.50, stats[1]

        stats = transition_model.accumulate(prob=1.0, trans_id=10, stats=stats)
        assert stats[10].item() == 1.0, stats[10]

        # test pickle
        data = pickle.dumps(transition_model, 2)  # Must use pickle protocol >= 2
        transition_model2 = pickle.loads(data)
        assert isinstance(transition_model2, khg.TransitionModel)

        assert str(transition_model.topo) == str(transition_model2.topo)

        for i in range(len(transition_model.tuples)):
            assert str(transition_model.tuples[i]) == str(transition_model2.tuples[i])

        assert transition_model.state2id == transition_model2.state2id
        assert transition_model.id2state == transition_model2.id2state
        assert transition_model.id2pdf_id == transition_model2.id2pdf_id
        assert transition_model.log_probs == transition_model2.log_probs
        assert (
            transition_model.non_self_loop_log_probs
            == transition_model2.non_self_loop_log_probs
        )
        assert transition_model.num_pdfs == transition_model2.num_pdfs

        torch.save(transition_model, "transition_model.pt")
        transition_model3 = torch.load("transition_model.pt")

        assert isinstance(transition_model3, khg.TransitionModel)

        assert str(transition_model.topo) == str(transition_model3.topo)

        for i in range(len(transition_model.tuples)):
            assert str(transition_model.tuples[i]) == str(transition_model3.tuples[i])

        assert transition_model.state2id == transition_model3.state2id
        assert transition_model.id2state == transition_model3.id2state
        assert transition_model.id2pdf_id == transition_model3.id2pdf_id
        assert transition_model.log_probs == transition_model3.log_probs
        assert (
            transition_model.non_self_loop_log_probs
            == transition_model3.non_self_loop_log_probs
        )
        assert transition_model.num_pdfs == transition_model3.num_pdfs


"""
<TransitionModel>
<Topology>
<TopologyEntry>
<ForPhones>
1
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 <PdfClass> 3 <Transition> 3 0.5 <Transition> 4 0.5 </State>
<State> 4 <PdfClass> 4 <Transition> 4 0.5 <Transition> 5 0.5 </State>
<State> 5 </State>
</TopologyEntry>
<TopologyEntry>
<ForPhones>
2 3 4
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
</Topology>
<Triples> 14
1 0 0
1 1 1
1 2 2
1 3 3
1 4 4
2 0 5
2 1 6
2 2 7
3 0 8
3 1 9
3 2 10
4 0 11
4 1 12
4 2 13
</Triples>
<LogProbs>
 [ 0 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.6
93147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 -0.693147 ]
</LogProbs>
</TransitionModel>
"""


if __name__ == "__main__":
    unittest.main()
