#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_trnasition_model_py
import unittest
import torch
import kaldi_hmm_gmm as khg


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
        assert transition_model.transition_ids_is_start_of_phone(1)  is True
        assert transition_model.transition_ids_is_start_of_phone(2)  is True
        assert transition_model.transition_ids_is_start_of_phone(3)  is False

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
'''
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
'''


if __name__ == "__main__":
    unittest.main()
