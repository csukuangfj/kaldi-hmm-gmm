#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_hmm_topology_py
import pickle
import unittest

import kaldi_hmm_gmm as khg


class TestHmmTopology(unittest.TestCase):
    def test_simple(self):
        s = """
 <Topology>
 <TopologyEntry>
 <ForPhones> 1 3 </ForPhones>
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
        # state3 has no pdf classes so it is a non-emitting state
        topo = khg.HmmTopology()
        topo.read(s)

        # We specified only PdfClass, so ForwardPdfClass = SelfLoopPdfClass = PdfClass,
        # which means topo.is_hmm returns True
        assert topo.is_hmm is True

        topo.check()  # will crash if the given topo is invalid
        # Actually, it is called in topo.read(s)

        assert topo.phones == [1, 3], topo.phones

        phone2num_pdf_classes: List[int] = topo.get_phone_to_num_pdf_classes()
        # phone2num_pdf_classes is indexed by phones
        assert phone2num_pdf_classes[0] == -1, phone2num_pdf_classes
        assert phone2num_pdf_classes[1] == 3, phone2num_pdf_classes
        assert phone2num_pdf_classes[2] == -1, phone2num_pdf_classes
        assert phone2num_pdf_classes[3] == 3, phone2num_pdf_classes
        assert len(phone2num_pdf_classes) == 4, phone2num_pdf_classes

        phone_topo: List[khg.HmmState] = topo.topology_for_phone(1)

        # phone_topo[0] is for state 0
        assert phone_topo[0].forward_pdf_class == 0, phone_topo[0].forward_pdf_class
        assert phone_topo[0].self_loop_pdf_class == 0, phone_topo[0].self_loop_pdf_class
        phone_topo[0].transitions[0][0] == 0  # dst state, self loop
        phone_topo[0].transitions[0][1] == 0.5  # transitions prob

        phone_topo[0].transitions[1][0] == 1  # dst state, foward
        phone_topo[0].transitions[1][1] == 0.5  # transtions prob

        phone_topo[1].transitions[0][0] == 1  # dst state, self loop
        phone_topo[1].transitions[0][1] == 0.5  # transtions prob

        phone_topo[1].transitions[1][0] == 2  # dst state, forward
        phone_topo[1].transitions[1][1] == 0.5  # transtions prob

        phone_topo[2].transitions[0][0] == 3  # dst state, self loop
        phone_topo[2].transitions[0][1] == 0.5  # transtions prob

        phone_topo[2].transitions[0][0] == 4  # dst state, forward
        phone_topo[2].transitions[0][1] == 0.5  # transtions prob

        assert topo.num_pdf_classes(phone=1) == 3, topo.num_pdf_classes(phone=1)

        # 0->1->2->3, the path length is 3
        assert topo.min_length(phone=1) == 3

        # test pickle
        data = pickle.dumps(topo, 2)  # Must use pickle protocol >= 2
        topo2 = pickle.loads(data)
        assert isinstance(topo2, khg.HmmTopology)

        assert str(topo) == str(topo2)

    def test_non_hmm(self):
        s = """
 <Topology>
 <TopologyEntry>
 <ForPhones> 1 2 3 4 5 6 7 8 </ForPhones>
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
 <TopologyEntry>
 <ForPhones> 10 </ForPhones>
 <State> 0 <ForwardPdfClass> 0 <SelfLoopPdfClass> 1
 <Transition> 0 0.5
 <Transition> 1 0.25
 <Transition> 2 0.25
 </State>
 <State> 1 <ForwardPdfClass> 2 <SelfLoopPdfClass> 3
 <Transition> 1 0.5
 <Transition> 2 0.25
 <Transition> 3 0.25
 </State>
 <State> 2 <ForwardPdfClass> 4 <SelfLoopPdfClass> 5
 <Transition> 2 0.5
 <Transition> 3 0.25
 <Transition> 4 0.25
 </State>
 <State> 3 <ForwardPdfClass> 6 <SelfLoopPdfClass> 7
 <Transition> 3 0.5
 <Transition> 4 0.25
 <Transition> 5 0.25
 </State>
 <State> 4 <ForwardPdfClass> 8 <SelfLoopPdfClass> 9
 <Transition> 4 0.5
 <Transition> 5 0.5
 </State>
 <State> 5
 </State>
 </TopologyEntry>
 </Topology>
        """
        # state3 has no pdf classes so it is a non-emitting state
        topo = khg.HmmTopology()
        topo.read(s)
        #  print(topo)
        topo.check()

        assert topo.num_pdf_classes(phone=1) == 3, topo.num_pdf_classes(phone=1)
        assert topo.num_pdf_classes(phone=10) == 10, topo.num_pdf_classes(phone=10)
        phone2num_pdf_classes = topo.get_phone_to_num_pdf_classes()
        assert phone2num_pdf_classes[0] == -1  # 0 is reserved
        assert phone2num_pdf_classes[9] == -1  # not exist
        assert phone2num_pdf_classes[1:9] == [3] * 8
        assert phone2num_pdf_classes[10] == 10

        assert topo.min_length(phone=1) == 3, topo.min_length(phones=1)
        assert topo.min_length(phone=10) == 3, topo.min_length(phones=10)
        assert topo.phones == [1, 2, 3, 4, 5, 6, 7, 8, 10], topo.phone
        assert topo.is_hmm is False  # topo for 10 is not HMM

        # test pickle
        data = pickle.dumps(topo, 2)  # Must use pickle protocol >= 2
        topo2 = pickle.loads(data)
        assert isinstance(topo2, khg.HmmTopology)

        assert str(topo) == str(topo2)

        dot = khg.draw_hmm_topology(topo, phone=1)
        print(dot)
        print("--------------------")

        dot = khg.draw_hmm_topology(topo, phone=10)
        print(dot)


if __name__ == "__main__":
    unittest.main()
