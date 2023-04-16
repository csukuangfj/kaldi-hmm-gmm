#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_hmm_topology_py
import unittest
import kaldi_hmm_gmm as khg


class TestHmmTopology(unittest.TestCase):
    def test(self):
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
 </Topology>
        """
        topo = khg.HmmTopology()
        topo.read(s)
        print(topo)
        topo.check()
        t = topo.topology_for_phone(1)
        for i in t:
            print(i)
        print(topo.num_pdf_classes(phone=1))
        print(topo.get_phone_to_num_pdf_classes())
        print(topo.min_length(phone=1))
        print(topo.phones)
        print(topo.is_hmm)


if __name__ == "__main__":
    unittest.main()
