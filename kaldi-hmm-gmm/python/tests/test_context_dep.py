#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_context_dep_py

import pickle
import unittest
from typing import List, Tuple

import graphviz
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
 <ForPhones> 2 3 4 5 6 7 8 9 10 </ForPhones>
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


class TestContextDep(unittest.TestCase):
    def test_monophone_context_dependency(self):
        topo = get_hmm_topo()
        tree = khg.monophone_context_dependency(
            phones=topo.phones,
            phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
        )

        # mono-phone
        assert tree.context_width == 1, tree.context_width
        assert tree.central_position == 0, tree.central_position

        # num_pdfs equals to number leaves in the tree
        assert tree.num_pdfs == sum(topo.get_phone_to_num_pdf_classes()[1:]), (
            tree.num_pdfs,
            topo.get_phone_to_num_pdf_classes()[1:],
        )

        # it is mono-phone, so len(phone_seq) must be 1
        found, pdf_id = tree.compute(phone_seq=[1], pdf_class=0)
        assert found is True
        assert pdf_id == 0, pdf_id

        # The largest pdf class for phone 1 is 4
        found, pdf_id = tree.compute(phone_seq=[1], pdf_class=5)
        assert found is False

        # phone 1 has 5 pdf classes, so the pdf id of phone 2 starts from 5
        found, pdf_id = tree.compute(phone_seq=[2], pdf_class=0)
        assert found is True
        assert pdf_id == 5, pdf_id

        found, pdf_id = tree.compute(phone_seq=[2], pdf_class=1)
        assert found is True
        assert pdf_id == 6, pdf_id

        # tree.get_pdf_info()
        # it is the inverse of tree.compute()
        pdf_info: List[List[Tuple[int, int]]] = tree.get_pdf_info(
            phones=topo.phones, num_pdf_classes=topo.get_phone_to_num_pdf_classes()
        )
        assert len(pdf_info) == tree.num_pdfs, (len(pdf_info), tree.num_pdfs)

        assert pdf_info[0] == [(1, 0)], pdf_info[0]  # phone0, pdf class 0
        assert pdf_info[1] == [(1, 1)]  # phone0, pdf class 1

        assert pdf_info[5] == [(2, 0)]  # phone1, pdf class 0
        assert pdf_info[6] == [(2, 1)]  # phone1, pdf class 1

        # Draw a tree

        tree.write(binary=False, filename="tree")
        with open("phones.txt", "w") as f:
            f.write("SIL 1\n")
            f.write("a 2\n")
            f.write("b 3\n")
            f.write("c 4\n")
            f.write("d 5\n")
            f.write("e 6\n")
            f.write("f 7\n")
            f.write("g 8\n")
            f.write("h 9\n")
            f.write("i 10\n")
        dot = khg.draw_tree(phones_txt="phones.txt", tree="./tree")
        source = graphviz.Source(dot)
        source.render("tree", format="pdf")  # It will generate ./tree.pdf

        # test pickle
        data = pickle.dumps(tree, 2)  # Must use pickle protocol >= 2
        tree2 = pickle.loads(data)
        assert isinstance(tree, khg.ContextDependency)

        assert str(tree) == str(tree2)

    def test_monophone_context_dependency_2(self):
        topo = get_hmm_topo()
        phones = topo.phones
        phone_set = [[phones[0]], phones[1:5], phones[5:8], phones[8:9], phones[9:10]]
        tree = khg.monophone_context_dependency_shared(
            phone_classes=phone_set,
            phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
        )
        tree.write(binary=False, filename="tree-shared")
        with open("phones.txt", "w") as f:
            f.write("SIL 1\n")
            f.write("a 2\n")
            f.write("b 3\n")
            f.write("c 4\n")
            f.write("d 5\n")
            f.write("e 6\n")
            f.write("f 7\n")
            f.write("g 8\n")
            f.write("h 9\n")
            f.write("i 10\n")

        # query: pdf_class/context-phone1/context-phone2/context-phoneN
        # note: it uses phone symbol instead of ID in the query
        dot = khg.draw_tree(phones_txt="phones.txt", tree="./tree-shared", query="0/h")
        source = graphviz.Source(dot)
        source.render("tree-shared", format="pdf")  # It will generate ./tree.pdf

        # test pickle
        data = pickle.dumps(tree, 2)  # Must use pickle protocol >= 2
        tree2 = pickle.loads(data)
        assert isinstance(tree, khg.ContextDependency)

        assert str(tree) == str(tree2)


# tree-shared
"""
ContextDependency 1 0 ToPdf SE 0 [ 1 2 3 4 5 ]
{ SE 0 [ 1 ]
{ TE -1 5 ( CE 0 CE 1 CE 2 CE 3 CE 4 )
TE -1 3 ( CE 5 CE 6 CE 7 )
}
SE 0 [ 6 7 8 ]
{ TE -1 3 ( CE 8 CE 9 CE 10 )
TE 0 11 ( NULL NULL NULL NULL NULL NULL NULL NULL NULL TE -1 3 ( CE 11 CE 12 CE 13 )
TE -1 3 ( CE 14 CE 15 CE 16 )
)
}
}
EndContextDependency
"""


if __name__ == "__main__":
    unittest.main()
