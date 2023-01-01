#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_draw_tree_py

import unittest

import kaldi_hmm_gmm as khg
import graphviz


class TestDrawTree(unittest.TestCase):
    def test_monophone_context_dependency(self):
        tree = khg.monophone_context_dependency(
            phones=[1, 2, 3],
            phone2num_pdf_classes=[0, 5, 3, 3],
        )
        tree.write(binary=False, filename="tree")
        with open("phones.txt", "w") as f:
            f.write("SIL 1\n")
            f.write("a 2\n")
            f.write("b 3\n")
        dot = khg.draw_tree(phones_txt="phones.txt", tree="./tree")
        source = graphviz.Source(dot)
        source.render("tree", format="pdf")  # It will generate ./tree.pdf

    def test_monophone_context_dependency_shared(self):
        tree = khg.monophone_context_dependency_shared(
            phone_classes=[[1], [2, 4], [3]],
            phone2num_pdf_classes=[0, 5, 3, 3, 3],
        )
        tree.write(binary=False, filename="tree_shared")
        with open("phones.txt", "w") as f:
            f.write("SIL 1\n")
            f.write("a 2\n")
            f.write("b 3\n")
            f.write("c 4\n")
        dot = khg.draw_tree(phones_txt="phones.txt", tree="./tree_shared")
        source = graphviz.Source(dot)
        source.render("tree_shared", format="pdf")  # It will generate ./tree_shared.pdf


if __name__ == "__main__":
    unittest.main()
