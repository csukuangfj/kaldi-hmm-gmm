#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_init_gmm_mono_py


import unittest

import torch

import kaldi_hmm_gmm as khg


class TestInitGmmMono(unittest.TestCase):
    def test(self):
        tree = khg.monophone_context_dependency(
            phones=[1, 2, 3],
            phone2num_pdf_classes=[0, 5, 3, 3],
        )
        num_pdfs = tree.num_pdfs
        print(num_pdfs)


if __name__ == "__main__":
    torch.manual_seed(20230416)
    unittest.main()
