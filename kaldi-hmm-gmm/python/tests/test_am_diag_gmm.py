#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_am_diag_gmm_py
import pickle
import unittest

import kaldi_hmm_gmm as khg
import numpy as np


class TestAmDiagGmm(unittest.TestCase):
    def test(self):
        nmix = 10
        dim = 8

        diag_gmm = khg.DiagGmm(nmix=nmix, dim=dim)
        weights = np.random.rand(nmix).astype(np.float32)
        weights /= weights.sum()

        mean = np.random.rand(nmix, dim)
        var = np.random.rand(nmix, dim)
        diag_gmm.set_weights(weights)
        diag_gmm.set_means(mean)
        diag_gmm.set_invvars(1 / var)

        diag_gmm.compute_gconsts()

        am = khg.AmDiagGmm()
        am.add_pdf(diag_gmm)
        assert am.num_pdfs == 1
        assert am.num_gauss == nmix

        am.add_pdf(diag_gmm)
        assert am.num_pdfs == 2
        assert am.num_gauss == nmix * 2
        assert am.num_gauss_in_pdf(0) == nmix
        assert am.num_gauss_in_pdf(1) == nmix

        am.split_pdf(pdf_idx=1, target_components=nmix * 5, perturb_factor=0.01)
        assert am.num_gauss_in_pdf(1) == nmix * 5
        assert am.num_gauss == nmix + nmix * 5

        w = am.get_pdf(0).weights[0].item()
        am.get_pdf(0).weights[0] += 0.25
        w2 = am.get_pdf(0).weights[0].item()
        assert abs(w2 - w - 0.25) < 1e-3, (w2, w, w2 - w)

        # now test pickle
        data = pickle.dumps(am, 2)  # Must use pickle protocol >= 2
        am2 = pickle.loads(data)
        assert isinstance(am2, khg.AmDiagGmm), type(am2)
        assert am.num_pdfs == am2.num_pdfs, (am.num_pdfs, am2.num_pdfs)
        for i in range(am.num_pdfs):
            pdf = am.get_pdf(i)
            pdf2 = am2.get_pdf(i)

            assert pdf.valid_gconsts is True
            assert pdf2.valid_gconsts is True

            assert np.allclose(
                np.asarray(pdf.weights), np.asarray(pdf2.weights)
            )
            assert np.allclose(
                np.asarray(pdf.means_invvars),
                np.asarray(pdf2.means_invvars),
            )
            assert np.allclose(
                np.asarray(pdf.inv_vars), np.asarray(pdf2.inv_vars)
            )


if __name__ == "__main__":
    np.random.seed(20230416)
    unittest.main()
