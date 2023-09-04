#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_init_gmm_mono_py


import unittest

import torch

import kaldi_hmm_gmm as khg


def get_hmm_topo():
    s = """
 <Topology>
 <TopologyEntry>
 <ForPhones> 1 2 3 </ForPhones>
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


class TestInitGmmMono(unittest.TestCase):
    def test(self):
        hmm_topo = get_hmm_topo()

        phones = hmm_topo.phones
        # 0 is not a valid phone, so we use +1 here
        phone2num_pdf_classes = [0] * (phones[-1] + 1)
        for p in phones:
            phone2num_pdf_classes[p] = hmm_topo.num_pdf_classes(p)

        ctx_dep = khg.monophone_context_dependency(
            phones=phones,
            phone2num_pdf_classes=phone2num_pdf_classes,
        )
        num_pdfs = ctx_dep.num_pdfs

        #  dim = 80
        dim = 3

        am_gmm = khg.AmDiagGmm()

        gmm = khg.DiagGmm(nmix=1, dim=dim)

        features = torch.rand(100, dim, dtype=torch.float32)

        glob_inv_var = 1.0 / features.var(dim=0)
        assert glob_inv_var.shape == (dim,)

        glob_mean = features.mean(dim=0)
        gmm.set_invvars_and_means(glob_inv_var.unsqueeze(0), glob_mean.unsqueeze(0))

        weights = torch.tensor([1], dtype=torch.float32)
        gmm.set_weights(weights)
        gmm.compute_gconsts()
        for i in range(num_pdfs):
            am_gmm.add_pdf(gmm)

        for i in range(1, num_pdfs):
            assert torch.allclose(
                torch.from_numpy(am_gmm.get_pdf(i).means),
                torch.from_numpy(am_gmm.get_pdf(0).means),
            )

        for i in range(num_pdfs):
            am_gmm.get_pdf(i).perturb(perturb_factor=0.01)

        for i in range(1, num_pdfs):
            assert not torch.allclose(
                torch.from_numpy(am_gmm.get_pdf(i).means),
                torch.from_numpy(am_gmm.get_pdf(0).means),
            )

        trans_model = khg.TransitionModel(ctx_dep, hmm_topo)
        assert trans_model.num_pdfs == num_pdfs, (trans_model.num_pdfs, num_pdfs)
        print(trans_model.topo)
        print(trans_model.num_transition_ids)
        # transition id starts from 1
        for i in range(1, trans_model.num_transition_ids + 1):
            print(
                i,
                trans_model.transition_id_to_phone(i),
                trans_model.transition_ids_is_start_of_phone(i),
                trans_model.is_final(i),
                trans_model.is_self_loop(i),
                trans_model.transition_id_to_pdf(i),
                end=", ",
            )
        print()
        print(trans_model.transition_id_to_pdf_array())
        print(len(trans_model.transition_id_to_pdf_array()))


if __name__ == "__main__":
    torch.manual_seed(20230416)
    unittest.main()
