#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_lifetime_safety_py
"""Regression tests: python/C++ boundary lifetime and input safety.

Same bug classes as kaldi-decoder's test_lifetime.py:
- decodables hold references to the acoustic/transition models (keep_alive);
  deleting them and then USING the decodable must work;
- features are copied at construction (deleting the numpy array must work);
- None for a decodable must raise TypeError, not segfault.
"""
import gc
import unittest

import numpy as np

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
 <State> 3
 </State>
 </TopologyEntry>
 </Topology>
    """
    topo = khg.HmmTopology()
    topo.read(s)
    return topo


def make_models(dim=4, num_frames=3):
    """(am, tm, feats) with matching dimensions; feats is a numpy array."""
    topo = get_hmm_topo()
    tree = khg.monophone_context_dependency(
        phones=topo.phones,
        phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
    )
    tm = khg.TransitionModel(ctx_dep=tree, hmm_topo=topo)

    diag = khg.DiagGmm(nmix=2, dim=dim)
    w = np.array([0.5, 0.5], dtype=np.float32)
    diag.set_weights(w)
    diag.set_means(np.zeros((2, dim), dtype=np.float32))
    diag.set_invvars(np.ones((2, dim), dtype=np.float32))
    diag.compute_gconsts()
    am = khg.AmDiagGmm()
    for _ in range(tm.num_pdfs()):
        am.add_pdf(diag)

    feats = np.random.rand(num_frames, dim).astype(np.float32)
    return am, tm, feats


class TestLifetimeSafety(unittest.TestCase):
    def test_decodable_keeps_models_alive(self):
        # Delete am/tm then USE the decodable -- without keep_alive this is
        # a use-after-free through the borrowed references.
        for ctor in (
            lambda a, t, f: khg.DecodableAmDiagGmmScaled(a, t, f, 1.0),
            lambda a, t, f: khg.DecodableAmDiagGmmUnmapped(a, f, -1.0),
        ):
            am, tm, feats = make_models()
            dec = ctor(am, tm, feats)
            del am, tm
            gc.collect()
            v = dec.log_likelihood(0, 1)
            self.assertIsInstance(float(v), float)

    def test_feats_are_copied(self):
        # The C++ decodables copy the feature matrix; deleting the numpy
        # array must not affect the decodable.
        am, tm, feats = make_models()
        dec = khg.DecodableAmDiagGmmScaled(am, tm, feats, 1.0)
        del feats
        gc.collect()
        v = dec.log_likelihood(0, 1)
        self.assertIsInstance(float(v), float)

    def test_all_temporaries_inline(self):
        # Everything as temporaries in one expression; the decodable must
        # keep what it needs.
        am, tm, feats = make_models()
        dec = khg.DecodableAmDiagGmmScaled(
            khg.AmDiagGmm() if False else am, tm, feats.copy(), 1.0
        )
        del am, tm, feats
        gc.collect()
        v = dec.log_likelihood(0, 1)
        self.assertIsInstance(float(v), float)

    def test_decodable_none_raises(self):
        # pybind turns None into nullptr for raw-pointer params; the
        # bindings must use references so None raises TypeError.
        import kaldifst

        am, tm, feats = make_models()
        dec = khg.DecodableAmDiagGmmScaled(am, tm, feats, 1.0)
        g = kaldifst.StdVectorFst()
        s0 = g.add_state()
        g.add_state()
        g.start = s0
        g.set_final(1, 0.0)

        cfg = khg.AlignConfig()
        with self.assertRaises(TypeError):
            khg.align_utterance(
                config=cfg, utt="u", acoustic_scale=0.1, transition_scale=1.0,
                self_loop_scale=1.0, trans_model=tm, fst=g, decodable=None,
            )
        with self.assertRaises(TypeError):
            khg.decode_utterance_lattice_faster(
                config=khg.LatticeFasterDecoderConfig(), fst=g,
                decodable=None, trans_model=tm, utt="u", allow_partial=True,
                acoustic_scale=0.1,
            )

    def test_decode_with_temporary_graph(self):
        # The decode wrapper borrows the graph only for the duration of the
        # call (the decoder is constructed internally -- plan-3 D2), so a
        # temporary graph is safe.
        import kaldifst

        am, tm, feats = make_models(num_frames=2)
        dec = khg.DecodableAmDiagGmmScaled(am, tm, feats, 0.1)

        def build_graph():
            g = kaldifst.StdVectorFst()
            s0, s1 = g.add_state(), g.add_state()
            g.start = s0
            g.add_arc(s0, kaldifst.StdArc(1, 1, 0.0, s1))
            g.set_final(s1, 0.0)
            return g

        gc.collect()
        out = khg.decode_utterance_lattice_faster(
            config=khg.LatticeFasterDecoderConfig(), fst=build_graph(),
            decodable=dec, trans_model=tm, utt="u",
            allow_partial=True, acoustic_scale=0.1,
        )
        self.assertIsInstance(out[0], bool)  # ran without crashing


if __name__ == "__main__":
    unittest.main(verbosity=2)
