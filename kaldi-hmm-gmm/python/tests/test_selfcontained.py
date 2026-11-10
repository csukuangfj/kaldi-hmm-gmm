#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R  test_selfcontained_py
#
# Verifies that kaldi_hmm_gmm is self-contained at the Python level.
#   1. the full API flow works without the kaldi-decoder package;
#   2. the registered option types match the kaldi-decoder package's
#      field-by-field (when it happens to be installed);
#   3. importing both packages in one process is safe in any order, but
#      their objects are intentionally not interchangeable;
#   4. DecodableInterface/TransitionInformation support Python subclasses
#      (trampolines).

import subprocess
import sys
import unittest

import numpy as np

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


def build_mono_system(dim=3, num_frames=80):
    """Build a tiny mono GMM system in memory (no model files)."""
    hmm_topo = get_hmm_topo()
    phones = hmm_topo.phones
    phone2num_pdf_classes = [0] * (phones[-1] + 1)
    for p in phones:
        phone2num_pdf_classes[p] = hmm_topo.num_pdf_classes(p)
    ctx_dep = khg.monophone_context_dependency(
        phones=phones, phone2num_pdf_classes=phone2num_pdf_classes
    )

    rng = np.random.default_rng(0)
    features = rng.random((num_frames, dim), dtype=np.float32)

    am_gmm = khg.AmDiagGmm()
    for _ in range(ctx_dep.num_pdfs):
        gmm = khg.DiagGmm(nmix=1, dim=dim)
        glob_inv_var = (1.0 / features.var(axis=0)).astype(np.float32)
        glob_mean = features.mean(axis=0)
        gmm.set_invvars_and_means(
            np.expand_dims(glob_inv_var, 0), np.expand_dims(glob_mean, 0)
        )
        gmm.set_weights(np.asarray([1], dtype=np.float32))
        gmm.compute_gconsts()
        am_gmm.add_pdf(gmm)

    tm = khg.TransitionModel(ctx_dep=ctx_dep, hmm_topo=hmm_topo)
    return hmm_topo, ctx_dep, tm, am_gmm, features


def build_lex_fst(phone2id, word_phones):
    """kaldi-style L.fst with optional silence around words (no disambig)."""
    import kaldifst

    sil_id = phone2id["sil"]
    sil_prob = 0.5
    sil_cost = -np.log(sil_prob)
    no_sil_cost = -np.log(1.0 - sil_prob)
    fst = kaldifst.StdVectorFst()
    start_s = fst.add_state()
    loop = fst.add_state()
    sil_s = fst.add_state()
    fst.start = start_s
    fst.set_final(loop, 0.0)
    fst.add_arc(start_s, kaldifst.StdArc(0, 0, no_sil_cost, loop))
    fst.add_arc(start_s, kaldifst.StdArc(0, 0, sil_cost, sil_s))
    fst.add_arc(sil_s, kaldifst.StdArc(sil_id, 0, 0.0, loop))
    for w, ps in word_phones.items():
        seq = [phone2id[p] for p in ps]
        cur = loop
        for k in range(len(seq) - 1):
            nxt = fst.add_state()
            fst.add_arc(
                cur, kaldifst.StdArc(seq[k], w if k == 0 else 0, 0.0, nxt)
            )
            cur = nxt
        last = seq[-1]
        olab = w if len(seq) == 1 else 0
        fst.add_arc(cur, kaldifst.StdArc(last, olab, no_sil_cost, loop))
        fst.add_arc(cur, kaldifst.StdArc(last, olab, sil_cost, sil_s))
    kaldifst.arcsort(fst, sort_type="olabel")
    return fst


class TestSelfContained(unittest.TestCase):
    def test_full_api_flow(self):
        """read-free end to end: graph compile -> align -> decode -> best path."""
        hmm_topo, ctx_dep, tm, am_gmm, feats = build_mono_system()

        # phone ids: 1..3 are real phones; give 'sil' a free id for the
        # optional silence of the lexicon FST.
        phone2id = {p: p for p in hmm_topo.phones}
        phone2id["sil"] = max(phone2id.values()) + 1
        word_phones = {1: [1, 2], 2: [2, 3]}
        lex_fst = build_lex_fst(phone2id, word_phones)

        gc = khg.TrainingGraphCompiler(
            trans_model=tm,
            ctx_dep=ctx_dep,
            lex_fst=lex_fst,
            disambig_syms=[],
            opts=khg.TrainingGraphCompilerOptions(
                transition_scale=0.0, self_loop_scale=0.0
            ),
        )
        graph = gc.compile_graph_from_text([1, 2])

        decodable = khg.DecodableAmDiagGmmScaled(
            am=am_gmm, tm=tm, feats=feats, scale=0.1
        )

        cfg = khg.AlignConfig(beam=10.0, retry_beam=40.0)
        ans = khg.align_utterance(
            config=cfg,
            utt="test",
            acoustic_scale=0.1,
            transition_scale=1.0,
            self_loop_scale=0.1,
            trans_model=tm,
            fst=graph.copy(),
            decodable=decodable,
        )
        self.assertEqual(len(ans), 7)
        num_done, num_error, _retried, _like, frames, alignment, words = ans
        self.assertEqual(num_error, 0)
        self.assertEqual(num_done, 1)
        self.assertEqual(frames, feats.shape[0])
        self.assertGreater(len(alignment), 0)
        self.assertEqual(words, [1, 2])

        dec_cfg = khg.LatticeFasterDecoderConfig(
            beam=16.0, max_active=2000, lattice_beam=10.0
        )
        ok, _ali, _w, _ll, lat = khg.decode_utterance_lattice_faster(
            config=dec_cfg,
            fst=graph.copy(),
            decodable=decodable,
            trans_model=tm,
            utt="test",
            allow_partial=True,
            acoustic_scale=0.1,
        )
        self.assertTrue(ok)
        best_ok, _b_ali, b_words, _b_like = khg.lattice_best_path(lat)
        self.assertTrue(best_ok)
        self.assertEqual(b_words, [1, 2])

    def test_config_registration_matches_kaldi_decoder(self):
        """Registration parity with the kaldi-decoder package (if installed).

        The option structs are ported from kaldi-decoder's
        bindings; field names and defaults must match exactly.
        """
        try:
            import kaldi_decoder as kd
        except ImportError:
            self.skipTest("kaldi-decoder is not installed")

        pairs = [
            (khg.AlignConfig, kd.AlignConfig, ["beam", "retry_beam", "careful"]),
            (
                khg.LatticeFasterDecoderConfig,
                kd.LatticeFasterDecoderConfig,
                [
                    "beam",
                    "max_active",
                    "min_active",
                    "lattice_beam",
                    "prune_interval",
                    "determinize_lattice",
                    "beam_delta",
                    "hash_ratio",
                    "prune_scale",
                    "memory_pool_tokens_block_size",
                    "memory_pool_links_block_size",
                ],
            ),
            (
                khg.LatticeSimpleDecoderConfig,
                kd.LatticeSimpleDecoderConfig,
                [
                    "beam",
                    "lattice_beam",
                    "prune_interval",
                    "determinize_lattice",
                    "prune_lattice",
                    "beam_ratio",
                    "prune_scale",
                ],
            ),
        ]
        for ours, theirs, fields in pairs:
            a = ours()
            b = theirs()
            for f in fields:
                self.assertEqual(
                    getattr(a, f), getattr(b, f), f"{ours.__name__}.{f}"
                )

    def test_dual_package_coexistence(self):
        """Both import orders are safe; objects do not cross."""
        try:
            import kaldi_decoder  # noqa: F401
        except ImportError:
            self.skipTest("kaldi-decoder is not installed")

        for order in ("khg_first", "kd_first"):
            code = (
                "import importlib\n"
                "first, second = "
                f"{'(\"kaldi_hmm_gmm\", \"kaldi_decoder\")' if order == 'khg_first' else ('(\"kaldi_decoder\", \"kaldi_hmm_gmm\")')}\n"
                "importlib.import_module(first)\n"
                "importlib.import_module(second)\n"
            )
            r = subprocess.run([sys.executable, "-c", code], capture_output=True)
            self.assertEqual(r.returncode, 0, r.stderr.decode())

        # objects are intentionally NOT interchangeable across the packages
        import kaldi_decoder as kd

        self.assertFalse(isinstance(khg.AlignConfig(), kd.AlignConfig))
        self.assertFalse(
            issubclass(khg.DecodableAmDiagGmmScaled, kd.DecodableInterface)
        )

    def test_trampoline_subclassing(self):
        """DecodableInterface/TransitionInformation accept Python subclasses."""

        class MyDecodable(khg.DecodableInterface):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def log_likelihood(self, frame, index):
                self.calls += 1
                return float(frame + index)

            def is_last_frame(self, frame):
                return frame >= 2

            def num_frames_ready(self):
                return 3

            def num_indices(self):
                return 5

        d = MyDecodable()
        self.assertIsInstance(d, khg.DecodableInterface)
        self.assertEqual(d.log_likelihood(1, 2), 3.0)
        self.assertTrue(d.is_last_frame(2))
        self.assertEqual(d.num_frames_ready(), 3)
        self.assertEqual(d.num_indices(), 5)

        class MyTm(khg.TransitionInformation):
            def transition_ids_equivalent(self, a, b):
                return a == b

            def transition_id_is_start_of_phone(self, t):
                return t == 1

            def transition_id_to_phone(self, t):
                return t

            def is_final(self, t):
                return False

            def is_self_loop(self, t):
                return t == 2

            def transition_id_to_pdf_array(self):
                return [0, 0, 1, 1, 2]

            def num_pdfs(self):
                return 2

        tm = MyTm()
        self.assertIsInstance(tm, khg.TransitionInformation)
        self.assertEqual(tm.transition_id_to_phone(3), 3)
        self.assertTrue(tm.is_self_loop(2))
        self.assertEqual(tm.transition_id_to_pdf(3), 1)
        self.assertEqual(tm.num_transition_ids(), 4)
        self.assertEqual(tm.num_pdfs(), 2)


    def test_trampoline_dispatch_through_cxx(self):
        """A Python decodable is actually called back from C++ dispatch."""
        _hmm_topo, _ctx, tm, am_gmm, feats = build_mono_system()
        phone2id = {p: p for p in _hmm_topo.phones}
        phone2id["sil"] = max(phone2id.values()) + 1
        lex_fst = build_lex_fst(phone2id, {1: [1, 2], 2: [2, 3]})
        gc = khg.TrainingGraphCompiler(
            trans_model=tm,
            ctx_dep=_ctx,
            lex_fst=lex_fst,
            disambig_syms=[],
            opts=khg.TrainingGraphCompilerOptions(
                transition_scale=0.0, self_loop_scale=0.0
            ),
        )
        graph = gc.compile_graph_from_text([1, 2])
        inner = khg.DecodableAmDiagGmmScaled(
            am=am_gmm, tm=tm, feats=feats, scale=0.1
        )

        class CountingDecodable(khg.DecodableInterface):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.calls = 0

            def log_likelihood(self, frame, index):
                self.calls += 1
                return self.wrapped.log_likelihood(frame, index)

            def is_last_frame(self, frame):
                return self.wrapped.is_last_frame(frame)

            def num_frames_ready(self):
                return self.wrapped.num_frames_ready()

            def num_indices(self):
                return self.wrapped.num_indices()

        decodable = CountingDecodable(inner)
        ok, _ali, words, like, _lat = khg.decode_utterance_lattice_faster(
            config=khg.LatticeFasterDecoderConfig(),
            fst=graph.copy(),
            decodable=decodable,
            trans_model=tm,
            utt="test",
            allow_partial=True,
            acoustic_scale=0.1,
        )
        self.assertTrue(ok)
        self.assertEqual(words, [1, 2])
        self.assertGreater(decodable.calls, 0)  # C++ dispatched into Python

        # Same decode with the pure C++ decodable must give the same answer.
        ok2, _ali2, words2, like2, _lat2 = khg.decode_utterance_lattice_faster(
            config=khg.LatticeFasterDecoderConfig(),
            fst=graph.copy(),
            decodable=inner,
            trans_model=tm,
            utt="test",
            allow_partial=True,
            acoustic_scale=0.1,
        )
        self.assertTrue(ok2)
        self.assertEqual(words2, words)
        self.assertAlmostEqual(like, like2, places=4)


if __name__ == "__main__":
    unittest.main()
