#!/usr/bin/env python3
# To run this single test, use
#
#  ctest --verbose -R test_online_ivector_feature_py
"""OnlineIvectorFeature / OnlineSilenceWeighting through the Python
bindings (the C++ semantics are covered by online-ivector-feature-test.cc;
kaldi/vosk parity by tests/parity/parity_online_ivector_feature.py)."""
import gc
import io
import struct
import unittest

import numpy as np

import kaldi_hmm_gmm as khg


def make_extractor(tmpdir, D=3, S=2, I=4):
    """A tiny extractor in kaldi's final.ie binary format."""
    rng = np.random.default_rng(7)

    def token(s):
        return s.encode() + b" "

    def int32(v):
        return b"\x04" + struct.pack("<i", v)

    def dmat(m):
        rows, cols = m.shape
        return token("DM") + int32(rows) + int32(cols) + m.astype("<f8").tobytes()

    def dvec(v):
        return token("DV") + int32(v.size) + v.astype("<f8").tobytes()

    buf = io.BytesIO()
    buf.write(b"\x00B")
    buf.write(token("<IvectorExtractor>"))
    buf.write(token("<w>") + dmat(np.zeros((0, 0))))
    buf.write(token("<w_vec>") + dvec(np.full(I, 1.0 / I)))
    buf.write(token("<M>") + int32(I))
    for _ in range(I):
        buf.write(dmat(rng.uniform(-1, 1, (D, S))))
    buf.write(token("<SigmaInv>"))
    for _ in range(I):
        packed = []
        for r in range(D):
            for c in range(r + 1):
                packed.append(1.0 + rng.uniform(0, 1) if r == c else 0.05)
        buf.write(token("DP") + int32(D) + np.array(packed, "<f8").tobytes())
    buf.write(token("<IvectorOffset>") + b"\x08" + struct.pack("<d", 1.5))
    buf.write(token("</IvectorExtractor>"))
    path = f"{tmpdir}/final.ie"
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return khg.IvectorExtractor.read(path)


def make_setup(tmpdir, T=40):
    extractor = make_extractor(tmpdir)
    D = extractor.feat_dim
    rng = np.random.default_rng(11)
    lda = np.zeros((D, 3 * D), dtype=np.float32)
    lda[:, D : 2 * D] = np.eye(D)
    glob = np.zeros((2, D + 1))
    glob[0, D] = 100.0
    I = extractor.num_gauss
    ubm = khg.DiagGmm(nmix=I, dim=D)
    ubm.set_weights(np.full(I, 1.0 / I, dtype=np.float32))
    ubm.set_invvars_and_means(
        np.full((I, D), 0.1, dtype=np.float32),
        rng.uniform(-3, 3, (I, D)).astype(np.float32),
    )
    ubm.compute_gconsts()
    cfg = khg.OnlineIvectorExtractorConfig(
        left_context=1,
        right_context=1,
        lda_mat=lda,
        global_cmvn_stats=glob,
        max_count=20.0,
    )
    feats = rng.uniform(-5, 5, (T, D)).astype(np.float32)
    return extractor, ubm, cfg, feats


class TestOnlineIvectorFeature(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_kaldi_defaults(self):
        cfg = khg.OnlineIvectorExtractorConfig()
        self.assertEqual(cfg.num_gselect, 5)
        self.assertAlmostEqual(cfg.min_post, 0.025)
        self.assertAlmostEqual(cfg.posterior_scale, 0.1)
        self.assertTrue(cfg.use_most_recent_ivector)
        self.assertFalse(cfg.greedy_ivector_extractor)

    def test_streaming_matches_batch(self):
        # Feeding the frames in pieces and asking for the i-vector at the
        # last ready frame equals feeding everything at once.
        extractor, ubm, cfg, feats = make_setup(self.tmp.name)
        batch = khg.OnlineIvectorFeature(cfg, extractor, ubm)
        batch.accept_features(feats)
        batch.input_finished()

        stream = khg.OnlineIvectorFeature(cfg, extractor, ubm)
        for start in range(0, len(feats), 7):
            stream.accept_features(feats[start : start + 7])
            n = stream.num_frames_ready
            if n > 0:
                np.testing.assert_allclose(
                    stream.get_frame(n - 1), batch.get_frame(n - 1), atol=1e-5
                )
        stream.input_finished()
        self.assertEqual(stream.num_frames_ready, len(feats))
        np.testing.assert_allclose(
            stream.get_frame(len(feats) - 1),
            batch.get_frame(len(feats) - 1),
            atol=1e-5,
        )
        with self.assertRaises(IndexError):
            stream.get_frame(len(feats))

    def test_zero_weights_keep_prior(self):
        extractor, ubm, cfg, feats = make_setup(self.tmp.name)
        f = khg.OnlineIvectorFeature(cfg, extractor, ubm)
        f.accept_features(feats)
        f.input_finished()
        f.update_frame_weights([(t, 0.0) for t in range(len(feats))])
        np.testing.assert_allclose(f.get_frame(len(feats) - 1), 0.0, atol=1e-6)
        self.assertEqual(f.num_frames, 0.0)

    def test_lifetime(self):
        extractor, ubm, cfg, feats = make_setup(self.tmp.name)
        f = khg.OnlineIvectorFeature(cfg, extractor, ubm)
        del extractor, ubm
        gc.collect()
        f.accept_features(feats)
        self.assertEqual(f.get_frame(5).shape, (2,))


TOPO = """
<Topology>
<TopologyEntry>
<ForPhones> 1 2 </ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 </State>
</TopologyEntry>
</Topology>
"""


class TestOnlineSilenceWeighting(unittest.TestCase):
    def make_tm(self):
        topo = khg.HmmTopology()
        topo.read(TOPO)
        ctx_dep = khg.monophone_context_dependency(
            phones=topo.phones,
            phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
        )
        tm = khg.TransitionModel(ctx_dep=ctx_dep, hmm_topo=topo)
        sil = next(
            t
            for t in range(1, tm.num_transition_ids() + 1)
            if tm.transition_id_to_phone(t) == 1
        )
        speech = next(
            t
            for t in range(1, tm.num_transition_ids() + 1)
            if tm.transition_id_to_phone(t) == 2
        )
        return tm, sil, speech

    def test_delta_weights(self):
        tm, sil, speech = self.make_tm()
        cfg = khg.OnlineSilenceWeightingConfig(
            silence_phones_str="1", silence_weight=0.001
        )
        self.assertTrue(cfg.active())
        sw = khg.OnlineSilenceWeighting(tm, cfg, frame_subsampling_factor=3)
        del tm
        gc.collect()
        sw.compute_current_traceback([])
        dw = sw.get_delta_weights(6)
        self.assertEqual([f for f, _ in dw], list(range(6)))
        np.testing.assert_allclose([w for _, w in dw], 0.001, rtol=1e-6)

        sw.compute_current_traceback([sil, speech])
        dw = sw.get_delta_weights(9)
        self.assertEqual([f for f, _ in dw], list(range(3, 9)))
        np.testing.assert_allclose(
            [w for _, w in dw], [0.999] * 3 + [1.0] * 3, rtol=1e-6
        )
        self.assertEqual(sw.get_nonsilence_frames(9), [1])


if __name__ == "__main__":
    unittest.main()
