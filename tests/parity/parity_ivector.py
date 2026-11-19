#!/usr/bin/env python3
"""i-vector parity: kaldi binaries vs this repo's OnlineIvectorExtractor.

Reference chain per utterance (mirrors kaldi's OnlineIvectorFeature):

  feats -> apply-cmvn-online -> splice-feats -> transform-feats  -> gmm-global-get-post
  feats -----------------------> splice-feats -> transform-feats --+
                                                                  v
                            ivector-extract-online (final.ie, post, stats-feats)

Our side consumes the raw MFCC matrix only (the driver builds both chains).

Usage (from tests/parity):
  python3 parity_ivector.py --kaldi-dir /path/to/kaldi/src \
      --model-dir /path/to/vosk-model/ivector \
      --wav a.wav --wav b.wav [--config /path/to/mfcc.conf]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldi_native_io  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi-dir", default="/Users/fangjun/open-source/kaldi/src")
    p.add_argument("--model-dir", required=True,
                   help="a model's ivector/ dir (final.ie, final.dubm, ...)")
    p.add_argument("--wav", action="append", required=True,
                   help="test wav (mono, 16 kHz)")
    p.add_argument("--config", required=True, help="mfcc.conf")
    p.add_argument("--max-count", type=float, default=100.0,
                   help="kaldi/vosk value")
    p.add_argument("--ivector-period", type=int, default=10)
    p.add_argument("--num-cg-iters", type=int, default=15)
    p.add_argument("--tolerance", type=float, default=1e-2,
                   help="max relative error of iVector rows (residuals are "
                        "warm-started CG float noise; the cosine gate is the "
                        "stronger correctness criterion)")
    return p.parse_args()


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def read_ark(path):
    with kaldi_native_io.SequentialFloatMatrixReader(f"ark:{path}") as r:
        return {k: np.array(v, dtype=np.float32, copy=True) for k, v in r}


def kaldi_reference(args, tmp, raw_ark):
    """Return {utt: [N, S]} iVectors from the kaldi binary chain."""
    kaldi = Path(args.kaldi_dir)
    md = Path(args.model_dir)

    cmvn_ark, post_feats_ark, stats_feats_ark, post_ark, iv_ark = (
        tmp / n for n in ("cmvn.ark", "post_feats.ark", "stats_feats.ark",
                         "post.ark", "iv.ark"))

    # posterior chain: cmvn -> splice -> lda
    run([str(kaldi / "online2bin" / "apply-cmvn-online"),
         str(md / "global_cmvn.stats"), f"ark:{raw_ark}", f"ark:{cmvn_ark}"])
    run([str(kaldi / "featbin" / "splice-feats"), "--left-context=3",
         "--right-context=3", f"ark:{cmvn_ark}", f"ark:{tmp / 'cmvn_sp.ark'}"])
    run([str(kaldi / "featbin" / "transform-feats"), str(md / "final.mat"),
         f"ark:{tmp / 'cmvn_sp.ark'}", f"ark:{post_feats_ark}"])

    # stats chain: splice -> lda (no cmvn; matches vosk's config)
    run([str(kaldi / "featbin" / "splice-feats"), "--left-context=3",
         "--right-context=3", f"ark:{raw_ark}", f"ark:{tmp / 'raw_sp.ark'}"])
    run([str(kaldi / "featbin" / "transform-feats"), str(md / "final.mat"),
         f"ark:{tmp / 'raw_sp.ark'}", f"ark:{stats_feats_ark}"])

    # UBM posteriors + i-vectors
    run([str(kaldi / "gmmbin" / "gmm-global-get-post"), "--min-post=0",
         str(md / "final.dubm"), f"ark:{post_feats_ark}", f"ark:{post_ark}"])
    run([str(kaldi / "ivectorbin" / "ivector-extract-online"),
         f"--ivector-period={args.ivector_period}",
         f"--max-count={args.max_count}",
         f"--num-cg-iters={args.num_cg_iters}",
         str(md / "final.ie"), f"ark:{stats_feats_ark}", f"ark:{post_ark}",
         f"ark:{iv_ark}"])
    with kaldi_native_io.SequentialFloatMatrixReader(f"ark:{iv_ark}") as r:
        return {k: np.array(v, dtype=np.float64, copy=True) for k, v in r}


def our_ivectors(args, raw_by_utt):
    md = Path(args.model_dir)
    extractor = khg.IvectorExtractor.read(str(md / "final.ie"))
    ubm = khg.read_diag_gmm(str(md / "final.dubm"))
    lda = kaldi_native_io.FloatMatrix.read(str(md / "final.mat")).numpy()
    gs = kaldi_native_io.DoubleMatrix.read(str(md / "global_cmvn.stats")).numpy()

    cfg = khg.OnlineIvectorExtractorConfig(
        left_context=3, right_context=3,
        lda_mat=lda.astype(np.float32),
        global_cmvn_stats=gs.astype(np.float64),
        max_count=args.max_count,
        num_cg_iters=args.num_cg_iters,
        ivector_period=args.ivector_period)
    out = {}
    for utt, feats in raw_by_utt.items():
        eng = khg.OnlineIvectorExtractor(cfg, extractor, ubm)
        eng.accept_features(feats)
        eng.flush()
        out[utt] = np.array(eng.get_ivectors(), dtype=np.float64, copy=True)
    return out


def main():
    args = get_args()
    wav_paths = [Path(w) for w in args.wav]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        kaldi = Path(args.kaldi_dir)

        wav_scp = tmp / "wav.scp"
        wav_scp.write_text("".join(
            f"{p.stem} {p.resolve()}\n" for p in wav_paths))
        raw_ark = tmp / "raw.ark"
        run([str(kaldi / "featbin" / "compute-mfcc-feats"), "--dither=0",
             f"--config={args.config}", f"scp,p:{wav_scp}", f"ark:{raw_ark}"])
        raw = read_ark(raw_ark)

        want = kaldi_reference(args, tmp, raw_ark)
        got = our_ivectors(args, raw)

        failed = False
        for p in wav_paths:
            utt = p.stem
            w, g = want[utt], got[utt]
            if w.shape != g.shape:
                print(f"  {utt}: shape mismatch kaldi {w.shape} vs ours {g.shape} FAIL")
                failed = True
                continue
            abs_err = np.abs(w - g).max()
            rel_err = float((np.abs(w - g).max() /
                             max(np.abs(w).max(), 1e-10)))
            # per-row cosine of the non-offset dims
            cos = [float(np.dot(w[i, 1:], g[i, 1:]) /
                         (np.linalg.norm(w[i, 1:]) * np.linalg.norm(g[i, 1:]) + 1e-12))
                   for i in range(len(w))]
            ok = rel_err < args.tolerance and min(cos) > 0.9999
            failed |= not ok
            print(f"  {utt}: {w.shape} max_abs={abs_err:.6f} "
                  f"max_rel={rel_err:.2e} min_cos={min(cos):.6f} "
                  f"{'OK' if ok else 'FAIL'}")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
