#!/usr/bin/env python3
"""Parity: kaldi's OnlineIvectorFeature (ivector-extract-online2) vs this
repo's OnlineIvectorFeature, with and without frame weights.

kaldi side, per utterance:

  compute-mfcc-feats (dither 0) -> ivector-extract-online2 --repeat=true
      [--frame-weights-rspecifier=ark:weights.ark]

ivector-extract-online2 runs OnlineIvectorFeature with
use_most_recent_ivector=false and prints the iVector of EVERY frame
(--repeat), i.e. the estimate at the last multiple of --ivector-period.
Our side feeds the same MFCC matrix to khg.OnlineIvectorFeature with the
same weights and calls get_frame(t) for every t.  Both use the online2
defaults (num_gselect 5, min_post 0.025, posterior_scale 0.1).

The weights exercise the silence-weighting path (UpdateFrameWeights,
GetMinPost(weight)); they are random in [0, 1] with runs of 0.001 and 0.

Usage (from tests/parity):
  python3 parity_online_ivector_feature.py \
      --model-dir /path/to/vosk-model/ivector --wav a.wav --wav b.wav \
      --config /path/to/vosk-model/conf/mfcc.conf
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python")
)

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldi_native_io  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi-dir", default=None, help="kaldi src dir")
    p.add_argument("--model-dir", required=True, help="a model's ivector/ dir")
    p.add_argument("--wav", action="append", required=True)
    p.add_argument("--config", required=True, help="mfcc.conf")
    p.add_argument("--max-count", type=float, default=100.0)
    p.add_argument("--ivector-period", type=int, default=10)
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="max abs difference per element (float noise)",
    )
    return p.parse_args()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed:\n{r.stderr[-2000:]}")


def default_kaldi_src():
    for cand in (
        os.environ.get("KALDI_ROOT"),
        "/opt/kaldi",
        "/Users/fangjun/open-source/kaldi",
    ):
        if cand and os.path.isdir(os.path.join(cand, "src")):
            return os.path.join(cand, "src")
    return None


def make_weights(num_frames, seed):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 1, num_frames).astype(np.float32)
    # silence-like runs
    for start in rng.integers(0, max(1, num_frames - 30), size=3):
        w[start : start + 20] = 0.001
    w[num_frames // 2 : num_frames // 2 + 5] = 0.0
    return w


def main():
    args = get_args()
    kaldi = Path(args.kaldi_dir or default_kaldi_src())
    md = Path(args.model_dir)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        utts = [f"utt{i}" for i in range(len(args.wav))]
        (tmp / "wav.scp").write_text(
            "".join(f"{u} {w}\n" for u, w in zip(utts, args.wav))
        )
        (tmp / "spk2utt").write_text("".join(f"{u} {u}\n" for u in utts))
        run(
            [
                str(kaldi / "featbin" / "compute-mfcc-feats"),
                f"--config={args.config}",
                "--dither=0",
                f"scp:{tmp / 'wav.scp'}",
                f"ark:{tmp / 'feats.ark'}",
            ]
        )
        with kaldi_native_io.SequentialFloatMatrixReader(
            f"ark:{tmp / 'feats.ark'}"
        ) as r:
            feats = {k: np.array(v, dtype=np.float32, copy=True) for k, v in r}

        weights = {u: make_weights(len(feats[u]), i) for i, u in enumerate(utts)}
        with open(tmp / "weights.txt", "w") as f:
            for u in utts:
                f.write(u + " [ " + " ".join(f"{x:.9g}" for x in weights[u]) + " ]\n")

        conf = tmp / "ivector_extractor.conf"
        conf.write_text(
            f"--splice-config={md / 'splice.conf'}\n"
            f"--cmvn-config={md / 'online_cmvn.conf'}\n"
            f"--lda-matrix={md / 'final.mat'}\n"
            f"--global-cmvn-stats={md / 'global_cmvn.stats'}\n"
            f"--diag-ubm={md / 'final.dubm'}\n"
            f"--ivector-extractor={md / 'final.ie'}\n"
            f"--max-count={args.max_count}\n"
            f"--ivector-period={args.ivector_period}\n"
        )

        extractor = khg.IvectorExtractor.read(str(md / "final.ie"))
        ubm = khg.read_diag_gmm(str(md / "final.dubm"))
        lda = kaldi_native_io.FloatMatrix.read(str(md / "final.mat")).numpy()
        gs = kaldi_native_io.DoubleMatrix.read(str(md / "global_cmvn.stats")).numpy()
        splice = {}
        for line in (md / "splice.conf").read_text().split():
            k, v = line.lstrip("-").split("=")
            splice[k] = int(v)
        cfg = khg.OnlineIvectorExtractorConfig(
            left_context=splice["left-context"],
            right_context=splice["right-context"],
            lda_mat=lda.astype(np.float32),
            global_cmvn_stats=gs.astype(np.float64),
            max_count=args.max_count,
            ivector_period=args.ivector_period,
            use_most_recent_ivector=False,
        )

        failed = False
        for weighted in (False, True):
            cmd = [
                str(kaldi / "online2bin" / "ivector-extract-online2"),
                f"--config={conf}",
                "--repeat=true",
            ]
            if weighted:
                cmd.append(f"--frame-weights-rspecifier=ark,t:{tmp / 'weights.txt'}")
            cmd += [
                f"ark:{tmp / 'spk2utt'}",
                f"ark:{tmp / 'feats.ark'}",
                f"ark:{tmp / 'iv.ark'}",
            ]
            run(cmd)
            with kaldi_native_io.SequentialFloatMatrixReader(
                f"ark:{tmp / 'iv.ark'}"
            ) as r:
                ref = {k: np.array(v, dtype=np.float32, copy=True) for k, v in r}

            for u in utts:
                f = khg.OnlineIvectorFeature(cfg, extractor, ubm)
                f.accept_features(feats[u])
                f.input_finished()
                if weighted:
                    f.update_frame_weights(
                        [(t, float(w)) for t, w in enumerate(weights[u])]
                    )
                ours = np.stack([f.get_frame(t) for t in range(len(feats[u]))])
                diff = np.abs(ours - ref[u]).max()
                cos = np.min(
                    np.sum(ours * ref[u], axis=1)
                    / (np.linalg.norm(ours, axis=1) * np.linalg.norm(ref[u], axis=1))
                )
                ok = diff <= args.tolerance
                failed |= not ok
                print(
                    f"{'weighted' if weighted else 'unweighted':10s} {u}: "
                    f"T={len(feats[u])} max|diff|={diff:.2e} "
                    f"min cosine={cos:.6f} {'OK' if ok else 'FAIL'}"
                )
        if failed:
            raise SystemExit("parity FAILED")
        print("parity OK")


if __name__ == "__main__":
    main()
