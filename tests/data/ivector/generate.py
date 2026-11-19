#!/usr/bin/env python3
"""Generate the synthetic i-vector model fixtures in this directory.

Produces a tiny, deterministic kaldi-format model that both kaldi-hmm-gmm and
the kaldi binaries (gmm-global-get-post, ivector-extract-online) can read:

  final.ie            extractor (D=8, S=4, I=16), kaldi binary format
  final.dubm          UBM DiagGmm (16 Gauss, dim 8), kaldi binary format
  final.mat           lda-like transform [8, 57] (affine: 56 = 8*(3+1+3) + 1)
  global_cmvn.stats   [2, 9] text matrix
  mfcc.conf           8-cepstrum config so raw feats match the fixture dims
  splice.conf         --left-context=3 --right-context=3

Everything is derived from a fixed integer seed; re-running reproduces the
files byte-for-byte.

Usage:  python3 generate.py
"""

import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

D = 8        # feature dim (after lda; equals raw dim)
S = 4        # ivector dim
I = 16       # UBM Gaussians
L = R = 3    # splice context -> spliced dim = 8 * 7 = 56
PRIOR_OFFSET = 2.0

rng = np.random.default_rng(20261119)


# ---- kaldi binary writers (token + space; size-prefixed basic types) -------

def w_token(f, s):
    f.write(s.encode() + b" ")


def w_int(f, v):
    f.write(bytes([4]) + struct.pack("<i", v))


def w_double(f, v):
    f.write(bytes([8]) + struct.pack("<d", v))


def w_vector(f, tag, v, fmt):
    w_token(f, tag)
    w_token(f, "DV" if fmt == "d" else "FV")
    w_int(f, len(v))
    f.write(np.asarray(v, dtype=np.float64 if fmt == "d" else np.float32).tobytes())


def w_matrix(f, tag, m, fmt):
    if tag:
        w_token(f, tag)
    w_token(f, "DM" if fmt == "d" else "FM")
    m = np.asarray(m)
    w_int(f, m.shape[0])
    w_int(f, m.shape[1])
    f.write(np.asarray(m, dtype=np.float64 if fmt == "d" else np.float32).tobytes())


def w_packed(f, tag, m):
    """Packed lower-triangular symmetric double matrix ("DP")."""
    if tag:
        w_token(f, tag)
    w_token(f, "DP")
    n = m.shape[0]
    w_int(f, n)
    data = [m[r, c] for r in range(n) for c in range(r + 1)]
    f.write(np.asarray(data, dtype=np.float64).tobytes())


def w_text_matrix(path, m):
    lines = [" ["]
    for row in np.asarray(m):
        lines.append("  " + " ".join(f"{x:.8e}" for x in row))
    lines.append("]")
    path.write_text("\n".join(lines) + "\n")


# ---- fixture data ---------------------------------------------------------

def main():
    # UBM: uniform weights, reasonable inverse variances / means
    weights = np.full(I, 1.0 / I)
    inv_vars = rng.uniform(0.5, 2.0, (I, D))
    means = rng.standard_normal((I, D))
    means_invvars = means * inv_vars

    # extractor: small subspaces, diagonal-dominant inverse covariances
    M = rng.standard_normal((I, D, S)) * 0.3
    sigma_inv = []
    for i in range(I):
        m = np.diag(1.0 + rng.uniform(0.0, 1.0, D))
        m += np.full((D, D), 0.05)
        sigma_inv.append(m)
    w_vec = weights.copy()

    # final.dubm  (float; gconsts are recomputed on load)
    with open(HERE / "final.dubm", "wb") as f:
        f.write(b"\x00B")
        w_token(f, "<DiagGMM>")
        w_vector(f, "<GCONSTS>", np.zeros(I), "f")
        w_vector(f, "<WEIGHTS>", w_vec, "f")
        w_matrix(f, "<MEANS_INVVARS>", means_invvars, "f")
        w_matrix(f, "<INV_VARS>", inv_vars, "f")
        w_token(f, "</DiagGMM>")

    # final.ie  (double; empty w, no weight projection)
    with open(HERE / "final.ie", "wb") as f:
        f.write(b"\x00B")
        w_token(f, "<IvectorExtractor>")
        w_matrix(f, "<w>", np.zeros((0, 0)), "d")
        w_vector(f, "<w_vec>", w_vec, "d")
        w_token(f, "<M>")
        w_int(f, I)
        for i in range(I):
            w_matrix(f, "", M[i], "d")
        w_token(f, "<SigmaInv>")
        for i in range(I):
            w_packed(f, "", sigma_inv[i])
        w_token(f, "<IvectorOffset>")
        w_double(f, PRIOR_OFFSET)
        w_token(f, "</IvectorExtractor>")

    # final.mat: affine transform of the spliced frame (offset column)
    lda = rng.standard_normal((D, D * (L + R + 1) + 1)) * 0.2
    lda[:, -1] = 0.1  # affine offset
    w_text_matrix(HERE / "final.mat", lda)

    # global cmvn stats: zero mean, unit variance, 500 frames
    stats = np.zeros((2, D + 1))
    stats[0, D] = 500.0
    stats[1, :D] = 500.0
    w_text_matrix(HERE / "global_cmvn.stats", stats)

    (HERE / "mfcc.conf").write_text(
        "--use-energy=false\n"
        "--sample-frequency=16000\n"
        "--num-mel-bins=8\n"
        "--num-ceps=8\n"
        "--low-freq=40\n"
        "--high-freq=-200\n"
        "--allow-upsample=true\n"
        "--allow-downsample=true\n")
    (HERE / "splice.conf").write_text(
        "--left-context=3\n--right-context=3\n")
    print(f"fixtures written to {HERE}")


if __name__ == "__main__":
    main()
