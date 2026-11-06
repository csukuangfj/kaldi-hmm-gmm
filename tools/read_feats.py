# Copyright (c)  2026  Xiaomi Corporation
#
# Data loading for the kaldi-style data layer: features are read with
# kaldi_native_io (ark/scp), transcripts from plain 'text' files.
# No lhotse.
#
# This is tool-side helper code (dev/recipe tooling); it is not part of the
# kaldi_hmm_gmm package, whose Python API only exposes the C++ API.

from typing import Dict, List

try:
    import kaldi_native_io
except ImportError as e:
    raise ImportError(
        "This tool requires kaldi_native_io for reading kaldi ark/scp files.\n"
        "Install it with:\n"
        "  pip install kaldi_native_io"
    ) from e

import numpy as np
import torch

try:
    import kaldi_native_fbank as knf
except ImportError as e:
    raise ImportError(
        "This tool requires kaldi-native-fbank for feature computation.\n"
        "Install it with:\n"
        "  pip install kaldi-native-fbank"
    ) from e


def load_feats(rspecifier: str) -> Dict[str, torch.Tensor]:
    """Load features for all utterances.

    Args:
      rspecifier: a kaldi rspecifier, e.g. "scp:exp/yesno_feats/train_feats.scp"
        or "ark:foo.ark".

    Returns:
      A dict mapping utterance id to a 2-D float32 tensor of shape
      (num_frames, feat_dim).
    """
    ans: Dict[str, torch.Tensor] = {}
    with kaldi_native_io.SequentialFloatMatrixReader(rspecifier) as reader:
        for utt, feats in reader:
            # `feats` is a numpy float32 array
            ans[utt] = torch.from_numpy(feats.copy())
    return ans


def load_feat(rspecifier: str, utt: str) -> torch.Tensor:
    """Load features for a single utterance."""
    with kaldi_native_io.RandomAccessFloatMatrixReader(rspecifier) as reader:
        if utt not in reader:
            raise KeyError(f"{utt} not found in {rspecifier}")
        feats = reader[utt]
    return torch.from_numpy(feats.copy())


def load_text(path: str) -> Dict[str, List[str]]:
    """Load a kaldi 'text' file.

    Returns:
      A dict mapping utterance id to its list of words.
    """
    ans: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            ans[parts[0]] = parts[1:]
    return ans


def load_delta_feats(feats_scp: str, cmvn_scp: str = "") -> Dict[str, torch.Tensor]:
    """kaldi's use-time feature pipe (train_mono.sh's `feats` variable):

      apply-cmvn --utt2spk=... cmvn.scp feats.scp | add-deltas

    i.e. per-utterance CMVN with the speaker stats from cmvn.scp followed by
    deltas (order 2, window 2) -> 39-dim features.

    Args:
      feats_scp: scp of the RAW MFCC (e.g. data/train_yesno/feats.scp).
      cmvn_scp: cmvn scp; defaults to cmvn.scp next to feats_scp.
    """
    from pathlib import Path as _Path

    if not cmvn_scp:
        cmvn_scp = str(_Path(feats_scp).parent / "cmvn.scp")
    raw = load_feats(f"scp:{feats_scp}")
    cmvn = load_feats(f"scp:{cmvn_scp}")
    stats = next(iter(cmvn.values())).numpy().astype(np.float64)
    cmvn_opts = knf.CmvnOptions()
    delta_opts = knf.DeltaOptions(order=2, window=2)
    out: Dict[str, torch.Tensor] = {}
    for utt, x in raw.items():
        f = x.numpy().astype(np.float32)
        f = knf.apply_cmvn(stats, cmvn_opts, f)
        f = knf.compute_deltas(delta_opts, f)
        out[utt] = torch.from_numpy(np.ascontiguousarray(f))
    return out
