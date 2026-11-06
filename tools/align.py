# Copyright (c)  2023  Xiaomi Corporation
#
# Python equivalent of kaldi's gmm-align-compiled.
# Ported from this repo's scripts/gmm_align_compiled.py.

from typing import Any, Dict

import kaldi_hmm_gmm as khg
import kaldifst
import numpy as np


def gmm_align_compiled(
    am_gmm: khg.AmDiagGmm,
    transition_model: khg.TransitionModel,
    utt: str,
    fst: kaldifst.StdVectorFst,
    feats: np.ndarray,
    align_config: khg.AlignConfig,
    acoustic_scale: float = 1.0,
    transition_scale: float = 1.0,
    self_loop_scale: float = 1.0,
    num_done: int = 0,
    num_error: int = 0,
    num_retried: int = 0,
    tot_like: float = 0,
    frame_count: int = 0,
) -> Dict[str, Any]:
    """
    NOTE: this MUTATES `fst` in place (transition probs are added to the
    caller's graph object; same behavior as kaldi's gmm-align-compiled).
    Pass a copy if you need to reuse the graph afterwards.

    Args:
      acoustic_scale:
        Scaling factor for acoustic likelihoods.
      transition_scale:
        Transition-probability scale [relative to acoustics].
      self_loop_scale:
        Scale of self-loop versus non-self-loop log probs [relative to
        acoustics].
    """

    gmm_decodable = khg.DecodableAmDiagGmmScaled(
        am=am_gmm,
        tm=transition_model,
        feats=feats,
        scale=acoustic_scale,
    )

    # NOTE: AddTransitionProbs + alignment run in ONE C++ call on purpose:
    # FSTs cross the python boundary via a copy (see fst-caster.h), so
    # mutating the FST in a separate call would not be visible here.
    (
        num_done,
        num_error,
        num_retried,
        tot_like,
        frame_count,
        alignment,
        words,
    ) = khg.align_utterance(
        config=align_config,
        utt=utt,
        acoustic_scale=acoustic_scale,
        transition_scale=transition_scale,
        self_loop_scale=self_loop_scale,
        trans_model=transition_model,
        fst=fst,
        decodable=gmm_decodable,
    )

    return {
        "num_done": num_done,
        "num_error": num_error,
        "num_retried": num_retried,
        "tot_like": tot_like,
        "frame_count": frame_count,
        "alignment": alignment,
        "words": words,
    }
