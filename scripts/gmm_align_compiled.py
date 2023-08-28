# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

from typing import Any, Dict

import kaldi_hmm_gmm as khg
import kaldifst
import torch


def gmm_align_compiled(
    am_gmm: khg.AmDiagGmm,
    transition_model: khg.TransitionModel,
    utt: str,
    fst: kaldifst.StdVectorFst,
    feats: torch.Tensor,
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
    Args:
      acoustic_scale:
        Scaling factor for acoustic likelihoods
      transition_scale:
        Transition-probability scale [relative to acoustics]
      self_loop_scale:
        Scale of self-loop versus non-self-loop log probs [relative to acoustics]
    """

    khg.add_transition_probs(
        trans_model=transition_model,
        transition_scale=transition_scale,
        self_loop_scale=self_loop_scale,
        fst=fst,
    )

    gmm_decodable = khg.DecodableAmDiagGmmScaled(
        am=am_gmm,
        tm=transition_model,
        feats=feats,
        scale=acoustic_scale,
    )

    (
        num_done,
        num_error,
        num_retried,
        tot_like,
        frame_count,
        alignment,
        words,
    ) = khg.align_utterance_wrapper(
        config=align_config,
        utt=utt,
        acoustic_scale=acoustic_scale,
        fst=fst,
        decodable=gmm_decodable,
        num_done=num_done,
        num_error=num_error,
        num_retried=num_retried,
        tot_like=tot_like,
        frame_count=frame_count,
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
