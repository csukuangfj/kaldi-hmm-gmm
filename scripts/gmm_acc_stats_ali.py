# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

from typing import List, Optional

import kaldi_hmm_gmm as khg
import torch


def gmm_acc_stats_ali(
    am_gmm: khg.AmDiagGmm,
    gmm_accs: khg.AccumAmDiagGmm,
    transition_model: khg.TransitionModel,
    feats: torch.Tensor,
    ali: List[int],
    transition_accs: Optional[torch.Tensor] = None,
) -> [float, torch.Tensor]:
    """
    Args:
      am_gmm:
        The acoustic model.
      gmm_accs:
        Stats accumulator. It is changed in place.
      transition_model:
        The transition model.
      feats:
        A 2-D float tensor of shape (num_frame, feat_dim).
      ali:
        A list of transition IDs.
      transition_accs:
        Optional. If not None, it is a 1-D double tensor of shape
        (num_transition_ids+1,). It is changed in place if not None.

    Returns:
      Return a tuple containing:
        - loglike, the total loglike of the input feature frames
        - transition_accs. If the input transition_accs is None,
          we will create a new one
    """
    assert feats.ndim == 2, feats.shape
    assert len(ali) == feats.shape[0], (len(ali), feats.shape[0])

    if transition_accs is None:
        transition_accs = transition_model.init_stats()

    log_like = 0.0
    for i in range(len(ali)):
        tid = ali[i]
        feat_frame = feats[i]

        pdf_id = transition_model.transition_id_to_pdf(tid)
        transition_model.accumulate(prob=1.0, trans_id=tid, stats=transition_accs)
        log_like += gmm_accs.accumulate_for_gmm(
            model=am_gmm, data=feat_frame, gmm_index=pdf_id, weight=1
        )

    return log_like, transition_accs
