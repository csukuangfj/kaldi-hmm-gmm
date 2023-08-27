# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

from typing import List, Optional, Tuple

import kaldi_hmm_gmm as khg
import lhotse
import torch


def gmm_init_mono(
    topo: khg.HmmTopology,
    cuts: lhotse.CutSet,
    shared_phones: Optional[List[List[int]]] = None,
    perturb_factor: float = 0.0,
) -> Tuple[khg.TransitionModel, khg.ContextDependency, khg.AmDiagGmm]:
    """
    Args:
      topo:
        The HMM topology.
      cuts:
        It's used to compute the mean and variance for initializing Gaussians.
      shared_phones:
        Optional. If not none, phones in the same sublist share the same pdfs
      perturb_factor:
        If not zero, we perturb and mean of the resulting gaussians.
    """
    stats = cuts.compute_global_feature_stats()

    # means is a 1-d tensor of (feature_dim,)
    means = stats["norm_means"]

    # stddev is a 1-d tensor of (feat_dim,)
    stddev = stats["norm_stds"]

    means = torch.from_numpy(means).unsqueeze(0)
    variances = torch.from_numpy(stddev).square().unsqueeze(0)

    feat_dim = means.shape[1]

    if shared_phones is None:
        tree = khg.monophone_context_dependency(
            phones=topo.phones,
            phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
        )
    else:
        tree = khg.monophone_context_dependency_shared(
            phone_classes=shared_phones,
            phone2num_pdf_classes=topo.get_phone_to_num_pdf_classes(),
        )

    diag_gmm = khg.DiagGmm(nmix=1, dim=feat_dim)

    weights = torch.ones(1, dtype=torch.float32)

    diag_gmm.set_weights(weights)
    diag_gmm.set_means(means)
    diag_gmm.set_invvars(1 / variances)
    diag_gmm.compute_gconsts()

    num_pdfs = tree.num_pdfs

    am = khg.AmDiagGmm()
    for i in range(num_pdfs):
        # It will copy diag_gmm on the C++ side
        am.add_pdf(diag_gmm)

    if perturb_factor != 0:
        for i in range(num_pdfs):
            am.get_pdf(i).perturb(perturb_factor)

    transition_model = khg.TransitionModel(ctx_dep=tree, hmm_topo=topo)

    return transition_model, tree, am
