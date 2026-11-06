# Copyright (c)  2023  Xiaomi Corporation
#
# Python equivalents of kaldi's gmm-init-mono, gmm-acc-stats-ali, gmm-est,
# gmm-boost-silence and gmm-info.  Ported from this repo's scripts/gmm_*.py;
# feature input is plain tensors (no lhotse).

from typing import Dict, List, Optional, Tuple, Union

import kaldi_hmm_gmm as khg
import torch


def _first_n_feats(
    feats: Union[Dict[str, torch.Tensor], List[torch.Tensor]], n: int
) -> List[torch.Tensor]:
    """First n utterances in order (mirrors kaldi's `subset-feats --n=...`)."""
    if isinstance(feats, dict):
        mats = list(feats.values())
    else:
        mats = list(feats)
    return mats[:n] if n > 0 else mats


def gmm_init_mono(
    topo: khg.HmmTopology,
    feats: Union[Dict[str, torch.Tensor], List[torch.Tensor]],
    shared_phones: Optional[List[List[int]]] = None,
    perturb_factor: float = 0.0,
    num_init_utts: int = 10,
) -> Tuple[khg.TransitionModel, khg.ContextDependency, khg.AmDiagGmm]:
    """Initialize a monophone GMM (kaldi's gmm-init-mono equivalent).

    Args:
      topo:
        The HMM topology.
      feats:
        Feature matrices; used to compute the global mean/variance for
        initializing the Gaussians.  If a dict, utterance order is the
        insertion order (scp order).  Only the first `num_init_utts`
        utterances are used (mirrors kaldi's
        `--train-feats="$feats subset-feats --n=10"`); pass 0 to use all.
      shared_phones:
        Optional. If not none, phones in the same sublist share the same pdfs.
      perturb_factor:
        If not zero, we perturb the means of the resulting gaussians.
      num_init_utts:
        Number of leading utterances used for the global statistics.
    """
    mats = _first_n_feats(feats, num_init_utts)
    assert len(mats) > 0

    # kaldi computes the statistics in double precision.
    count = 0.0
    feat_dim = mats[0].shape[1]
    mean_stats = torch.zeros(feat_dim, dtype=torch.float64)
    var_stats = torch.zeros(feat_dim, dtype=torch.float64)
    for mat in mats:
        m = mat.double()
        count += m.shape[0]
        mean_stats += m.sum(dim=0)
        var_stats += (m * m).sum(dim=0)
    mean_stats /= count
    var_stats /= count
    var_stats -= mean_stats * mean_stats  # E[x^2] - (E[x])^2
    assert var_stats.min() > 0, "bad variance"

    means = mean_stats.float().unsqueeze(0)  # (1, feat_dim)
    variances = var_stats.float().unsqueeze(0)

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


def gmm_acc_stats_ali(
    am_gmm: khg.AmDiagGmm,
    gmm_accs: khg.AccumAmDiagGmm,
    transition_model: khg.TransitionModel,
    feats: torch.Tensor,
    ali: List[int],
    transition_accs: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor]:
    """
    Args:
      am_gmm:
        The acoustic model.
      gmm_accs:
        Stats accumulator. It is changed in place.
      transition_model:
        The transition model.
      feats:
        A 2-D float tensor of shape (num_frames, feat_dim).
      ali:
        A list of transition IDs.
      transition_accs:
        Optional. If not None, it is a 1-D double tensor of shape
        (num_transition_ids+1,). It is changed in place if not None.

    Returns:
      A tuple containing:
        - log_like, the total log-like of the input feature frames
        - transition_accs. If the input transition_accs is None,
          a new one is created.
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
        transition_accs = transition_model.accumulate(
            prob=1.0, trans_id=tid, stats=transition_accs
        )
        log_like += gmm_accs.accumulate_for_gmm(
            model=am_gmm, data=feat_frame, gmm_index=pdf_id, weight=1
        )

    return log_like, transition_accs


def gmm_est(
    am_gmm: khg.AmDiagGmm,
    gmm_accs: khg.AccumAmDiagGmm,
    transition_model: khg.TransitionModel,
    transition_accs: torch.Tensor,
    tcfg: khg.MleTransitionUpdateConfig,
    gmm_opts: khg.MleDiagGmmOptions,
    mixup: int = 0,
    mixdown: int = 0,
    perturb_factor: float = 0.01,
    power: float = 0.2,
    min_count: float = 20.0,
    update_flags: str = "mvwt",
    verbose: bool = True,
) -> torch.Tensor:
    """kaldi's gmm-est equivalent (updates models in place).

    Returns:
      pdf_occs: 1-D float tensor of per-pdf occupancies (for --write-occs).
    """
    update_flags = khg.str_to_gmm_flags(update_flags)

    if int(update_flags) & int(khg.GmmUpdateFlags.kGmmTransitions):
        objf_impr, count = transition_model.mle_update(transition_accs, tcfg)
        if verbose:
            print(
                "Transition model update: Overall",
                objf_impr / count,
                "log-like improvement per frame over",
                count,
                "frames.",
            )

    tot_like = gmm_accs.tot_log_like
    tot_t = gmm_accs.tot_count

    objf_impr, count = khg.mle_am_diag_gmm_update(
        config=gmm_opts,
        amdiag_gmm_acc=gmm_accs,
        flags=update_flags,
        am_gmm=am_gmm,
    )

    if verbose:
        print(
            "GMM update: Overall",
            objf_impr / count,
            "objective function improvement per frame over",
            count,
            "frames",
        )
        print(
            "GMM update: Overall avg like per frame =",
            tot_like / tot_t,
            "over",
            tot_t,
            "frames.",
        )

    pdf_occs = []
    for i in range(gmm_accs.num_accs):
        pdf_occs.append(gmm_accs.get_acc(i).occupancy.sum().item())
    pdf_occs = torch.tensor(pdf_occs, dtype=torch.float32)

    if mixdown != 0:
        am_gmm.merge_by_count(
            state_occs=pdf_occs,
            target_components=mixdown,
            power=power,
            min_count=min_count,
        )

    if mixup != 0:
        am_gmm.split_by_count(
            state_occs=pdf_occs,
            target_components=mixup,
            perturb_factor=perturb_factor,
            power=power,
            min_count=min_count,
        )

    return pdf_occs


def gmm_boost_silence(
    am_gmm: khg.AmDiagGmm,
    transition_model: khg.TransitionModel,
    silence_phones: List[int],
    boost: float = 1.5,
    verbose: bool = True,
) -> khg.AmDiagGmm:
    """
    Returns:
      A new khg.AmDiagGmm with silence pdf weights boosted.
    """
    assert len(silence_phones) > 0, silence_phones
    silence_phones = sorted(silence_phones)

    is_unique, pdfs = khg.get_pdfs_for_phones(transition_model, silence_phones)
    if not is_unique and verbose:
        print(
            "The pdfs for the silence phones may be shared by other phones "
            "(note: this probably does not matter.)"
        )

    # create a copy as we will change am_gmm in-place
    dgm = khg.AmDiagGmm()
    dgm.copy_from_am_diag_gmm(am_gmm)

    for pdf in pdfs:
        gmm = dgm.get_pdf(pdf)

        weights = torch.from_numpy(gmm.weights)
        weights.mul_(boost)

        gmm.set_weights(weights)
        gmm.compute_gconsts()

    if verbose:
        print("Boosted weights for", len(pdfs), "pdfs, by factor of", boost)

    return dgm


def gmm_info(
    am_gmm: khg.AmDiagGmm,
    transition_model: khg.TransitionModel,
) -> Dict[str, int]:
    ans = {}
    ans["number_of_phones"] = len(transition_model.phones)
    ans["number_of_pdfs"] = transition_model.num_pdfs
    ans["number_of_transition_ids"] = transition_model.num_transition_ids
    ans["number_of_transition_states"] = transition_model.num_transition_states
    ans["feature_dimension"] = am_gmm.dim
    ans["number_of_gaussians"] = am_gmm.num_gauss

    return ans
