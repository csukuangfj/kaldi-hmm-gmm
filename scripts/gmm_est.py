# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)


import kaldi_hmm_gmm as khg
import torch


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
) -> None:
    """
    Args:
      mixup:
        Increase number of mixture components to this overall target.
      mixdown:
        If nonzero, merge mixture components to this target.
      perturb_factor:
        While mixing up, perturb means by standard deviation times this factor.
      power:
        If mixing up, power to allocate Gaussians to  states.
      min_count:
        Minimum per-Gaussian count enforced while mixing up and down.
      update_flags:
        Which GMM parameters to update: subset of mvwt.
    """
    update_flags = khg.str_to_gmm_flags(update_flags)

    if int(update_flags) & int(khg.GmmUpdateFlags.kGmmTransitions):
        objf_impr, count = transition_model.mle_update(transition_accs, tcfg)
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

    if mixup != 0 or mixdown != 0:
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
