# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)


from typing import Dict

import kaldi_hmm_gmm as khg


def gmm_info(
    am_gmm: khg.AmDiagGmm,
    transition_model: khg.TransitionModel,
) -> Dict[str, int]:
    #  print("number of phones", len(transition_model.phones))
    #  print("number of pdfs", transition_model.num_pdfs)
    #  print("number of transition ids", transition_model.num_transition_ids)
    #  print("number of transition states", transition_model.num_transition_states)
    #  print("feature dimension", am_gmm.dim)
    #  print("number of gaussians", am_gmm.num_gauss)

    ans = {}
    ans["number_of_phones"] = len(transition_model.phones)
    ans["number_of_pdfs"] = transition_model.num_pdfs
    ans["number_of_transition_ids"] = transition_model.num_transition_ids
    ans["number_of_transition_states"] = transition_model.num_transition_states
    ans["feature_dimensition"] = am_gmm.dim
    ans["number_of_gaussians"] = am_gmm.num_gauss

    return ans
