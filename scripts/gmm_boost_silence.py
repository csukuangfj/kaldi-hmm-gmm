# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)


from typing import List

import kaldi_hmm_gmm as khg


def gmm_boost_silence(
    am_gmm: khg.AmDiagGmm,
    transition_model: khg.TransitionModel,
    silence_phones: List[int],
    boost: float = 1.5,
) -> khg.AmDiagGmm:
    """
    Returns:
      Return a new khg.AmDiagGmm.
    """
    assert len(silence_phones) > 0, silence_phones
    silence_phones.sort()  # sorted in-place

    is_unique, pdfs = khg.get_pdfs_for_phones(transition_model, silence_phones)
    if not is_unique:
        print(
            "The pdfs for the silence phones may be shared by other phones "
            "(note: this probably does not matter.)"
        )

    # create a copy as we will change am_gmm in-place
    dgm = khg.AmDiagGmm()
    dgm.copy_from_am_diag_gmm(am_gmm)

    for pdf in pdfs:
        gmm = dgm.get_pdf(pdf)

        weights = gmm.weights
        weights.mul_(boost)

        gmm.set_weights(weights)
        gmm.compute_gconsts()

    print("Boosted weights for", len(pdfs), "pdfs, by factor of", boost)

    return dgm
