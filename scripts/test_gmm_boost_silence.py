#!/usr/bin/env python3

from pathlib import Path

import kaldi_hmm_gmm as khg
import lhotse
import torch

from gmm_boost_silence import gmm_boost_silence
from gmm_init_mono import gmm_init_mono
from prepare_lang import (
    Lexicon,
    Lexiconp,
    generate_hmm_topo,
)


def test_gmm_boost_silence():
    word2phones = {
        "foo": ["f o o", "f o o2"],
        "bar": ["b a r"],
        "foobar": ["f o o b a r"],
        "bark": ["b a r k"],
        "<sil>": ["SIL"],
    }
    lexicon = Lexicon(word2phones=word2phones)
    lexiconp = Lexiconp.from_lexicon(lexicon)
    lexiconp_disambig = lexiconp.add_lex_disambig()

    topo = generate_hmm_topo(
        non_sil_phones=lexiconp_disambig.get_non_sil_phone_ids(),
        sil_phone=lexiconp_disambig.get_sil_phone_id(),
    )

    cuts_filename = "./data/fbank/audio_mnist_cuts.jsonl.gz"
    if not Path(cuts_filename).is_file():
        print(f"{cuts_filename} does not exist - skipping testing")
        return
    cuts = lhotse.CutSet.from_file(cuts_filename).subset(first=10)

    transition_model, tree, am = gmm_init_mono(topo=topo, cuts=cuts)

    boost = 1.5

    am_boosted = gmm_boost_silence(
        am_gmm=am,
        transition_model=transition_model,
        silence_phones=[lexiconp_disambig.get_sil_phone_id()],
        boost=boost,
    )

    _, pdfs = khg.get_pdfs_for_phones(
        transition_model, [lexiconp_disambig.get_sil_phone_id()]
    )

    for pdf in pdfs:
        assert torch.allclose(
            torch.from_numpy(am.get_pdf(pdf).weights) * boost,
            torch.from_numpy(am_boosted.get_pdf(pdf).weights),
        )


def main():
    test_gmm_boost_silence()


if __name__ == "__main__":
    main()
