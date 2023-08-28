#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "scripts/"))

import lhotse
from prepare_lang import (
    Lexicon,
    Lexiconp,
    generate_hmm_topo,
    make_lexicon_fst_with_silence,
)
from gmm_init_mono import gmm_init_mono
from gmm_info import gmm_info


def get_lexicon():
    word2phones = {
        "<SIL>": ["SIL"],
        "YES": ["Y"],
        "NO": ["N"],
    }
    lexicon = Lexicon(word2phones=word2phones)
    lexiconp = Lexiconp.from_lexicon(lexicon)
    lexiconp_disambig = lexiconp.add_lex_disambig()
    return lexiconp_disambig


def main():
    lexiconp_disambig = get_lexicon()
    topo = generate_hmm_topo(
        non_sil_phones=lexiconp_disambig.get_non_sil_phone_ids(),
        sil_phone=lexiconp_disambig.get_sil_phone_id(),
    )

    cuts_filename = "./data/fbank/yesno_cuts_train.jsonl.gz"
    if not Path(cuts_filename).is_file():
        raise RuntimeError(f"{cuts_filename} does not exist")
    cuts = lhotse.CutSet.from_file(cuts_filename).subset(first=10)
    transition_model, tree, am = gmm_init_mono(topo=topo, cuts=cuts.subset(first=10))
    info = gmm_info(am, transition_model)
    print(info)


if __name__ == "__main__":
    main()
