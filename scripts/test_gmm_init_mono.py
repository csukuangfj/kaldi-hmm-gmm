#!/usr/bin/env python3

from gmm_init_mono import gmm_init_mono
from prepare_lang import (
    Lexicon,
    Lexiconp,
    generate_hmm_topo,
    make_lexicon_fst_with_silence,
)
from pathlib import Path
import lhotse
import kaldi_hmm_gmm as khg
import graphviz


def test_gmm_init_mono():
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

    cuts_filename = "./data/fbank/yesno_cuts_train.jsonl.gz"
    if not Path(cuts_filename).is_file():
        print(f"{cuts_filename} does not exist - skipping testing")
        return
    cuts = lhotse.CutSet.from_file(cuts_filename).subset(first=10)

    transition_model, tree, am = gmm_init_mono(topo=topo, cuts=cuts)
    print(transition_model)
    print(tree)
    print(am)

    with open("phones.txt", "w") as f:
        for p, i in lexiconp_disambig.phone2id.items():
            f.write(f"{p} {i}\n")

    tree.write(binary=False, filename="tree")
    dot = khg.draw_tree(phones_txt="phones.txt", tree="./tree")
    source = graphviz.Source(dot)
    source.render("tree", format="pdf")  # It will generate ./tree.pdf


def main():
    test_gmm_init_mono()


if __name__ == "__main__":
    main()
