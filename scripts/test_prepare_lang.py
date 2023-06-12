#!/usr/bin/env python3

from prepare_lang import (
    Lexicon,
    Lexiconp,
    make_lexicon_fst_with_silence,
    generate_hmm_topo,
)
import kaldifst
import graphviz
import kaldi_hmm_gmm as khg


def test_lexicon_from_word_phones():
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

    print(lexiconp_disambig.phone2id)
    print(lexiconp_disambig.id2phone)
    print(lexiconp_disambig.word2id)
    print(lexiconp_disambig.id2word)
    print(lexiconp_disambig.get_non_sil_phone_ids())
    print(lexiconp_disambig.get_sil_phone_id())

    fst = make_lexicon_fst_with_silence(lexiconp=lexiconp_disambig, sil_phone="SIL")
    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render("L_with_silence")

    topo = generate_hmm_topo(
        non_sil_phones=lexiconp_disambig.get_non_sil_phone_ids(),
        sil_phone=lexiconp_disambig.get_sil_phone_id(),
    )
    topo_dot = khg.draw_hmm_topology(
        topo, phone=lexiconp_disambig.get_non_sil_phone_ids()[0]
    )
    source = graphviz.Source(topo_dot)
    source.render("topo_non_silence")

    topo_dot = khg.draw_hmm_topology(topo, phone=lexiconp_disambig.get_sil_phone_id())
    source = graphviz.Source(topo_dot)
    source.render("topo_silence")


def main():
    test_lexicon_from_word_phones()


if __name__ == "__main__":
    main()
