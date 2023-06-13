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

    # L_disambig.fst
    fst = make_lexicon_fst_with_silence(
        lexiconp=lexiconp_disambig,
        sil_prob=0.5,
        sil_phone="SIL",
        sil_disambig=lexiconp_disambig.phone2id[
            f"#{lexiconp_disambig._max_disambig + 1}"
        ],
    )

    kaldifst.add_self_loops(
        fst,
        isyms=[lexiconp_disambig.phone2id["#0"]],
        osyms=[lexiconp_disambig.word2id["#0"]],
    )
    kaldifst.arcsort(fst, sort_type="olabel")
    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="L_disambig.pdf")

    # Now for fst without disambig symbols (used for training)
    # L.fst
    # Note: It does not invoke add_self_loops()
    fst = make_lexicon_fst_with_silence(
        lexiconp=lexiconp,
        sil_prob=0.5,
        sil_phone="SIL",
    )
    kaldifst.arcsort(fst, sort_type="olabel")
    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="L.pdf")

    topo = generate_hmm_topo(
        non_sil_phones=lexiconp_disambig.get_non_sil_phone_ids(),
        sil_phone=lexiconp_disambig.get_sil_phone_id(),
    )
    topo_dot = khg.draw_hmm_topology(
        topo, phone=lexiconp_disambig.get_non_sil_phone_ids()[0]
    )
    topo_dot.render(filename="topo_non_silence", format="pdf", cleanup=True)

    topo_dot = khg.draw_hmm_topology(topo, phone=lexiconp_disambig.get_sil_phone_id())
    topo_dot.render(filename="topo_silence", format="pdf", cleanup=True)


def main():
    test_lexicon_from_word_phones()


if __name__ == "__main__":
    main()
