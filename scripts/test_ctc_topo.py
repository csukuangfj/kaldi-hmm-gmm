#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

import kaldifst
import graphviz

from ctc_topo import build_standard_ctc_topo, add_one, add_disambig_self_loops

from prepare_lang import (
    Lexicon,
    Lexiconp,
    make_lexicon_fst_no_silence,
)


def test_standard_ctc_topo():
    word2phones = {
        "cat": ["c a t"],
        "ca": ["c a"],
        "at": ["a t"],
        "a": ["a"],
        "cate": ["c a t e"],
    }
    lexicon = Lexicon(word2phones=word2phones)
    lexiconp = Lexiconp.from_lexicon(lexicon)
    lexiconp_disambig = lexiconp.add_lex_disambig()

    id2phone = lexiconp_disambig.id2phone
    max_token_id = lexiconp_disambig.phone2id["#0"] - 1

    H = build_standard_ctc_topo(max_token_id=max_token_id)
    isym = kaldifst.SymbolTable()
    isym.add_symbol(symbol="<blk>", key=0)
    for i in range(1, max_token_id + 1):
        isym.add_symbol(symbol=id2phone[i], key=i)

    osym = kaldifst.SymbolTable()
    osym.add_symbol(symbol="<eps>", key=0)
    for i in range(1, max_token_id + 1):
        osym.add_symbol(symbol=id2phone[i], key=i)

    H.input_symbols = isym
    H.output_symbols = osym

    fst_dot = kaldifst.draw(H, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="standard_ctc_topo.pdf")

    L_disambig = make_lexicon_fst_no_silence(
        lexiconp=lexiconp_disambig,
    )

    fst_dot = kaldifst.draw(L_disambig, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="L_disambig.pdf")

    add_one(H, update_olabel=True)
    add_disambig_self_loops(
        H,
        start=lexiconp_disambig.phone2id["#0"] + 1,
        end=lexiconp_disambig._max_disambig + 1 + lexiconp_disambig.phone2id["#0"],
    )

    fst_dot = kaldifst.draw(H, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="standard_ctc_topo2.pdf")

    add_one(L_disambig, update_olabel=False)

    H.output_symbols = None

    kaldifst.arcsort(H, sort_type="olabel")
    kaldifst.arcsort(L_disambig, sort_type="ilabel")
    HL = kaldifst.compose(H, L_disambig)
    #  kaldifst.rmepsilon(HL)

    fst_dot = kaldifst.draw(HL, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="HL.pdf")


def main():
    test_standard_ctc_topo()


if __name__ == "__main__":
    main()
