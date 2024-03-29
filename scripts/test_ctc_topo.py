#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (author: Fangjun Kuang)

import graphviz
import kaldifst
import torch

from ctc_topo import add_disambig_self_loops, add_one, build_standard_ctc_topo
from prepare_lang import (
    Lexicon,
    Lexiconp,
    make_lexicon_fst_no_silence,
    make_lexicon_fst_with_silence,
)

import numpy as np

from kaldi_hmm_gmm import DecodableInterface, FasterDecoder, FasterDecoderOptions


class CtcDecodable(DecodableInterface):
    def __init__(self, nnet_output: torch.Tensor):
        DecodableInterface.__init__(self)
        assert nnet_output.ndim == 2, nnet_output.shape
        self.nnet_output = nnet_output

    def log_likelihood(self, frame: int, index: int) -> float:
        return self.nnet_output[frame][index - 1].item()

    def is_last_frame(self, frame: int) -> bool:
        return frame == self.nnet_output.shape[0] - 1

    def num_frames_ready(self) -> int:
        return self.nnet_output.shape[0]

    def num_indices(self) -> int:
        return self.nnet_output.shape[1]


def build_G():
    G = kaldilm.arpa2fst(
        input_arpa="./G.txt", disambig_symbol="#0", read_symbol_table="./words.txt"
    )
    isym = kaldifst.SymbolTable.read_text("./words.txt")
    osym = kaldifst.SymbolTable.read_text("./words.txt")
    fst = kaldifst.compile(
        s=G,
        acceptor=False,
        keep_isymbols=True,
        keep_osymbols=True,
    )
    fst.input_symbols = isym
    fst.output_symbols = osym

    kaldifst.arcsort(fst, sort_type="ilabel")

    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="G.svg")

    return fst


def test_standard_ctc_topo():
    word2phones = {
        "<SIL>": ["SIL"],
        "YES": ["Y"],
        "NO": ["N"],
        "<UNK>": ["SIL"],
    }
    lexicon = Lexicon(word2phones=word2phones)
    lexiconp = Lexiconp.from_lexicon(lexicon)
    lexiconp_disambig = lexiconp.add_lex_disambig()

    phone2id = {"<eps>": 0, "N": 1, "SIL": 2, "Y": 3, "#0": 4, "#1": 5, "#2": 6}
    id2phone = {i: p for p, i in phone2id.items()}

    lexiconp_disambig._phone2id = phone2id
    lexiconp_disambig._id2phone = id2phone
    del id2phone
    del phone2id

    id2phone = lexiconp_disambig.id2phone

    max_token_id = lexiconp_disambig.phone2id["#0"] - 1

    print(lexiconp_disambig.phone2id)
    print(lexiconp_disambig.word2id)

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

    L_disambig = make_lexicon_fst_with_silence(
        lexiconp=lexiconp_disambig,
    )

    fst_dot = kaldifst.draw(L_disambig, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="L_disambig.pdf")

    add_one(H, treat_ilabel_zero_specially=False)
    add_disambig_self_loops(
        H,
        start=lexiconp_disambig.phone2id["#0"] + 1,
        end=lexiconp_disambig._max_disambig + 1 + lexiconp_disambig.phone2id["#0"],
    )

    fst_dot = kaldifst.draw(H, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="standard_ctc_topo2.pdf")

    add_one(L_disambig, treat_ilabel_zero_specially=True)

    H.output_symbols = None

    kaldifst.arcsort(H, sort_type="olabel")
    kaldifst.arcsort(L_disambig, sort_type="ilabel")
    HL = kaldifst.compose(H, L_disambig)
    #  kaldifst.rmepsilon(HL)

    fst_dot = kaldifst.draw(HL, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="HL.pdf")

    state_dict = torch.load("./nnet_output.pt")
    print(list(state_dict.keys()))
    wave_filename = state_dict["wave"]
    nnet_output = state_dict["nnet_output"]
    print(wave_filename)
    argmax = nnet_output[0].argmax(dim=-1)
    hyps = torch.unique_consecutive(argmax)
    hyps = hyps[hyps != 0]
    hyps = [id2phone[i] for i in hyps.tolist()]
    hyps = [s for s in hyps if s != "SIL"]
    print(hyps)

    decodable = CtcDecodable(nnet_output[0])
    decoder_opts = FasterDecoderOptions()
    decoder = FasterDecoder(HL, decoder_opts)
    decoder.decode(decodable)
    if not decoder.reached_final():
        print("failed to decode")
        return
    ok, best_path = decoder.get_best_path()

    if not ok:
        print("failed to get best path")
        return

    (
        ok,
        isymbols_out,
        osymbols_out,
        total_weight,
    ) = kaldifst.get_linear_symbol_sequence(best_path)
    if not ok:
        print("failed to get linear symbol sequence")
        return

    hyps = [lexiconp_disambig.id2word[i - 1] for i in osymbols_out]
    print(hyps)


def main():
    test_standard_ctc_topo()


if __name__ == "__main__":
    main()
