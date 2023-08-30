#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "scripts/"))

import re
from pathlib import Path

import graphviz
import kaldi_hmm_gmm as khg
import kaldifst
import kaldilm
import lhotse
import torch
from prepare_lang import Lexicon, Lexiconp, make_lexicon_fst_with_silence
from utils import write_error_stats

from train import get_lexicon


def build_L():
    lexicon = get_lexicon()
    lexiconp = Lexiconp.from_lexicon(lexicon)
    lexiconp_disambig = lexiconp.add_lex_disambig()

    fst = make_lexicon_fst_with_silence(
        lexiconp=lexiconp_disambig,
        sil_prob=0.5,
        sil_phone="SIL",
    )
    kaldifst.arcsort(fst, sort_type="olabel")

    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="L_disambig.svg")

    word2id = lexiconp_disambig.word2id
    with open("words.txt", "w") as f:
        for w, i in word2id.items():
            f.write(f"{w} {i}\n")

    phone2id = lexiconp_disambig.phone2id
    with open("phones.txt", "w") as f:
        for p, i in phone2id.items():
            f.write(f"{p} {i}\n")

    return fst


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


def main():
    L_disambig = build_L()
    G = build_G()
    LG = kaldifst.compose(L_disambig, G)

    fst_dot = kaldifst.draw(LG, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="LG.svg")

    kaldifst.determinize_star(LG, use_log=True)

    disambig_phone_ids = []
    pattern = re.compile(r"^#\d+$")
    with open("phones.txt") as f:
        for line in f:
            p, i = line.split()
            if pattern.match(p):
                disambig_phone_ids.append(int(i))

    state_dict = torch.load("./checkpoint.pt")
    am = state_dict["acoustic_model"]
    tm = state_dict["transition_model"]
    tree = state_dict["tree"]

    CLG, ilabels = kaldifst.compose_context(
        disambig_syms=disambig_phone_ids,
        context_width=tree.context_width,
        central_position=tree.central_position,
        ifst=LG,
    )

    fst_dot = kaldifst.draw(CLG, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="CLG.svg")

    kaldifst.arcsort(CLG, sort_type="ilabel")
    hconfig = khg.HTransducerConfig(transition_scale=1.0)

    Ha, _ = khg.get_h_transducer(
        ilabel_info=ilabels,
        ctx_dep=tree,
        trans_model=tm,
        config=hconfig,
    )

    fst_dot = kaldifst.draw(Ha, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="Ha.svg")

    # Now: we omit fstrmsymbols and fstrmepslocal here
    # since the yesno dataset does not have disambig_syms

    HCLGa = kaldifst.compose(Ha, CLG)

    fst_dot = kaldifst.draw(HCLGa, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="HCLGa.svg")

    HCLG = khg.add_self_loops(
        self_loop_scale=1.0, disambig_syms=[], reorder=False, trans_model=tm, ifst=HCLGa
    )
    HCLG.write("HCLG.fst")

    fst_dot = kaldifst.draw(HCLG, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="HCLG.svg")

    config = khg.LatticeFasterDecoderConfig(
        max_active=7000,
        beam=13.0,
        lattice_beam=6.0,
    )
    allow_partial = True
    decoder = khg.LatticeFasterDecoder(HCLG, config)

    name = "test"
    cuts_filename = f"./data/fbank/yesno_cuts_{name}.jsonl.gz"
    if not Path(cuts_filename).is_file():
        raise RuntimeError(f"{cuts_filename} does not exist")

    sym = kaldifst.SymbolTable.read_text("./words.txt")

    #  cuts = lhotse.CutSet.from_file(cuts_filename).subset(first=10)
    cuts = lhotse.CutSet.from_file(cuts_filename)
    ans = []
    for c in cuts:
        decodable = khg.DecodableAmDiagGmmScaled(
            am=am,
            tm=tm,
            feats=torch.from_numpy(c.load_features()),
            scale=2.0,
        )
        (
            succeeded,
            alignment_ids,
            word_ids,
            log_like,
        ) = khg.decode_utterance_lattice_faster(
            decoder=decoder,
            decodable=decodable,
            trans_model=tm,
            utt=c.id,
            allow_partial=allow_partial,
        )
        if not succeeded:
            print("Failed to decode")
            ans.append([c.id, c.supervisions[0].text.split(), []])
            continue
        hyp = [sym.find(i) for i in word_ids]

        ans.append([c.id, c.supervisions[0].text.split(), hyp])

    with open(f"{name}-err.txt", "w") as f:
        write_error_stats(f, name, ans)


if __name__ == "__main__":
    main()
