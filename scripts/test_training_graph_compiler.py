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
import kaldifst
import graphviz


def test_training_graph_compiler():
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

    lex_fst = make_lexicon_fst_with_silence(
        lexiconp=lexiconp,
        sil_prob=0.5,
        sil_phone="SIL",
    )
    kaldifst.arcsort(lex_fst, sort_type="olabel")

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

    opts = khg.TrainingGraphCompilerOptions()
    #  opts.reorder = True # default is True

    disambig_syms = [f"#{i}" for i in range(lexiconp_disambig._max_disambig + 1)]
    disambig_syms_ids = [lexiconp_disambig.phone2id[p] for p in disambig_syms]

    gc = khg.TrainingGraphCompiler(
        trans_model=transition_model,
        ctx_dep=tree,
        lex_fst=lex_fst,
        disambig_syms=disambig_syms_ids,
        opts=opts,
    )
    s = ["bar", "foo", "bark"]
    words = [lexiconp_disambig.word2id[w] for w in s]
    fst = gc.compile_graph_from_text(words)

    phone2id = lexiconp_disambig.phone2id
    id2phone = lexiconp_disambig.id2phone
    isym = kaldifst.SymbolTable()
    isym.add_symbol("<eps>", 0)
    for i in range(1, transition_model.num_transition_ids + 1):
        phone_id = transition_model.transition_id_to_phone(i)
        isym.add_symbol(f"{id2phone[phone_id]}_{i-1}", i)
    fst.input_symbols = isym

    osym = kaldifst.SymbolTable()
    for w, i in lexiconp_disambig.word2id.items():
        osym.add_symbol(w, i)
    fst.output_symbols = osym

    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="transcript.pdf", cleanup=True)

    # for align-equal-compiled
    num_feature_frames = 100
    succeeded, aligned_fst = kaldifst.equal_align(
        ifst=fst, length=num_feature_frames, rand_seed=3, num_retries=10
    )
    assert succeeded is True
    aligned_fst.input_symbols = isym
    aligned_fst.output_symbols = osym

    aligned_fst_dot = kaldifst.draw(aligned_fst, acceptor=False, portrait=True)
    source = graphviz.Source(aligned_fst_dot)
    source.render(outfile="aligned.pdf", cleanup=True)

    (
        succeeded,
        aligned_seq,
        osymbols_out,
        total_weight,
    ) = kaldifst.get_linear_symbol_sequence(aligned_fst)
    assert succeeded is True
    assert len(aligned_seq) == num_feature_frames
    id2word = lexiconp_disambig.id2word
    assert [lexiconp_disambig.id2word[i] for i in osymbols_out] == s
    print(aligned_seq)


def main():
    test_training_graph_compiler()


if __name__ == "__main__":
    main()
