#!/usr/bin/env python3
from pathlib import Path

import graphviz
import kaldi_hmm_gmm as khg
import kaldifst
import lhotse
import torch

from gmm_acc_stats_ali import gmm_acc_stats_ali
from gmm_init_mono import gmm_init_mono
from prepare_lang import (
    Lexicon,
    Lexiconp,
    generate_hmm_topo,
    make_lexicon_fst_with_silence,
)


def test_gmm_acc_stats_ali():
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

    gmm_accs = khg.AccumAmDiagGmm()
    gmm_accs.init(model=am, flags=khg.GmmUpdateFlags.kGmmAll)

    s = ["bar", "foo", "bark"]
    words = [lexiconp_disambig.word2id[w] for w in s]
    fst = gc.compile_graph_from_text(words)

    # for align-equal-compiled
    num_feature_frames = cuts[0].load_features().shape[0]
    succeeded, aligned_fst = kaldifst.equal_align(
        ifst=fst, length=num_feature_frames, rand_seed=3, num_retries=10
    )
    assert succeeded is True

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

    feats = torch.from_numpy(cuts[0].load_features())

    # For the first call, transition_accs is set to None
    log_like, transition_accs = gmm_acc_stats_ali(
        am_gmm=am,
        gmm_accs=gmm_accs,
        transition_model=transition_model,
        feats=feats,
        ali=aligned_seq,
        transition_accs=None,
    )
    assert transition_accs.sum() == feats.shape[0]
    print(log_like, log_like / len(aligned_seq))

    # for the second one

    s = ["foo", "bar"]
    words = [lexiconp_disambig.word2id[w] for w in s]
    fst = gc.compile_graph_from_text(words)

    # for align-equal-compiled
    num_feature_frames = cuts[1].load_features().shape[0]
    succeeded, aligned_fst = kaldifst.equal_align(
        ifst=fst, length=num_feature_frames, rand_seed=3, num_retries=10
    )
    assert succeeded is True

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

    feats = torch.from_numpy(cuts[1].load_features())

    # For the first call, transition_accs is set to None
    log_like_2, transition_accs = gmm_acc_stats_ali(
        am_gmm=am,
        gmm_accs=gmm_accs,
        transition_model=transition_model,
        feats=feats,
        ali=aligned_seq,
        transition_accs=transition_accs,
    )
    assert (
        transition_accs.sum()
        == cuts[0].load_features().shape[0] + cuts[1].load_features().shape[0]
    )
    log_like += log_like_2
    print(log_like, log_like / transition_accs.sum())


def main():
    test_gmm_acc_stats_ali()


if __name__ == "__main__":
    main()
