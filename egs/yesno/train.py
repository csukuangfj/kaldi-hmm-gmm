#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve() / "scripts/"))

import kaldi_hmm_gmm as khg
import kaldifst
import lhotse
import torch
from gmm_align_compiled import gmm_align_compiled
from gmm_acc_stats_ali import gmm_acc_stats_ali
from gmm_info import gmm_info
from gmm_init_mono import gmm_init_mono
from prepare_lang import (
    Lexicon,
    Lexiconp,
    generate_hmm_topo,
    make_lexicon_fst_with_silence,
)
from gmm_est import gmm_est
from gmm_boost_silence import gmm_boost_silence


def get_lexicon():
    word2phones = {
        "<SIL>": ["SIL"],
        "YES": ["Y"],
        "NO": ["N"],
    }
    lexicon = Lexicon(word2phones=word2phones)
    return lexicon


def main():
    lexicon = get_lexicon()
    lexiconp = Lexiconp.from_lexicon(lexicon)
    lexiconp_disambig = lexiconp.add_lex_disambig()

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
    num_gauss = info["number_of_gaussians"]
    total_gauss = 1000
    max_iter_inc = 30
    inc_gauss = (total_gauss - num_gauss) // max_iter_inc
    print(info)

    lex_fst = make_lexicon_fst_with_silence(
        lexiconp=lexiconp,
        sil_prob=0.5,
        sil_phone="SIL",
    )
    kaldifst.arcsort(lex_fst, sort_type="olabel")

    training_graph_compiler_opts = khg.TrainingGraphCompilerOptions()

    disambig_syms = [f"#{i}" for i in range(lexiconp_disambig._max_disambig + 1)]
    disambig_syms_ids = [lexiconp_disambig.phone2id[p] for p in disambig_syms]

    gc = khg.TrainingGraphCompiler(
        trans_model=transition_model,
        ctx_dep=tree,
        lex_fst=lex_fst,
        disambig_syms=disambig_syms_ids,
        opts=training_graph_compiler_opts,
    )

    train_graphs = {}
    for c in cuts:
        words = c.supervisions[0].text.split()
        word_ids = [lexiconp_disambig.word2id[w] for w in words]
        fst = gc.compile_graph_from_text(word_ids)
        train_graphs[c.id] = fst

    # for align-equal-compiled
    ali = {}
    for c in cuts:
        succeeded, aligned_fst = kaldifst.equal_align(
            ifst=train_graphs[c.id],
            length=c.features.num_frames,
            rand_seed=3,
            num_retries=10,
        )
        if not succeeded:
            print(f"failed to compute alignment fst for {c.id}")
            continue

        (
            succeeded,
            aligned_seq,
            osymbols_out,
            total_weight,
        ) = kaldifst.get_linear_symbol_sequence(aligned_fst)

        if not succeeded:
            print(f"failed to get alignment for {c.id}")
            continue
        ali[c.id] = aligned_seq

    gmm_accs = khg.AccumAmDiagGmm()
    gmm_accs.init(model=am, flags=khg.GmmUpdateFlags.kGmmAll)

    transition_accs = None

    tot_log_like = 0.0
    for c in cuts:
        feats = torch.from_numpy(c.load_features())

        log_like, transition_accs = gmm_acc_stats_ali(
            am_gmm=am,
            gmm_accs=gmm_accs,
            transition_model=transition_model,
            feats=feats,
            ali=ali[c.id],
            transition_accs=transition_accs,
        )
        tot_log_like += log_like

    print(f"average log_like: {tot_log_like/len(cuts)}")

    tcfg = khg.MleTransitionUpdateConfig()
    gmm_opts = khg.MleDiagGmmOptions()
    gmm_opts.min_gaussian_occupancy = 3

    info = gmm_info(am_gmm=am, transition_model=transition_model)

    gmm_est(
        am_gmm=am,
        gmm_accs=gmm_accs,
        transition_model=transition_model,
        transition_accs=transition_accs,
        tcfg=tcfg,
        gmm_opts=gmm_opts,
        mixup=num_gauss,
        mixdown=0,
        perturb_factor=0.01,
        power=0.2,
        min_count=20.0,
        update_flags="mvwt",
    )

    num_iters = 80
    realign_iters = "1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38 40 42 44 46 49 52 55 58 60 65 70 75 78 79"
    for i in range(num_iters):
        print(f"Pass {i}")
        if str(i) in realign_iters:
            print("Aligning data")
            am = gmm_boost_silence(
                am_gmm=am,
                transition_model=transition_model,
                silence_phones=[lexiconp_disambig.get_sil_phone_id()],
                boost=1.0,
            )

            align_config = khg.AlignConfig()
            align_config.beam = 6.0
            align_config.retry_beam = 40.0
            align_config.careful = False

            for c in cuts:
                feats = torch.from_numpy(c.load_features())
                ans = gmm_align_compiled(
                    am_gmm=am,
                    transition_model=transition_model,
                    utt=c.id,
                    fst=train_graphs[c.id].copy(),
                    feats=feats,
                    align_config=align_config,
                    acoustic_scale=0.1,
                    transition_scale=1.0,
                    self_loop_scale=0.1,
                )
                ali[c.id] = ans["alignment"]

        gmm_accs = khg.AccumAmDiagGmm()
        gmm_accs.init(model=am, flags=khg.GmmUpdateFlags.kGmmAll)

        transition_accs = None

        tot_log_like = 0.0
        for c in cuts:
            feats = torch.from_numpy(c.load_features())

            log_like, transition_accs = gmm_acc_stats_ali(
                am_gmm=am,
                gmm_accs=gmm_accs,
                transition_model=transition_model,
                feats=feats,
                ali=ali[c.id],
                transition_accs=transition_accs,
            )
            tot_log_like += log_like

        print(f"average log_like: {tot_log_like/len(cuts)}")

        gmm_opts = khg.MleDiagGmmOptions()
        gmm_est(
            am_gmm=am,
            gmm_accs=gmm_accs,
            transition_model=transition_model,
            transition_accs=transition_accs,
            tcfg=tcfg,
            gmm_opts=gmm_opts,
            mixup=num_gauss,
            mixdown=0,
            perturb_factor=0.01,
            power=0.2,
            min_count=20.0,
            update_flags="mvwt",
        )
        if i < max_iter_inc:
            num_gauss += inc_gauss
    info = gmm_info(am, transition_model)
    state_dict = {
        "acoustic_model": am,
        "transition_model": transition_model,
        "tree": tree,
    }
    torch.save(state_dict, "checkpoint.pt")
    print(info)


if __name__ == "__main__":
    main()
