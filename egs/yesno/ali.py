#!/usr/bin/env python3
"""Forced alignment for the yesno recipe (the ali-to-phones deliverable).

For each utterance: compile the training graph from its transcript, run
gmm-align-compiled with the trained model, and export

  exp/mono0a/ali_phones.ark        per-frame phone-ids  (ark,t:, like
                                   `ali-to-phones --per-frame=true ... ark,t:-`)
  exp/mono0a/ali_phones_sym.txt    same with symbols (like utils/int2sym.pl)
  exp/mono0a/ali_phones.ark        collapsed phone sequence
  exp/mono0a/ali_full.ark          the transition-id alignment itself
  exp/mono0a/ali_stats.txt         log-like / error summary

kaldi counterpart:
  ali-to-phones --per-frame=true exp/mono0a/final.mdl \\
      "ark:gunzip -c exp/mono0a/ali.1.gz|" ark,t:- | utils/int2sym.pl -f 2-
"""

import argparse
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldifst  # noqa: E402
import kaldi_native_io  # noqa: E402

from align import gmm_align_compiled  # noqa: E402
from ali_to_phones import ali_to_phones, int2sym  # noqa: E402
from gmm import gmm_boost_silence  # noqa: E402
from read_feats import load_delta_feats, load_feats, load_text  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/train_yesno")
    parser.add_argument("--lang-dir", type=str, default="data/lang")
    parser.add_argument("--feats-scp", type=str, default="data/train_yesno/feats.scp")
    parser.add_argument("--exp-dir", type=str, default="exp/mono0a")
    parser.add_argument("--beam", type=float, default=10.0)
    parser.add_argument("--retry-beam", type=float, default=40.0)
    parser.add_argument("--acoustic-scale", type=float, default=0.1)
    parser.add_argument("--transition-scale", type=float, default=1.0)
    parser.add_argument("--self-loop-scale", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = get_args()
    exp_dir = Path(args.exp_dir)

    text = load_text(f"{args.data_dir}/text")
    feats = load_delta_feats(args.feats_scp)

    # lang
    lang_dir = Path(args.lang_dir)
    lex_fst = kaldifst.StdVectorFst.read(str(lang_dir / "L.fst"))
    phone2id = {}
    with open(lang_dir / "phones.txt", encoding="utf-8") as f:
        for line in f:
            p, i = line.split()
            phone2id[p] = int(i)
    id2phone = {i: p for p, i in phone2id.items()}
    word2id = {}
    with open(lang_dir / "words.txt", encoding="utf-8") as f:
        for line in f:
            w, i = line.split()
            word2id[w] = int(i)
    disambig_ids = sorted(phone2id[p] for p in phone2id if p.startswith("#"))

    # model + tree
    tm, am = khg.read_model(str(exp_dir / "final.mdl"))
    tree = khg.read_tree(str(exp_dir / "tree"))

    gc = khg.TrainingGraphCompiler(
        trans_model=tm,
        ctx_dep=tree,
        lex_fst=lex_fst,
        disambig_syms=disambig_ids,
        opts=khg.TrainingGraphCompilerOptions(transition_scale=0.0, self_loop_scale=0.0),
    )

    align_config = khg.AlignConfig()
    align_config.beam = args.beam
    align_config.retry_beam = args.retry_beam
    align_config.careful = False

    with contextlib.redirect_stdout(open(Path(args.exp_dir) / "log" / "ali_boost.log", "w")):
        boosted = gmm_boost_silence(
        am_gmm=am,
        transition_model=tm,
        silence_phones=[phone2id["SIL"]],
        boost=1.0,
        verbose=False,
    )

    utts = sorted(feats.keys())
    all_ali = {}
    all_phones_pf = {}
    all_phones = {}
    tot_like = 0.0
    n_err = 0
    for utt in utts:
        words = text[utt]
        word_ids = [word2id[w] for w in words]
        fst = gc.compile_graph_from_text(word_ids)
        ans = gmm_align_compiled(
            am_gmm=boosted,
            transition_model=tm,
            utt=utt,
            fst=fst,
            feats=feats[utt],
            align_config=align_config,
            acoustic_scale=args.acoustic_scale,
            transition_scale=args.transition_scale,
            self_loop_scale=args.self_loop_scale,
        )
        if ans["num_error"] != 0:
            n_err += 1
            print(f"warning: alignment failed for {utt}")
            continue
        ali = ans["alignment"]
        tot_like += ans["tot_like"]
        all_ali[utt] = ali
        all_phones_pf[utt] = ali_to_phones(tm, ali, per_frame=True)
        all_phones[utt] = ali_to_phones(tm, ali, per_frame=False)
        assert len(all_phones_pf[utt]) == feats[utt].shape[0]

    # ---- write outputs (kaldi ark,t format) ----
    with kaldi_native_io.Int32VectorWriter(f"ark,t:{exp_dir}/ali_full.ark") as w:
        for utt, ali in all_ali.items():
            w.write(utt, ali)
    with kaldi_native_io.Int32VectorWriter(f"ark,t:{exp_dir}/ali_phones.ark") as w:
        for utt, phones in all_phones_pf.items():
            w.write(utt, phones)
    with kaldi_native_io.Int32VectorWriter(f"ark,t:{exp_dir}/ali_seq.ark") as w:
        for utt, phones in all_phones.items():
            w.write(utt, phones)

    with open(exp_dir / "ali_phones_sym.txt", "w", encoding="utf-8") as f:
        for utt, phones in all_phones_pf.items():
            syms = int2sym(phones, id2phone)
            f.write(f"{utt} {' '.join(syms)}\n")

    n_done = len(all_ali)
    stats = (
        f"num_done={n_done} num_error={n_err} "
        f"tot_log_like={tot_like:.2f} avg_log_like={tot_like/max(n_done,1):.2f}\n"
    )
    (exp_dir / "ali_stats.txt").write_text(stats, encoding="utf-8")

    # head-style preview (as in todo.md)
    print(f"{exp_dir}/ali_phones.ark (first 5 lines):")
    for i, (utt, phones) in enumerate(all_phones_pf.items()):
        if i >= 5:
            break
        print(f"{utt} {' '.join(map(str, phones[:20]))} ... ({len(phones)} frames)")
    print(stats.strip())


if __name__ == "__main__":
    main()
