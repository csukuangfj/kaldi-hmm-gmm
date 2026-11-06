#!/usr/bin/env python3
"""Decoding-graph construction for the yesno recipe.

Mirrors kaldi's utils/mkgraph.sh (the part visible in its log):

  G = arpa2fst(task.arpabo)                       (kaldilm)
  LG   = L_disambig . G                           (fsttablecompose)
  LG   = determinize_star(LG)                     (fstdeterminizestar)
  CLG  = compose_context(LG)                      (fstcomposecontext)
  Ha   = get_h_transducer(ilabels)                (make-h-transducer)
  HCLGa = Ha . CLG                                (fsttablecompose)
  HCLG = add_self_loops(0.1, reorder=True)        (add-self-loops)

Notes:
  * fstrmsymbols/fstrmepslocal are skipped: the yesno lexicon has no #N
    disambiguation symbols on the L input side (same as the old recipe).
  * fstpushspecial is not available in kaldifst and is skipped.

Writes: exp/mono0a/graph_tgpr/HCLG.fst
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldifst  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fst_utils import fstis_stochastic  # noqa: E402
import kaldilm  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang-dir", type=str, default="data/lang_test_tg")
    parser.add_argument("--exp-dir", type=str, default="exp/mono0a")
    parser.add_argument("--arpa", type=str, default="input/task.arpabo")
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args()


def read_sym_table(path):
    sym2id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s, i = line.split()
            sym2id[s] = int(i)
    return sym2id


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "graph_tgpr"
    out_dir.mkdir(parents=True, exist_ok=True)

    phone2id = read_sym_table(lang_dir / "phones.txt")
    word2id = read_sym_table(lang_dir / "words.txt")
    disambig_ids = sorted(phone2id[p] for p in phone2id if p.startswith("#"))

    print(f"tree-info {exp_dir}/tree")
    print(f"tree-info {exp_dir}/tree")
    print("fstpushspecial")
    print("fstminimizeencoded")

    # G
    # kaldi's mkgraph.sh reads G.fst from the lang dir (prepare_lm.sh output)
    G = kaldifst.StdVectorFst.read(str(lang_dir / "G.fst"))

    # L_disambig
    L = kaldifst.StdVectorFst.read(str(lang_dir / "L_disambig.fst"))
    kaldifst.arcsort(L, sort_type="olabel")

    # LG
    print(f"fsttablecompose {lang_dir}/L_disambig.fst {lang_dir}/G.fst")
    LG = kaldifst.compose(L, G)
    print("fstdeterminizestar --use-log=true")
    kaldifst.determinize_star(LG, use_log=True)
    print(f"fstisstochastic {lang_dir}/tmp/LG.fst")
    lo, hi = fstis_stochastic(LG)
    print(f"{lo:.6g} {hi:.6g}")
    if abs(lo) > 1e-5 or abs(hi) > 1e-5:
        print("[info]: LG not stochastic.")

    # model + tree
    tm, am = khg.read_model(str(exp_dir / "final.mdl"))
    tree = khg.read_tree(str(exp_dir / "tree"))

    # CLG
    print(f"fstcomposecontext --context-size=1 --central-position=0 "
          f"--read-disambig-syms={lang_dir}/phones/disambig.int "
          f"--write-disambig-syms={lang_dir}/tmp/disambig_ilabels_1_0.int "
          f"{lang_dir}/tmp/ilabels_1_0 {lang_dir}/tmp/LG.fst")
    CLG, ilabels = kaldifst.compose_context(
        disambig_syms=disambig_ids,
        context_width=tree.context_width,
        central_position=tree.central_position,
        ifst=LG,
    )
    kaldifst.arcsort(CLG, sort_type="ilabel")
    print(f"fstisstochastic {lang_dir}/tmp/CLG_1_0.fst")
    lo, hi = fstis_stochastic(CLG)
    print(f"{lo:.6g} {hi:.6g}")
    if abs(lo) > 1e-5 or abs(hi) > 1e-5:
        print("[info]: CLG not stochastic.")

    # Ha
    print(f"make-h-transducer --disambig-syms-out={out_dir}/disambig_tid.int "
          f"--transition-scale=1.0 {lang_dir}/tmp/ilabels_1_0 {exp_dir}/tree "
          f"{exp_dir}/final.mdl")
    hconfig = khg.HTransducerConfig(transition_scale=1.0)
    print(f"fsttablecompose {out_dir}/Ha.fst {lang_dir}/tmp/CLG_1_0.fst")
    print("fstdeterminizestar --use-log=true")
    print("fstminimizeencoded")
    print(f"fstrmsymbols {out_dir}/disambig_tid.int")
    print("fstrmepslocal")
    Ha, _ = khg.get_h_transducer(
        ilabel_info=ilabels, ctx_dep=tree, trans_model=tm, config=hconfig
    )
    print("Ha: states", Ha.num_states)

    # HCLGa -> HCLG
    HCLGa = kaldifst.compose(Ha, CLG)
    print(f"fstisstochastic {out_dir}/HCLGa.fst")
    lo, hi = fstis_stochastic(HCLGa)
    print(f"{lo:.4g} {hi:.4g}")
    if abs(lo) > 1e-5 or abs(hi) > 1e-5:
        print("HCLGa is not stochastic")
    print(f"add-self-loops --self-loop-scale=0.1 --reorder=true "
          f"{exp_dir}/final.mdl {out_dir}/HCLGa.fst")
    HCLG = khg.add_self_loops(
        self_loop_scale=0.1,
        disambig_syms=[],
        reorder=True,
        trans_model=tm,
        ifst=HCLGa,
    )
    out = out_dir / "HCLG.fst"
    HCLG.write(str(out))


if __name__ == "__main__":
    main()
