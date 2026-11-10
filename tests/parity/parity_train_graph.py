#!/usr/bin/env python3
"""Training-graph parity: this repo's TrainingGraphCompiler vs kaldi's
compile-train-graphs binary, on identical tree/model/L.fst/transcripts.

Compares each compiled graph ARC-BY-ARC (src, dst, ilabel, olabel, weight,
final weights), exactly.  Graphs must be compiled with the compile-train-graphs
semantics: transition_scale=0.0, self_loop_scale=0.0 (kaldi
src/bin/compile-train-graphs.cc sets both to 0.0; transition probs are added
later, at alignment time).

Usage (from egs/yesno):
  python3 ../../tests/parity/parity_train_graph.py
Exit status 0 iff all graphs match exactly.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldifst  # noqa: E402

from read_feats import load_text  # noqa: E402


def read_sym(path):
    d = {}
    for line in open(path):
        s, i = line.split()
        d[s] = int(i)
    return d


def arcs_of(fst_text):
    """to_str() lines -> sorted (src, dst, ilabel, olabel, weight) list.

    Layout is openfst's FstPrinter: "src dest ilabel olabel [w]" plus
    "src [w]" final lines; weight omitted when 1.0 (= zero cost).  Values
    are compared rounded to 1e-6.
    """
    out = []
    for line in fst_text.strip().splitlines():
        f = line.split()
        if len(f) == 5:
            out.append((int(f[0]), int(f[1]), int(f[2]), int(f[3]), round(float(f[4]), 6)))
        elif len(f) == 4:
            out.append((int(f[0]), int(f[1]), int(f[2]), int(f[3]), 0.0))
        elif len(f) == 2:
            out.append((int(f[0]), -1, 0, 0, round(float(f[1]), 6)))
        elif len(f) == 1:
            # zero-weight final state: must be compared too (a missing final
            # state on either side is a real difference).
            out.append((int(f[0]), -1, 0, 0, 0.0))
    return sorted(out)


def read_fst_ark(path):
    d = {}
    for k, v in kaldifst.SequentialVectorFstReader(f"ark:{path}"):
        d[k] = v.copy().to_str()  # reader views dangle after iteration
    return d


def main():
    kaldi_dir = Path("../../../kaldi/egs/yesno/s5").resolve()
    kaldi_bin = str(Path("../../../kaldi/src/bin/compile-train-graphs").resolve())

    tm, _ = khg.read_model(str(kaldi_dir / "exp/mono0a/0.mdl"))
    tree = khg.read_tree(str(kaldi_dir / "exp/mono0a/tree"))
    word2id = read_sym("data/lang/words.txt")
    phone2id = read_sym("data/lang/phones.txt")
    disambig_ids = sorted(phone2id[p] for p in phone2id if p.startswith("#"))
    lex = kaldifst.StdVectorFst.read("data/lang/L.fst")
    text = load_text("data/train_yesno/text")
    utts = sorted(text)

    # ours
    gc = khg.TrainingGraphCompiler(
        trans_model=tm, ctx_dep=tree, lex_fst=lex,
        disambig_syms=disambig_ids,
        opts=khg.TrainingGraphCompilerOptions(transition_scale=0.0, self_loop_scale=0.0),
    )
    ours = {}
    for u in utts:
        ours[u] = gc.compile_graph_from_text([word2id[w] for w in text[u]]).to_str()

    # kaldi
    text_int = "/tmp/parity_tg_text.int"
    with open(text_int, "w") as f:
        for u in utts:
            f.write(u + " " + " ".join(str(word2id[w]) for w in text[u]) + "\n")
    kaldi_ark = "/tmp/parity_tg_kaldi.ark"
    cmd = [
        kaldi_bin,
        f"--read-disambig-syms=data/lang/phones/disambig.int",
        f"{kaldi_dir}/exp/mono0a/tree",
        f"{kaldi_dir}/exp/mono0a/0.mdl",
        "data/lang/L.fst",
        f"ark:{text_int}",
        f"ark:{kaldi_ark}",
    ]
    subprocess.run(cmd, check=True)
    theirs = read_fst_ark(kaldi_ark)

    n_bad = 0
    for u in utts:
        a = arcs_of(ours[u])
        b = arcs_of(theirs[u])
        if a != b:
            n_bad += 1
            print(f"MISMATCH {u}: ours {len(a)} entries, kaldi {len(b)} entries")
            for x, y in zip(a, b):
                if x != y:
                    print("  ours :", x)
                    print("  kaldi:", y)
                    break
        else:
            print(f"OK {u}: {len(a)} entries identical")

    print(f"\n{len(utts) - n_bad}/{len(utts)} training graphs identical to kaldi")
    if n_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
