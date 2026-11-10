#!/usr/bin/env python3
"""Single-step gmm-align-compiled parity: same model + graphs + features to
kaldi's binary and to this repo's gmm_align_compiled; compare transition-id
alignments EXACTLY.

Usage (from egs/yesno):
  python3 ../../tests/parity/parity_align.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldifst  # noqa: E402
import kaldi_native_io  # noqa: E402

from align import gmm_align_compiled  # noqa: E402
from read_feats import load_delta_feats, load_feats, load_text  # noqa: E402


def read_sym(path):
    d = {}
    for line in open(path):
        s, i = line.split()
        d[s] = int(i)
    return d


def main():
    kaldi_dir = str(Path("../../../kaldi/egs/yesno/s5").resolve())
    exp = str(Path("exp/mono0a").resolve())
    kexp = f"{kaldi_dir}/exp/mono0a"

    tm, am = khg.read_model(f"{kexp}/0.mdl")
    tree = khg.read_tree(f"{kexp}/tree")
    word2id = read_sym("data/lang/words.txt")
    phone2id = read_sym("data/lang/phones.txt")
    disambig_ids = sorted(phone2id[p] for p in phone2id if p.startswith("#"))
    lex = kaldifst.StdVectorFst.read("data/lang/L.fst")

    gc = khg.TrainingGraphCompiler(
        trans_model=tm, ctx_dep=tree, lex_fst=lex,
        disambig_syms=disambig_ids,
        # compile-train-graphs semantics (kaldi src/bin/compile-train-graphs.cc
        # sets both scales to 0.0); transition probs are added at alignment.
        opts=khg.TrainingGraphCompilerOptions(transition_scale=0.0, self_loop_scale=0.0),
    )
    text = load_text("data/train_yesno/text")
    feats = load_delta_feats("data/train_yesno/feats.scp")
    utts = sorted(feats)[:8]  # a representative subset is enough to catch bugs

    graphs = {u: gc.compile_graph_from_text([word2id[w] for w in text[u]]) for u in utts}
    fsts_ark = "/tmp/parity_align_fsts.ark"
    with kaldifst.VectorFstWriter(f"ark,scp:{fsts_ark},/tmp/parity_align_fsts.scp") as w:
        for u in utts:
            w.write(u, graphs[u])
    feats_ark = "/tmp/parity_align_feats.ark"
    with kaldi_native_io.FloatMatrixWriter(
        f"ark,scp:{feats_ark},/tmp/parity_align_feats.scp"
    ) as w:
        for u in utts:
            w.write(u, feats[u])

    # kaldi: gmm-align-compiled
    ali_out = "/tmp/parity_align_kaldi.ali"
    Path(ali_out).unlink(missing_ok=True)
    cmd = (
        f'cd {kaldi_dir} && source ./path.sh && gmm-align-compiled '
        f'--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1 '
        f'--beam=6.0 --retry-beam=40.0 {kexp}/0.mdl "ark:/tmp/parity_align_fsts.ark" '
        f'"scp:/tmp/parity_align_feats.scp" "ark,t:{ali_out}"'
    )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("kaldi gmm-align-compiled failed:", r.stderr[-400:])
        return
    kaldi_ali = {}
    with kaldi_native_io.SequentialInt32VectorReader(f"ark,t:{ali_out}") as rd:
        for k, v in rd:
            kaldi_ali[k] = v

    # ours
    cfg = khg.AlignConfig()
    cfg.beam = 6.0
    cfg.retry_beam = 40.0
    cfg.careful = False
    same = 0
    for u in utts:
        ans = gmm_align_compiled(
            am_gmm=am, transition_model=tm, utt=u, fst=graphs[u].copy(),
            feats=feats[u], align_config=cfg, acoustic_scale=0.1,
            transition_scale=1.0, self_loop_scale=0.1,
        )
        ours = ans["alignment"]
        ref = kaldi_ali[u]
        if ours == ref:
            same += 1
        else:
            # first difference position
            pos = next((i for i, (x, y) in enumerate(zip(ours, ref)) if x != y), -1)
            print(f"{u}: MISMATCH len {len(ours)} vs {len(ref)}, first diff at frame {pos}")
    print(f"exact alignment match: {same}/{len(utts)}")
    if same == len(utts):
        print("ALIGN PARITY: bit-exact (up to the same RNG state)")
    else:
        print("ALIGN PARITY: mismatch -> real divergence in Viterbi/decodable")


if __name__ == "__main__":
    main()
