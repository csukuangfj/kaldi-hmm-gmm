#!/usr/bin/env python3
"""Decoding + scoring for the yesno recipe.

Mirrors kaldi's steps/decode.sh + local/score.sh: decode the test set with
HCLG (lattice-faster decoder, acoustic-scale 0.1) and report WER.

Writes:
  exp/mono0a/decode_test_yesno/text     (hypotheses)
  exp/mono0a/decode_test_yesno/wer      (score, kaldi-style summary line)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import kaldialign  # noqa: E402
import kaldi_hmm_gmm as khg  # noqa: E402

from read_feats import load_delta_feats, load_feats, load_text  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/test_yesno")
    parser.add_argument("--lang-dir", type=str, default="data/lang")
    parser.add_argument("--feats-scp", type=str, default="data/test_yesno/feats.scp")
    parser.add_argument("--exp-dir", type=str, default="exp/mono0a")
    parser.add_argument("--graph", type=str, default="")
    parser.add_argument("--mdl", type=str, default="", help="model file (default: exp-dir/final.mdl)")
    parser.add_argument("--acoustic-scale", type=float, default=0.083333,
                        help="kaldi steps/decode.sh default acwt")
    parser.add_argument("--beam", type=float, default=13.0)
    parser.add_argument("--lattice-beam", type=float, default=6.0)
    parser.add_argument("--max-active", type=int, default=7000)
    parser.add_argument(
        "--lmwt", type=str, default="7,8,9,10,11,12,13,14,15,16,17",
        help="LM weight sweep (kaldi score.sh min_lmwt..max_lmwt)")
    parser.add_argument(
        "--word-ins-penalty",
        type=str,
        default="0.0,0.5,1.0",
        help="comma-separated word insertion penalties to sweep "
        "(mirrors kaldi local/score.sh); best result is reported",
    )
    return parser.parse_args()


def add_word_ins_penalty(fst, wip: float):
    """kaldi score.sh's lattice-add-penalty equivalent for best-path: add wip
    to every word-emitting arc's weight."""
    if wip == 0.0:
        return fst
    lines = []
    for line in fst.to_str().splitlines():
        parts = line.split()
        if len(parts) == 5:
            s, d, il, ol, w = parts
            if ol != "0":
                w = repr(float(w) + wip)
            lines.append(f"{s} {d} {il} {ol} {w}")
        else:
            lines.append(line)
    import kaldifst

    return kaldifst.compile(s="\n".join(lines) + "\n", acceptor=False)


def read_sym_table(path):
    id2sym = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            s, i = line.split()
            id2sym[int(i)] = s
    return id2sym


def main():
    print("decode.sh: feature type is delta")
    args = get_args()
    exp_dir = Path(args.exp_dir)
    graph_path = Path(args.graph) if args.graph else exp_dir / "graph_tgpr" / "HCLG.fst"
    out_dir = exp_dir / "decode_test_yesno"
    out_dir.mkdir(parents=True, exist_ok=True)

    mdl_path = Path(args.mdl) if args.mdl else exp_dir / "final.mdl"
    tm, am = khg.read_model(str(mdl_path))
    import kaldifst

    graph = kaldifst.StdVectorFst.read(str(graph_path))
    id2word = read_sym_table(Path(args.lang_dir) / "words.txt")

    feats = load_delta_feats(args.feats_scp)
    text = load_text(f"{args.data_dir}/text")
    utts = sorted(feats.keys())

    print(f"local/score.sh --cmd utils/run.pl {args.data_dir} "
          f"{args.lang_dir} {exp_dir / 'decode_test_yesno'}")
    print(f"local/score.sh: scoring with word insertion penalty="
          f"{args.word_ins_penalty}")
    best = None
    for lmwt in [float(x) for x in args.lmwt.split(",")]:
     for wip in [float(x) for x in args.word_ins_penalty.split(",")]:
        g = add_word_ins_penalty(graph.copy(), wip)
        config = khg.LatticeFasterDecoderConfig(
            max_active=args.max_active, beam=args.beam, lattice_beam=args.lattice_beam
        )
        decoder = khg.LatticeFasterDecoder(g, config)

        hyps = {}
        for utt in utts:
            decodable = khg.DecodableAmDiagGmmScaled(
                am=am, tm=tm, feats=feats[utt], scale=args.acoustic_scale
            )
            succeeded, _ali, word_ids, _ll = khg.decode_utterance_lattice_faster(
                decoder=decoder, decodable=decodable, trans_model=tm,
                utt=utt, allow_partial=True, acoustic_scale=args.acoustic_scale,
                    lm_scale=lmwt,
            )
            hyps[utt] = [id2word[w] for w in word_ids if w != 0] if succeeded else []

        r = kaldialign.batch_error_rate([text[u] for u in utts], [hyps[u] for u in utts])
        S, D, I, N = int(r["sub"]), int(r["del"]), int(r["ins"]), int(r["ref_len"])
        wer = 100.0 * (S + D + I) / max(N, 1)
        line = (
            f"%WER {wer:.2f} [ {S + D + I} / {N}, {I} in, {D} del, {S} sub ] "
            f"{out_dir}/wer_{int(lmwt)}_{wip}"
        )
        with open(
            out_dir / f"wer_{int(lmwt)}_{wip}", "w", encoding="utf-8", newline="\n"
        ) as f:
            f.write(line + "\n")
        if best is None or wer < best[0]:
            best = (wer, wip, lmwt, S, D, I, N, hyps)

    wer, wip, lmwt, S, D, I, N, hyps = best
    with open(out_dir / "text", "w", encoding="utf-8", newline="\n") as f:
        for utt, hyp in hyps.items():
            f.write(f"{utt} {' '.join(hyp)}\n")
    line = (f"%WER {wer:.2f} [ {S + D + I} / {N}, {I} in , {D} del, {S} sub ] "
            f"{out_dir}/wer_{int(lmwt)}_{wip}")
    with open(out_dir / "wer", "w", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
