#!/usr/bin/env python3
"""Compare this repo's per-frame phone alignment against kaldi's.

Both sides are mapped to phone SYMBOLS before comparison (the two
implementations use different phone-id numbering).

Usage:
  python3 tests/parity/compare_ali.py \
      --ours exp/mono0a/ali_phones.ark \
      --kaldi-dir /path/to/kaldi/egs/yesno/s5 --kaldi-exp exp/mono0a
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))

import kaldi_native_io  # noqa: E402

from ali_to_phones import read_sym_table  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=str, default="exp/mono0a/ali_phones.ark")
    parser.add_argument("--ours-phones-txt", type=str, default="data/lang/phones.txt")
    parser.add_argument("--kaldi-dir", type=str, default="../../../kaldi/egs/yesno/s5")
    parser.add_argument("--kaldi-exp", type=str, default="exp/mono0a")
    parser.add_argument("--report-dir", type=str, default="exp/parity")
    return parser.parse_args()


def collapse(seq):
    """Collapse a per-frame sequence to phone instances WITHOUT merging
    adjacent instances of the same phone... (we cannot distinguish instances
    from a per-frame sequence alone, so this yields phone runs; boundary
    comparison is done frame-wise below.)"""
    out = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out


def main():
    args = get_args()
    kaldi_dir = str(Path(args.kaldi_dir).resolve())
    kaldi_exp = args.kaldi_exp
    if not Path(kaldi_exp).is_dir():
        kaldi_exp = str(Path(kaldi_dir) / kaldi_exp)

    ours_id2sym = read_sym_table(args.ours_phones_txt)
    kaldi_id2sym = read_sym_table(f"{kaldi_dir}/data/lang/phones.txt")

    ours = {}
    with kaldi_native_io.SequentialInt32VectorReader(f"ark,t:{args.ours}") as reader:
        for utt, phones in reader:
            ours[utt] = [ours_id2sym[p] for p in phones]

    cmd = (
        f'cd "{kaldi_dir}" && source ./path.sh && '
        f'ali-to-phones --per-frame=true {kaldi_exp}/final.mdl '
        f'"ark:gunzip -c {kaldi_exp}/ali.1.gz|" ark,t:-'
    )
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
    ref = {}
    for line in out.splitlines():
        parts = line.split()
        if parts:
            ref[parts[0]] = [kaldi_id2sym[int(x)] for x in parts[1:]]

    assert set(ours) == set(ref), sorted(set(ours) ^ set(ref))[:5]

    lines = []
    total_frames = 0
    same_frames = 0
    for utt in sorted(ours):
        a, b = ours[utt], ref[utt]
        assert len(a) == len(b), (utt, len(a), len(b))
        total_frames += len(a)
        same_frames += sum(1 for x, y in zip(a, b) if x == y)
        if collapse(a) != collapse(b):
            lines.append(
                f"  {utt}: runs ours={collapse(a)} kaldi={collapse(b)}"
            )

    frame_rate = same_frames / total_frames * 100
    msg = (
        f"frame-level agreement: {same_frames}/{total_frames} = {frame_rate:.2f}%"
    )
    print(msg)
    n_run_mismatch = len(lines)
    print(f"utterances with different phone runs: {n_run_mismatch}/{len(ours)}")
    for line in lines[:10]:
        print(line)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "compare_ali.report").write_text(
        msg + "\n"
        + f"utterances with different phone runs: {n_run_mismatch}/{len(ours)}\n"
        + "\n".join(lines)
        + "\n"
    )
    print(f"report written to {report_dir / 'compare_ali.report'}")


if __name__ == "__main__":
    main()
