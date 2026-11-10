#!/usr/bin/env python3
"""Parity check: this repo's ali_to_phones vs kaldi's ali-to-phones binary.

Uses kaldi-generated artifacts (final.mdl + ali.1.gz) as the common input.

Usage:
  python3 tests/parity/parity_ali_to_phones.py \
      --kaldi-dir /path/to/kaldi/egs/yesno/s5 \
      --exp-dir exp/mono0a

Compares, for every utterance:
  * per-frame phone sequences  (kaldi: ali-to-phones --per-frame=true)
  * collapsed phone sequences  (kaldi: ali-to-phones default mode)

Report goes to exp/parity/ali_to_phones.report (and stdout).
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldi_native_io  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaldi-dir",
        type=str,
        default="../../../kaldi/egs/yesno/s5",
        help="kaldi yesno recipe dir (source of final.mdl / ali.1.gz)",
    )
    parser.add_argument("--exp-dir", type=str, default="exp/mono0a")
    parser.add_argument(
        "--report-dir", type=str, default="exp/parity", help="where to write the report"
    )
    return parser.parse_args()


def run_kaldi_ali_to_phones(kaldi_dir: str, exp_dir: str, per_frame: bool):
    """Returns {utt: [phone_id, ...]} using kaldi's ali-to-phones binary."""
    flag = "--per-frame=true" if per_frame else "--per-frame=false"
    cmd = (
        f'cd "{kaldi_dir}" && source ./path.sh && '
        f'ali-to-phones {flag} {exp_dir}/final.mdl '
        f'"ark:gunzip -c {exp_dir}/ali.1.gz|" ark,t:-'
    )
    out = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=True
    ).stdout
    ans = {}
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        ans[parts[0]] = [int(x) for x in parts[1:]]
    return ans


def ours_ali_to_phones(tm, exp_dir: str, per_frame: bool):
    """Returns {utt: [phone_id, ...]} using this repo's implementation."""
    ans = {}
    with kaldi_native_io.SequentialInt32VectorReader(
        f"ark:gunzip -c {exp_dir}/ali.1.gz|"
    ) as reader:
        for utt, ali in reader:
            if per_frame:
                ans[utt] = [tm.transition_id_to_phone(t) for t in ali]
            else:
                ok, split = khg.split_to_phones(tm, ali)
                if not ok:
                    print(f"warning: incomplete alignment for {utt}")
                ans[utt] = [tm.transition_id_to_phone(seg[0]) for seg in split]
    return ans


def compare(name, ref, hyp):
    total = 0
    mismatched = 0
    details = []
    assert set(ref.keys()) == set(hyp.keys()), (
        f"{name}: utt sets differ",
        sorted(set(ref) ^ set(hyp))[:5],
    )
    for utt in sorted(ref):
        r, h = ref[utt], hyp[utt]
        total += 1
        if r != h:
            mismatched += 1
            n_diff = sum(1 for a, b in zip(r, h) if a != b) + abs(len(r) - len(h))
            details.append(f"  {utt}: len ref={len(r)} hyp={len(h)} diff={n_diff}")
    return mismatched, total, details


def main():
    args = get_args()
    kaldi_dir = str(Path(args.kaldi_dir).resolve())
    exp_dir = args.exp_dir
    if not Path(exp_dir).is_dir():
        exp_dir = str(Path(kaldi_dir) / args.exp_dir)

    tm = khg.TransitionModel()
    tm.read(f"{exp_dir}/final.mdl")

    report_lines = []
    all_ok = True
    for per_frame in (True, False):
        mode = "per-frame" if per_frame else "collapsed"
        ref = run_kaldi_ali_to_phones(kaldi_dir, exp_dir, per_frame)
        hyp = ours_ali_to_phones(tm, exp_dir, per_frame)
        mismatched, total, details = compare(mode, ref, hyp)
        status = "OK" if mismatched == 0 else "FAIL"
        if mismatched:
            all_ok = False
        msg = f"[{status}] {mode}: {total - mismatched}/{total} utterances match"
        print(msg)
        report_lines.append(msg)
        report_lines.extend(details[:20])

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ali_to_phones.report"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"report written to {report_path}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
