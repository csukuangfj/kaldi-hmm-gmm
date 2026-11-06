#!/usr/bin/env python3
"""Single-step gmm-est parity: same inputs to kaldi binaries and to this
repo's implementation, compare the updated model per-pdf.

Round 1 (mixup=0): pure MLE math, no RNG — must match to float precision.
Round 2 (mixup=N): exercises DiagGmm::Split RNG — exposes RandGauss stream
mismatches (call count/order).

Usage (from egs/yesno):
  python3 ../../tests/parity/parity_gmm_est.py \
      --kaldi-dir ../../../kaldi/egs/yesno/s5 --exp-dir exp/mono0a
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldi_native_io  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gmm import gmm_acc_stats_ali, gmm_est  # noqa: E402
from read_feats import load_delta_feats, load_feats  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi-dir", default="../../../kaldi/egs/yesno/s5")
    p.add_argument("--exp-dir", default="exp/mono0a")
    p.add_argument("--feats-scp", default="data/train_yesno/feats.scp")
    p.add_argument("--mixup", type=int, default=40)
    return p.parse_args()


def read_ali(path_gz):
    ali = {}
    with kaldi_native_io.SequentialInt32VectorReader(
        f"ark:gunzip -c {path_gz}|"
    ) as r:
        for k, v in r:
            ali[k] = v
    return ali


def run_kaldi(kaldi_dir, exp, feats_scp, ali_txt, mixup):
    """gmm-acc-stats-ali + gmm-est via kaldi binaries; returns stdout."""
    src = Path(kaldi_dir) / "src"
    env_cmds = f"cd {kaldi_dir} && source ./path.sh &&"
    acc = f"/tmp/parity_gmm.acc"
    out_mdl = f"/tmp/parity_gmm_{mixup}.mdl"
    occs = f"/tmp/parity_gmm_{mixup}.occs"
    Path(out_mdl).unlink(missing_ok=True)
    scp_abs = str(Path(feats_scp).resolve())
    ali_abs = str(Path(ali_txt).resolve())
    cmd = (
        f'{env_cmds} gmm-acc-stats-ali {exp}/0.mdl '
        f'"scp:{scp_abs}" "ark:{ali_abs}" {acc} && '
        f'gmm-est --power=0.25 --min-count=20.0 --mix-up={mixup} '
        f'--write-occs={occs} --min-gaussian-occupancy=10 '
        f'{exp}/0.mdl {acc} {out_mdl}'
    )
    if mixup == 0:
        cmd = (
            f'{env_cmds} gmm-acc-stats-ali {exp}/0.mdl '
            f'"scp:{scp_abs}" "ark:{ali_abs}" {acc} && '
            f'gmm-est --power=0.25 --min-count=20.0 '
            f'--min-gaussian-occupancy=10 '
            f'{exp}/0.mdl {acc} {out_mdl}'
        )
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stderr + r.stdout, out_mdl, occs


def run_ours(exp, feats_scp, ali, mixup):
    tm, am = khg.read_model(f"{exp}/0.mdl")
    feats = load_delta_feats(feats_scp)
    gmm_accs = khg.AccumAmDiagGmm()
    gmm_accs.init(model=am, flags=khg.GmmUpdateFlags.kGmmAll)
    tacc = None
    for utt in sorted(feats):
        _, tacc = gmm_acc_stats_ali(am, gmm_accs, tm, feats[utt], ali[utt], tacc)
    tcfg = khg.MleTransitionUpdateConfig()
    opts = khg.MleDiagGmmOptions()  # min_gaussian_occupancy=10 default
    occs = gmm_est(
        am_gmm=am, gmm_accs=gmm_accs, transition_model=tm, transition_accs=tacc,
        tcfg=tcfg, gmm_opts=opts, mixup=mixup, mixdown=0, perturb_factor=0.01,
        power=0.25, min_count=20.0, update_flags="mvwt", verbose=False,
    )
    out = f"/tmp/parity_gmm_ours_{mixup}.mdl"
    khg.write_model(out, tm, am, binary=True)
    return out, occs.numpy()


def model_arrays(path):
    tm, am = khg.read_model(path)
    ws, ms, vs = [], [], []
    for i in range(am.num_pdfs):
        g = am.get_pdf(i)
        ws.append(np.array(g.weights, dtype=np.float64).copy())
        ms.append(np.array(g.means, dtype=np.float64).copy())
        vs.append(np.array(g.inv_vars, dtype=np.float64).copy())
    return ws, ms, vs


def compare(tag, a, b):
    global gmax
    gmax = [0.0]
    """a, b: (ws, ms, vs) lists per pdf."""
    ok = True
    if len(a[0]) != len(b[0]):
        print(f"[{tag}] pdf count differs: {len(a[0])} vs {len(b[0])}")
        return False
    for i, (wa, wb) in enumerate(zip(a[0], b[0])):
        if wa.shape != wb.shape:
            print(f"[{tag}] pdf {i}: nmix {wa.shape} vs {wb.shape}")
            ok = False
            continue
        dw = np.abs(wa - wb).max() if wa.size else 0.0
        dm = np.abs(a[1][i] - b[1][i]).max() if a[1][i].size else 0.0
        dv = np.abs(a[2][i] - b[2][i]).max() if a[2][i].size else 0.0
        if max(dw, dm, dv) > 1e-12:
            print(f"[{tag}] pdf {i}: max|dw|={dw:.3e} max|dm|={dm:.3e} max|dv|={dv:.3e}")
            ok = False
        gmax[0] = max(gmax[0], dw, dm, dv)
    print(f"[{tag}] max abs diff over all params: {gmax[0]:.3e}")
    return ok


def main():
    args = get_args()
    kaldi_dir = str(Path(args.kaldi_dir).resolve())
    exp = str(Path(args.exp_dir).resolve())
    ali = read_ali(f"{exp}/ali.1.gz")

    # write ali as kaldi text ark for the kaldi binaries
    ali_txt = "/tmp/parity_gmm_ali.txt"
    with kaldi_native_io.Int32VectorWriter(f"ark,t:{ali_txt}") as w:
        for k, v in ali.items():
            w.write(k, v)

    for mixup in (0, args.mixup):
        tag = f"mixup={mixup}"
        rc, log, kmdl, koccs = run_kaldi(kaldi_dir, exp, args.feats_scp, ali_txt, mixup)
        if rc != 0:
            print(f"[{tag}] kaldi failed:\n{log[-500:]}")
            continue
        omdl, ooccs = run_ours(exp, args.feats_scp, ali, mixup)
        ok = compare(tag, model_arrays(kmdl), model_arrays(omdl))
        print(f"[{tag}] {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            print(f"[{tag}] (kaldi log tail): {log.strip().splitlines()[-1] if log.strip() else ''}")


if __name__ == "__main__":
    main()
