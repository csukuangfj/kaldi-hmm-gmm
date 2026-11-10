#!/usr/bin/env python3
"""Monophone training for the yesno recipe.

Mirrors ../kaldi/steps/train_mono.sh step by step, using this repo's Python
API (kaldi_hmm_gmm) instead of kaldi binaries.  The output layout under
--exp-dir follows kaldi's, e.g. for --exp-dir exp/mono0a:

  0.mdl, 1.mdl ... (intermediate mdls are removed like train_mono.sh does),
  40.mdl, final.mdl, 40.occs, final.occs, tree, fsts.1.gz, ali.1.gz, log/

Features come from a kaldi feats.scp (pre-computed, 39-dim); no lhotse.
"""

import argparse
import contextlib
import gzip
import shutil
import sys
import time
from pathlib import Path

_src_python = Path(__file__).resolve().parents[2] / "kaldi-hmm-gmm" / "python"
if list((_src_python / "kaldi_hmm_gmm" / "lib").glob("_kaldi_hmm_gmm.*")):
    sys.path.insert(0, str(_src_python))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import kaldi_hmm_gmm as khg  # noqa: E402
import kaldifst  # noqa: E402
import kaldi_native_io  # noqa: E402

from align import gmm_align_compiled  # noqa: E402
from gmm import (  # noqa: E402
    gmm_acc_stats_ali,
    gmm_boost_silence,
    gmm_est,
    gmm_info,
    gmm_init_mono,
)
from read_feats import load_delta_feats, load_feats, load_text  # noqa: E402

# from steps/train_mono.sh
SCALE_OPTS = dict(transition_scale=1.0, acoustic_scale=0.1, self_loop_scale=0.1)
NUM_ITERS = 40
MAX_ITER_INC = 30
INITIAL_BEAM = 6
REGULAR_BEAM = 10
RETRY_BEAM = 40
TOTGAUSS = 400
CAREFUL = False
BOOST_SILENCE = 1.0
REALIGN_ITERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 38]
POWER = 0.25


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/train_yesno")
    parser.add_argument("--lang-dir", type=str, default="data/lang")
    parser.add_argument("--feats-scp", type=str, default="data/train_yesno/feats.scp")
    parser.add_argument("--exp-dir", type=str, default="exp/mono0a")
    parser.add_argument("--num-iters", type=int, default=NUM_ITERS)
    parser.add_argument(
        "--equal-align-seed",
        type=str,
        default="kaldi",
        choices=["kaldi", "3"],
        help="kaldi: StringHasher(utt_id) like align-equal-compiled; 3: fixed seed",
    )
    parser.add_argument("--totgauss", type=int, default=TOTGAUSS)
    return parser.parse_args()


class Logger:
    """Writes both to stdout and to exp/mono0a/log/<name>.log (kaldi-style)."""

    def __init__(self, log_dir: Path, name: str):
        self.f = open(log_dir / f"{name}.log", "w", encoding="utf-8", newline="\n")

    def log(self, msg: str):
        # stdout follows kaldi's train_mono.sh log (see the print() calls);
        # details go to exp/mono0a/log/<name>.log only.
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def string_hasher(key: str) -> int:
    """kaldi's StringHasher (util/stl-utils.h): ans = ans*7853 + c, size_t
    arithmetic, then truncated to int32 (as align-equal-compiled.cc does)."""
    ans = 0
    for c in key.encode():
        ans = (ans * 7853 + c) & 0xFFFFFFFFFFFFFFFF
    ans = ans & 0xFFFFFFFF  # int32 truncation
    if ans >= 0x80000000:
        ans -= 0x100000000
    return ans


def write_int_vector_archive_gz(path: str, data: dict):
    """Write {utt: [ints]} as a gzipped text archive (kaldi's ali.1.gz style)."""
    tmp = path + ".tmp"
    with kaldi_native_io.Int32VectorWriter(f"ark,t:{tmp}") as writer:
        for utt, vec in data.items():
            writer.write(utt, vec)
    with open(tmp, "rb") as fin, gzip.open(path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    Path(tmp).unlink()


def write_fst_archive_gz(path: str, fsts: dict):
    """Write {utt: VectorFst} as a gzipped binary fst archive (fsts.1.gz)."""
    tmp = path + ".tmp"
    with kaldifst.VectorFstWriter(f"ark:{tmp}") as writer:
        for utt, fst in fsts.items():
            writer.write(utt, fst)
    with open(tmp, "rb") as fin, gzip.open(path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    Path(tmp).unlink()


def write_occs(path: str, pdf_occs):
    vec = kaldi_native_io.FloatVector(pdf_occs.copy())
    vec.write(path, True)


def main():
    args = get_args()
    exp_dir = Path(args.exp_dir)
    log_dir = exp_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    text = load_text(f"{args.data_dir}/text")
    feats = load_delta_feats(args.feats_scp)
    assert set(text.keys()) == set(feats.keys()), (
        sorted(set(text) ^ set(feats))[:5],
    )
    utts = sorted(feats.keys())

    # ---- lang ----
    lang_dir = Path(args.lang_dir)
    topo = khg.HmmTopology()
    topo.read((lang_dir / "topo").read_text(encoding="utf-8"))
    lex_fst = kaldifst.StdVectorFst.read(str(lang_dir / "L.fst"))
    phone2id = {}
    with open(lang_dir / "phones.txt", encoding="utf-8") as f:
        for line in f:
            p, i = line.split()
            phone2id[p] = int(i)
    word2id = {}
    with open(lang_dir / "words.txt", encoding="utf-8") as f:
        for line in f:
            w, i = line.split()
            word2id[w] = int(i)
    disambig_ids = sorted(
        phone2id[p] for p in phone2id if p.startswith("#")
    )
    sil_phone_id = phone2id["SIL"]
    feat_dim = feats[utts[0]].shape[1]

    # ---- init (train_mono.sh stage -3..0) ----
    lg = Logger(log_dir, "init")
    lg.log(f"Initializing monophone system. feat_dim={feat_dim}")
    print("steps/train_mono.sh: Initializing monophone system.")
    # kaldi: gmm-init-mono --train-feats="$feats subset-feats --n=10"
    transition_model, tree, am = gmm_init_mono(
        topo=topo, feats=feats, num_init_utts=10
    )
    lg.log(str(gmm_info(am, transition_model)))
    khg.write_model(str(exp_dir / "0.mdl"), transition_model, am, binary=True)
    tree.write(binary=True, filename=str(exp_dir / "tree"))

    num_gauss = gmm_info(am, transition_model)["number_of_gaussians"]
    inc_gauss = (args.totgauss - num_gauss) // MAX_ITER_INC
    lg.log(f"num_gauss={num_gauss} inc_gauss={inc_gauss} target={args.totgauss}")
    lg.close()

    # ---- compile training graphs (stage -2) ----
    print("steps/train_mono.sh: Compiling training graphs")
    lg = Logger(log_dir, "compile_graphs")
    opts = khg.TrainingGraphCompilerOptions(
        transition_scale=0.0, self_loop_scale=0.0  # compile-train-graphs defaults
    )
    gc = khg.TrainingGraphCompiler(
        trans_model=transition_model,
        ctx_dep=tree,
        lex_fst=lex_fst,
        disambig_syms=disambig_ids,
        opts=opts,
    )
    train_graphs = {}
    for utt in utts:
        words = text[utt]
        word_ids = [word2id[w] for w in words]
        train_graphs[utt] = gc.compile_graph_from_text(word_ids)
    lg.log(f"compiled {len(train_graphs)} graphs")
    lg.close()

    # kaldi bookkeeping: fsts.1.gz (compiled train graphs), num_jobs,
    # phones.txt, cmvn_opts (mirrors train_mono.sh outputs in exp/mono0a)
    import gzip

    with kaldifst.VectorFstWriter(f"ark:{exp_dir}/fsts.1.ark") as w:
        for utt in utts:
            w.write(utt, train_graphs[utt])
    with open(exp_dir / "fsts.1.ark", "rb") as fin, gzip.open(
        exp_dir / "fsts.1.gz", "wb"
    ) as fout:
        fout.write(fin.read())
    (exp_dir / "fsts.1.ark").unlink()
    with open(exp_dir / "num_jobs", "w", encoding="utf-8", newline="\n") as f:
        f.write("1\n")
    shutil.copy(Path(args.lang_dir) / "phones.txt", exp_dir / "phones.txt")
    with open(exp_dir / "cmvn_opts", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n")

    print("steps/train_mono.sh: Aligning data equally (pass 0)")
    # ---- align-equal + first update (stage -1..0) ----
    lg = Logger(log_dir, "align.0")
    ali = {}
    for utt in utts:
        seed = string_hasher(utt) if args.equal_align_seed == "kaldi" else 3
        succeeded, aligned_fst = kaldifst.equal_align(
            ifst=train_graphs[utt],
            length=feats[utt].shape[0],
            rand_seed=seed,
            num_retries=10,
        )
        assert succeeded, utt
        succeeded, aligned_seq, _, _ = kaldifst.get_linear_symbol_sequence(
            aligned_fst
        )
        assert succeeded, utt
        ali[utt] = aligned_seq
    lg.close()

    lg = Logger(log_dir, "acc.0")
    gmm_accs = khg.AccumAmDiagGmm()
    gmm_accs.init(model=am, flags=khg.GmmUpdateFlags.kGmmAll)
    transition_accs = None
    tot_log_like = 0.0
    for utt in utts:
        log_like, transition_accs = gmm_acc_stats_ali(
            am_gmm=am,
            gmm_accs=gmm_accs,
            transition_model=transition_model,
            feats=feats[utt],
            ali=ali[utt],
            transition_accs=transition_accs,
        )
        tot_log_like += log_like
    lg.log(f"average log_like: {tot_log_like/len(utts)}")
    lg.close()

    lg = Logger(log_dir, "update.0")
    _lf = open(log_dir / "update.0.log", "a", encoding="utf-8")
    _redir = contextlib.redirect_stdout(_lf)
    tcfg = khg.MleTransitionUpdateConfig()
    gmm_opts = khg.MleDiagGmmOptions()
    gmm_opts.min_gaussian_occupancy = 3  # train_mono.sh stage 0 est
    with _redir:
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
            power=POWER,
            min_count=20.0,
            update_flags="mvwt",
        )
    _lf.close()
    khg.write_model(str(exp_dir / "1.mdl"), transition_model, am, binary=True)
    lg.close()

    # ---- main loop (x = 1 .. num_iters-1) ----
    beam = INITIAL_BEAM

    for x in range(1, args.num_iters):
        lg = Logger(log_dir, f"pass.{x}")
        lg.log(f"Pass {x}")
        print(f"steps/train_mono.sh: Pass {x}")
        if x in REALIGN_ITERS:
            lg.log("Aligning data")
            print("steps/train_mono.sh: Aligning data")
            with open(log_dir / f"align.{x}.log", "w", encoding="utf-8", newline="\n") as _lf3, \
                    contextlib.redirect_stdout(_lf3):
                boosted = gmm_boost_silence(
                    am_gmm=am,
                    transition_model=transition_model,
                    silence_phones=[sil_phone_id],
                    boost=BOOST_SILENCE,
                )
            align_config = khg.AlignConfig()
            align_config.beam = beam
            align_config.retry_beam = RETRY_BEAM
            align_config.careful = CAREFUL

            align_prob = 0.0
            align_frames = 0.0
            for utt in utts:
                ans = gmm_align_compiled(
                    am_gmm=boosted,
                    transition_model=transition_model,
                    utt=utt,
                    fst=train_graphs[utt].copy(),
                    feats=feats[utt],
                    align_config=align_config,
                    acoustic_scale=SCALE_OPTS["acoustic_scale"],
                    transition_scale=SCALE_OPTS["transition_scale"],
                    self_loop_scale=SCALE_OPTS["self_loop_scale"],
                )
                assert ans["num_error"] == 0, utt
                ali[utt] = ans["alignment"]
                align_prob += float(ans.get("tot_like", 0.0))
                align_frames += float(ans.get("frame_count", 0))
            # kaldi: ark,t:|gzip -c >ali.1.gz (overwritten at every realign)
            write_int_vector_archive_gz(str(exp_dir / "ali.1.gz"), ali)

        gmm_accs = khg.AccumAmDiagGmm()
        gmm_accs.init(model=am, flags=khg.GmmUpdateFlags.kGmmAll)
        transition_accs = None
        tot_log_like = 0.0
        for utt in utts:
            log_like, transition_accs = gmm_acc_stats_ali(
                am_gmm=am,
                gmm_accs=gmm_accs,
                transition_model=transition_model,
                feats=feats[utt],
                ali=ali[utt],
                transition_accs=transition_accs,
            )
            tot_log_like += log_like
        lg.log(f"average log_like: {tot_log_like/len(utts)}")

        gmm_opts = khg.MleDiagGmmOptions()  # min_gaussian_occupancy=10 default
        with open(log_dir / f"update.{x}.log", "w", encoding="utf-8", newline="\n") as _lf2, \
                contextlib.redirect_stdout(_lf2):
          pdf_occs = gmm_est(
            am_gmm=am,
            gmm_accs=gmm_accs,
            transition_model=transition_model,
            transition_accs=transition_accs,
            tcfg=tcfg,
            gmm_opts=gmm_opts,
            mixup=num_gauss,
            mixdown=0,
            perturb_factor=0.01,
            power=POWER,
            min_count=20.0,
            update_flags="mvwt",
        )
        write_occs(str(exp_dir / f"{x+1}.occs"), pdf_occs)
        khg.write_model(str(exp_dir / f"{x+1}.mdl"), transition_model, am, binary=True)
        # kaldi's train_mono.sh: rm $dir/$x.mdl $dir/$x.occs after the update
        (exp_dir / f"{x}.mdl").unlink(missing_ok=True)
        (exp_dir / f"{x}.occs").unlink(missing_ok=True)
        if x <= MAX_ITER_INC:
            num_gauss += inc_gauss
        beam = REGULAR_BEAM
        lg.close()

    # ---- final ----
    print("steps/diagnostic/analyze_alignments.sh --cmd utils/run.pl "
          f"{args.lang_dir} {args.exp_dir}")
    print("steps/diagnostic/analyze_alignments.sh: see stats in "
          f"{args.exp_dir}/log/analyze_alignments.log")
    gauss = sum(am.num_gauss_in_pdf(i) for i in range(am.num_pdfs))
    prob = align_prob / max(align_frames, 1.0)
    print(f"{args.exp_dir}: nj=1 align prob={prob:.2f} "
          f"over {align_frames * 0.01 / 3600:.2f}h "
          f"[retry=0.0%, fail=0.0%] states={am.num_pdfs} gauss={gauss}")
    print(f"steps/train_mono.sh: Done training monophone system in {args.exp_dir}")

    shutil.copy(exp_dir / f"{args.num_iters}.mdl", exp_dir / "final.mdl")
    shutil.copy(exp_dir / f"{args.num_iters}.occs", exp_dir / "final.occs")
    if (exp_dir / f"{args.num_iters}.occs").exists():
        shutil.copy(exp_dir / f"{args.num_iters}.occs", exp_dir / "final.occs")

    lg = Logger(log_dir, "analyze_alignments")
    info = gmm_info(am, transition_model)
    lg.log(str(info))
    # kaldi's gmm_dir_info.pl summary line style
    print(
        f"{exp_dir}: nj=1 align prob={tot_log_like/len(utts):.2f} "
        f"states={info['number_of_transition_states']} "
        f"gauss={info['number_of_gaussians']}"
    )
    lg.log("Done training monophone system in " + str(exp_dir))
    lg.close()


if __name__ == "__main__":
    main()
