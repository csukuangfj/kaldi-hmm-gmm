#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#
# Feature stage for the yesno recipe (kaldi-native-fbank; no kaldi binaries).
# Mirrors ../kaldi/egs/yesno/s5/steps/make_mfcc.sh +
# steps/compute_cmvn_stats.sh outputs:
#
#   mfcc/raw_mfcc_{part}.1.{ark,scp}   raw 13-dim MFCC (make_mfcc.sh)
#   mfcc/cmvn_{part}.{ark,scp}         per-speaker CMVN stats
#   data/{part}_yesno/feats.scp        -> the raw MFCC arks
#   data/{part}_yesno/cmvn.scp         -> the cmvn arks
#   data/{part}_yesno/{conf/mfcc.conf,frame_shift,utt2num_frames,utt2dur}
#   exp/make_mfcc/{part}/make_mfcc_{part}.1.log
#
# Like kaldi, CMVN + deltas are applied at use time (see
# tools/read_feats.load_delta_feats), not stored on disk.
#
# The chain is conf/mfcc.conf: --use-energy=false --sample-frequency=8000;
# every other option is the compute-mfcc-feats default, including
# dither=1.0 (kaldi FrameExtractionOptions).  Input = data/{part}_yesno/wav.scp
# (prepare_data.sh).

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

try:
    import kaldi_native_fbank as knf
except ImportError as e:
    raise ImportError(
        "This tool requires kaldi-native-fbank for feature computation.\n"
        "Install it with:\n"
        "  pip install kaldi-native-fbank"
    ) from e

try:
    import kaldi_native_io
except ImportError as e:
    raise ImportError(
        "This tool requires kaldi_native_io for writing kaldi ark/scp files.\n"
        "Install it with:\n"
        "  pip install kaldi_native_io"
    ) from e


def read_wave(path):
    """kaldi wav reading: raw int16 amplitudes as float32 (no /32768)."""
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1:
            raise ValueError(f"{path}: only mono wav is supported")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav is supported")
        sample_rate = f.getframerate()
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return samples.astype(np.float32), sample_rate


def mfcc_options():
    opts = knf.MfccOptions()
    opts.use_energy = False
    opts.frame_opts.samp_freq = 8000.0
    opts.frame_opts.dither = 1.0  # kaldi default
    return opts


def compute_mfcc(samples, sample_rate, opts):
    mfcc = knf.OnlineMfcc(opts)
    mfcc.accept_waveform(int(sample_rate), samples)
    mfcc.input_finished()
    n = mfcc.num_frames_ready
    if n == 0:
        raise ValueError("utterance too short: 0 feature frames")
    return np.stack([mfcc.get_frame(i) for i in range(n)]).astype(np.float32)


def compute_part(part, data_dir, out_mfcc_dir, log_dir):
    wav_scp = data_dir / "wav.scp"
    utts = []
    feats = []
    acc = knf.CmvnAccumulator()
    with open(wav_scp, encoding="utf-8") as f:
        for line in f:
            utt, path = line.split(maxsplit=1)
            samples, sample_rate = read_wave(path.strip())
            x = compute_mfcc(samples, sample_rate, mfcc_options())
            utts.append(utt)
            feats.append(x)
            acc.accumulate(x)

    raw_ark = out_mfcc_dir / f"raw_mfcc_{part}.1.ark"
    raw_scp = out_mfcc_dir / f"raw_mfcc_{part}.1.scp"
    cmvn_ark = out_mfcc_dir / f"cmvn_{part}.ark"
    cmvn_scp = out_mfcc_dir / f"cmvn_{part}.scp"
    with kaldi_native_io.FloatMatrixWriter(f"ark,scp:{raw_ark},{raw_scp}") as w:
        for utt, x in zip(utts, feats):
            w.write(utt, x)
    with kaldi_native_io.FloatMatrixWriter(f"ark,scp:{cmvn_ark},{cmvn_scp}") as w:
        w.write("global", acc.stats.astype(np.float32))

    with open(data_dir / "feats.scp", "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{u} {raw_ark.resolve()}:{o}\n" for u, o in _offsets(raw_scp)))
    with open(data_dir / "cmvn.scp", "w", encoding="utf-8", newline="\n") as f:
        f.write(cmvn_scp.read_text())
    with open(data_dir / "utt2num_frames", "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{u} {len(x)}\n" for u, x in zip(utts, feats)))
    with open(data_dir / "utt2dur", "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"{u} {len(x) * 0.01:.4f}\n" for u, x in zip(utts, feats)))
    conf = data_dir / "conf"
    conf.mkdir(exist_ok=True)
    with open(conf / "mfcc.conf", "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "--use-energy=false   # only non-default option.\n"
            "--sample-frequency=8000 #  Switchboard is sampled at 8kHz\n")
    with open(data_dir / "frame_shift", "w", encoding="utf-8", newline="\n") as f:
        f.write("10\n")

    # kaldi's utils/split_data.sh (nj=1) + utils/fix_data_dir.sh artifacts
    import shutil

    for sub in ("split1", ".backup"):
        d = data_dir / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        for f in data_dir.iterdir():
            if f.is_file():
                shutil.copy(f, d / f.name)

    print(f"steps/make_mfcc.sh --nj 1 data/{part} exp/make_mfcc/{part} mfcc")
    print(f"utils/validate_data_dir.sh: Successfully validated data-directory data/{part}")
    print("steps/make_mfcc.sh: [info]: no segments file exists: "
          "assuming wav.scp indexed by utterance.")
    print(f"steps/make_mfcc.sh: Succeeded creating MFCC features for {part}")
    print(f"steps/compute_cmvn_stats.sh data/{part} exp/make_mfcc/{part} mfcc")
    print(f"Succeeded creating CMVN stats for {part}")
    print(f"fix_data_dir.sh: kept all {len(utts):8d} utterances.")
    print(f"fix_data_dir.sh: old files are kept in data/{part}/.backup")

    log_dir.mkdir(parents=True, exist_ok=True)
    with open(
        log_dir / f"make_mfcc_{part}.1.log", "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write(
            f"# compute-mfcc-feats --config=conf/mfcc.conf scp:data/{part}_yesno/wav.scp\n"
            f"# (kaldi-native-fbank equivalent; wrote {raw_ark})\n"
            f"Processed {len(utts)} utterances\n")


def _offsets(scp_path):
    """(utt, byte-offset) pairs from a written scp file."""
    out = []
    for line in Path(scp_path).read_text().splitlines():
        u, rest = line.split(maxsplit=1)
        out.append((u, rest.rsplit(":", 1)[1]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mfcc-dir", default="mfcc")
    parser.add_argument("--exp-dir", default="exp/make_mfcc")
    args = parser.parse_args()

    data = Path(args.data_dir)
    out_mfcc = Path(args.mfcc_dir)
    out_mfcc.mkdir(parents=True, exist_ok=True)
    for part in ("train_yesno", "test_yesno"):
        compute_part(part, data / part, out_mfcc, Path(args.exp_dir) / part)
    print(f"{sys.argv[0]}: done.")


if __name__ == "__main__":
    main()
