#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file computes fbank features of the AudioMNIST dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from lhotse import (
    CutSet,
    RecordingSet,
    SupervisionSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
)
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_audio_mnist():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80
    logging.info(f"num_jobs: {num_jobs}")

    recordings = RecordingSet.from_file(
        "data/manifests/audio_mnist_recordings.jsonl.gz"
    )
    supervisions = SupervisionSet.from_file(
        "data/manifests/audio_mnist_supervisions.jsonl.gz"
    )

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    cuts_filename = f"audio_mnist_cuts.jsonl.gz"
    if (output_dir / cuts_filename).is_file():
        logging.info(f"cuts already exists - skipping.")
        return

    cut_set = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions,
    )

    cut_set = cut_set.resample(16000)

    #  cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
    cut_set = cut_set.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{output_dir}/audio_mnist_feats",
        num_jobs=num_jobs,
        storage_type=LilcomChunkyWriter,
    )
    cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    compute_fbank_audio_mnist()
