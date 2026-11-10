#!/usr/bin/env python3
"""Cut a wav into per-word clips according to a TextGrid's words tier.

Reads the words tier of a Praat TextGrid (the format written by
tools/align_mfa.py) and writes one wav file per word interval, named
``wNN.wav`` (NN = 1-based position in time), plus a ``manifest.json`` with
the word label, start/end times and file name of every clip.

Usage:
  python3 tools/textgrid_to_word_wavs.py --wav x.wav --textgrid x.TextGrid \
      --out-dir clips/

Output:
  clips/w01.wav ... clips/wNN.wav   per-word audio (same sample rate/format)
  clips/manifest.json               [{word, start, end, file}, ...]
"""

import argparse
import json
from pathlib import Path


def read_words_tier(textgrid_path):
    """Parse the words tier of a TextGrid into (start, end, label) tuples.

    Handles the interval-tier format written by tools/align_mfa.py:
    ``item [2]: ... name = "words" ... intervals [k]: xmin/xmax/text``.

    Args:
      textgrid_path: Path of the TextGrid file.

    Returns:
      List of (start, end, word) in time order.
    """
    lines = Path(textgrid_path).read_text(encoding="utf-8").splitlines()

    # locate the tier named "words"
    tier_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("item ["):
            block = "\n".join(lines[i : i + 4])
            if 'name = "words"' in block:
                tier_start = i
                break
    if tier_start is None:
        raise ValueError(f"{textgrid_path}: no words tier found")

    intervals = []
    i = tier_start + 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("item ["):
            break  # next tier
        if line.startswith("intervals ["):
            xmin = float(lines[i + 1].split("=", 1)[1])
            xmax = float(lines[i + 2].split("=", 1)[1])
            text = lines[i + 3].split("=", 1)[1].strip().strip('"')
            intervals.append((xmin, xmax, text))
            i += 4
        else:
            i += 1
    return intervals


def main():
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--textgrid", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    words = read_words_tier(args.textgrid)
    samples, sample_rate = sf.read(args.wav, dtype="int16", always_2d=False)
    if samples.ndim != 1:
        raise ValueError(f"{args.wav}: only mono wav is supported")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, (start, end, word) in enumerate(words, 1):
        i0 = int(round(start * sample_rate))
        i1 = int(round(end * sample_rate))
        clip = samples[i0:i1]
        name = f"w{k:02d}.wav"
        sf.write(out_dir / name, clip, sample_rate)
        manifest.append(
            {"word": word, "start": round(start, 4), "end": round(end, 4),
             "file": name}
        )
        print(f"{name}  {start:.2f}-{end:.2f}s  {word}")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(manifest)} clips + manifest.json to {out_dir}/")


if __name__ == "__main__":
    main()
