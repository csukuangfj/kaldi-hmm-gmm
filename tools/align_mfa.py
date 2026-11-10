#!/usr/bin/env python3
"""Forced alignment with MFA acoustic models (kaldi-format), using this repo.

Pipeline (mirrors MFA/kaldi feature+align conventions; no kaldi binaries):
  1) features via kaldi-native-fbank step operators (meta.json opts):
       MFCC -> [processed pitch + concat] -> CMVN -> splice -> LDA
       -> 40-dim model features
  2) graph: dictionary -> L (optional silence 'sil') + linear G (known text)
       -> TrainingGraphCompiler (this repo)
  3) alignment: gmm_align_compiled with final.alimdl (this repo)
  4) output: Praat TextGrid (phones + words tiers)

Usage:
  python3 tools/align_mfa.py --wav x.wav --text "..." --model-dir DIR \
      --dict FILE --phones-txt FILE [--pitch] --out x.TextGrid
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Prefer the pip-installed kaldi_hmm_gmm; only put the in-repo package tree
# first when it contains a locally built extension (development layout).
_src_python = REPO / "kaldi-hmm-gmm" / "python"
if list((_src_python / "kaldi_hmm_gmm" / "lib").glob("_kaldi_hmm_gmm.*")):
    sys.path.insert(0, str(_src_python))

try:
    import kaldi_native_fbank as knf
except ImportError as e:
    raise ImportError(
        "This tool requires kaldi-native-fbank for feature computation.\n"
        "Install it with:\n"
        "  pip install kaldi-native-fbank"
    ) from e

import numpy as np  # noqa: E402
import kaldi_hmm_gmm as khg  # noqa: E402
import kaldifst  # noqa: E402


def read_wave(path):
    """Read a wav file the way kaldi does (via soundfile).

    Returns raw int16 amplitudes as float32 (NOT divided by 32768): kaldi's
    wav reader keeps the integer scale and all kaldi feature references are
    produced on that scale. Returns (samples, sample_rate).
    """
    import soundfile as sf

    samples, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    if samples.ndim != 1:
        raise ValueError(f"{path}: only mono wav is supported")
    return samples.astype(np.float32), sample_rate


def mfcc_options(feat):
    """Map meta.json 'features' fields to knf.MfccOptions (kaldi MfccOptions)."""
    opts = knf.MfccOptions()
    opts.use_energy = bool(feat.get("use_energy", False))
    opts.energy_floor = float(feat.get("energy_floor", 1.0))
    opts.num_ceps = int(feat.get("num_coefficients", 13))
    opts.cepstral_lifter = float(feat.get("cepstral_lifter", 22))
    opts.mel_opts.num_bins = int(feat.get("num_mel_bins", 23))
    opts.mel_opts.low_freq = float(feat.get("low_frequency", 20))
    opts.mel_opts.high_freq = float(feat.get("high_frequency", 7800))
    opts.frame_opts.samp_freq = float(feat.get("sample_frequency", 16000))
    opts.frame_opts.frame_shift_ms = float(feat.get("frame_shift", 10))
    opts.frame_opts.frame_length_ms = float(feat.get("frame_length", 25))
    opts.frame_opts.dither = float(feat.get("dither", 0.0001))
    opts.frame_opts.preemph_coeff = float(
        feat.get("preemphasis_coefficient", 0.97)
    )
    opts.frame_opts.snip_edges = bool(feat.get("snip_edges", True))
    return opts


def pitch_options(feat):
    """Map meta.json 'features' fields to knf.PitchExtractionOptions."""
    opts = knf.PitchExtractionOptions()
    opts.samp_freq = float(feat.get("sample_frequency", 16000))
    opts.frame_shift_ms = float(feat.get("frame_shift", 10))
    opts.frame_length_ms = float(feat.get("frame_length", 25))
    opts.min_f0 = float(feat.get("min_f0", 50))
    opts.max_f0 = float(feat.get("max_f0", 500))
    opts.delta_pitch = float(feat.get("delta_pitch", 0.005))
    opts.penalty_factor = float(feat.get("penalty_factor", 0.1))
    opts.snip_edges = bool(feat.get("snip_edges", True))
    return opts


def resample_wave(samples, sr_in, sr_out):
    """Resample a waveform to sr_out Hz with librosa (soxr resampler).

    Only used when the wav's sample rate differs from the model's, so
    librosa is an optional dependency (pip install librosa).

    Args:
      samples: 1-D float32 waveform.
      sr_in: Input sample rate in Hz.
      sr_out: Output sample rate in Hz.

    Returns:
      1-D float32 waveform at sr_out (length ~ len(samples) * sr_out / sr_in).
    """
    if sr_in == sr_out:
        return samples
    try:
        import librosa
    except ImportError:
        raise ImportError(
            f"wav is {sr_in} Hz but the model expects {sr_out} Hz; "
            "install librosa to resample: pip install librosa"
        )
    return librosa.resample(
        samples.astype(np.float32), orig_sr=int(sr_in), target_sr=int(sr_out)
    ).astype(np.float32)


def compute_model_features(samples, sample_rate, meta):
    """Compute model features (e.g. 40-dim) for one utterance.

    Feature stage of MFA/kaldi forced alignment; the chain is selected by
    the model's meta.json ("features" section):

    1. resample to the model's sample_frequency if needed (MFA semantics:
       allow_downsample/allow_upsample, both default True);
    2. MFCC (kaldi compute-mfcc-feats semantics);
    3. if use_pitch: processed pitch (kaldi
       compute-and-process-kaldi-pitch-feats semantics; by default
       POV + normalized-log-pitch + delta-pitch) concatenated with the MFCCs;
    4. if uses_cmvn: utterance-level CMVN (compute-cmvn-stats + apply-cmvn);
    5. if meta["lda"] is set: splice +-splice_left/right_context frames
       (default 3) and apply the transform (splice-feats + transform-feats);
       else if uses_deltas: delta features (add-deltas).

    Args:
      samples: 1-D raw waveform (kaldi scale: raw int16 amplitudes as
        float32, not divided by 32768; see read_wave()).
      sample_rate: Sample rate of samples in Hz; resampled to the model's
        features.sample_frequency when they differ.
      meta: The model's meta.json content (a dict); meta["lda"] is the
        path of the LDA matrix file.

    Returns:
      A 2-D float32 array of shape (num_frames, feat_dim).
    """
    feat = meta.get("features", {})
    expected = float(feat.get("sample_frequency", 16000))
    if sample_rate != expected:
        if sample_rate > expected and not bool(feat.get("allow_downsample", True)):
            raise ValueError(
                f"sample rate mismatch: wav is {sample_rate} Hz but the model "
                f"expects {expected} Hz and allow_downsample is false"
            )
        if sample_rate < expected and not bool(feat.get("allow_upsample", True)):
            raise ValueError(
                f"sample rate mismatch: wav is {sample_rate} Hz but the model "
                f"expects {expected} Hz and allow_upsample is false"
            )
        samples = resample_wave(samples, int(sample_rate), int(expected))
        sample_rate = expected
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D, got shape {samples.shape}")

    # 1) MFCC
    mfcc = knf.OnlineMfcc(mfcc_options(feat))
    mfcc.accept_waveform(int(sample_rate), samples)
    mfcc.input_finished()
    num_frames = mfcc.num_frames_ready
    if num_frames == 0:
        raise ValueError("utterance too short: 0 feature frames")
    base = np.stack([mfcc.get_frame(i) for i in range(num_frames)]).astype(
        np.float32
    )

    # 2) pitch (processed: POV + normalized-log-pitch + delta-pitch by default)
    if feat.get("use_pitch", False):
        pitch = knf.compute_and_process_kaldi_pitch(
            pitch_options(feat), knf.ProcessPitchOptions(), samples
        )
        if pitch.shape[0] != base.shape[0]:
            raise RuntimeError(
                f"pitch/mfcc frame count mismatch: {pitch.shape[0]} vs "
                f"{base.shape[0]}"
            )
        base = np.hstack([base, pitch])

    # 3) CMVN
    if feat.get("uses_cmvn", True):
        stats = knf.compute_cmvn_stats(base)
        base = knf.apply_cmvn(stats, knf.CmvnOptions(), base)

    # 4) splice -> LDA, or deltas
    lda = meta.get("lda")
    if lda is not None:
        transform = (
            lda
            if isinstance(lda, knf.FeatureTransform)
            else knf.FeatureTransform.read(str(lda))
        )
        splice_opts = knf.SpliceOptions(
            left_context=int(feat.get("splice_left_context", 3)),
            right_context=int(feat.get("splice_right_context", 3)),
        )
        base = knf.splice_frames(base, splice_opts)
        base = knf.transform_features(base, transform)
    elif feat.get("uses_deltas", False):
        base = knf.compute_deltas(knf.DeltaOptions(), base)

    return np.ascontiguousarray(base)


def _is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def load_dictionary(path):
    """path may be a comma-separated list; entries merge (earlier files win)."""
    d = {}
    for one in str(path).split(","):
        with open(one.strip(), encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[0].lower() not in d:
                    # skip numeric weight columns (some dicts: word w1 w2 w3 w4 phones)
                    phones = [t for t in p[1:] if not _is_float(t)]
                    if phones:
                        d[p[0].lower()] = [phones]
    return d


def build_textgrid(intervals, word_intervals, duration, out_path):
    def tier(name, items):
        s = f'    item []:\n        item [1] if false:\n'  # placeholder unused
        return s

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration:.4f}",
        "tiers? <exists>",
        "size = 2",
        "item []:",
    ]

    def add_tier(idx, name, items):
        lines.append(f"    item [{idx}]:")
        lines.append('        class = "IntervalTier"')
        lines.append(f'        name = "{name}"')
        lines.append("        xmin = 0")
        lines.append(f"        xmax = {duration:.4f}")
        lines.append(f"        intervals: size = {len(items)}")
        for j, (t0, t1, txt) in enumerate(items, 1):
            lines.append(f"        intervals [{j}]:")
            lines.append(f"            xmin = {t0:.4f}")
            lines.append(f"            xmax = {t1:.4f}")
            lines.append(f'            text = "{txt}"')

    add_tier(1, "phones", intervals)
    add_tier(2, "words", word_intervals)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--phones-txt", default="")
    ap.add_argument("--pitch", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenize", default="space", choices=["space", "char", "pinyin"],
                    help="space: whitespace words; char: one word per char (mandarin syllables)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.load(open(model_dir / "meta.json"))
    meta["lda"] = str(model_dir / "lda.mat")

    # 1) features (kaldi-native-fbank; no kaldi binaries)
    t0 = time.perf_counter()
    if args.pitch:
        meta.setdefault("features", {})["use_pitch"] = True
    samples, sample_rate = read_wave(str(Path(args.wav).resolve()))
    audio_seconds = len(samples) / float(sample_rate)
    feat = compute_model_features(samples, sample_rate, meta)

    # 2) lexicon + word sequence
    lex = load_dictionary(args.dict)
    raw_text = args.text.strip()
    if args.tokenize == "pinyin":
        from pypinyin import lazy_pinyin, Style
        import re as _re
        chars = [c for c in raw_text if not c.isspace()]
        toks = []       # toneless pinyin keys for dictionary lookup
        disp = []       # TextGrid display labels
        for c in chars:
            if _re.match(r"[\u4e00-\u9fff]", c):
                plain = _re.sub(r"\d", "", lazy_pinyin(c)[0]).lower()
                toned = lazy_pinyin(c, style=Style.TONE)[0]  # e.g. dui4 -> duì (tone marks)
                toks.append(plain)
                disp.append(c)  # hanzi only (pinyin used for lookup only)
    elif args.tokenize == "char":
        # longest-match segmentation against the dictionary (MFA chinese-style)
        toks, i = [], 0
        chars = [c for c in raw_text if not c.isspace()]
        while i < len(chars):
            for j in range(len(chars), i, -1):
                w = "".join(chars[i:j])
                if w in lex:
                    toks.append(w)
                    i = j
                    break
            else:
                toks.append(chars[i])
                i += 1
    else:
        toks = [w.strip(".,!?;:\"'()[]").lower() for w in raw_text.split() if w.strip(".,!?;:\"'()[]")]
    lookups = toks
    words = disp if args.tokenize == "pinyin" else list(toks)
    word_phones = []
    for key in lookups:
        if key in lex:
            word_phones.append(lex[key][0])
        else:
            word_phones.append([meta.get("oov_phone", "spn")])

    # 3) phones.txt table (symbol -> model phone id)
    if args.phones_txt:
        phone2id = {}
        with open(args.phones_txt, encoding="utf-8") as f:
            for line in f:
                p, i = line.split()
                phone2id[p] = int(i)
    else:
        phone2id = {p: i + 1 for i, p in enumerate(meta["phones"])}

    sil = meta.get("optional_silence_phone", "sil")
    for p in set(sum(word_phones, [])):
        if p not in phone2id:
            raise SystemExit(f"phone {p!r} not in phone table (ids 1..{len(phone2id)})")
    sil_id = phone2id[sil]

    # 4) model + tree
    tm, am = khg.read_model(str(model_dir / "final.alimdl"))
    tree = khg.read_tree(str(model_dir / "tree"))

    # 5) L.fst with MODEL phone ids (kaldi/MFA lexicon structure:
    #    optional silence 'sil' at start/between words/end with sil_prob)
    import math

    sil_prob = meta.get("silence_probability", 0.5)
    sil_cost = -math.log(sil_prob)
    no_sil_cost = -math.log(1.0 - sil_prob)
    fst = kaldifst.StdVectorFst()
    start_s = fst.add_state()
    loop = fst.add_state()
    sil_s = fst.add_state()
    fst.start = start_s
    fst.set_final(loop, 0.0)
    fst.add_arc(start_s, kaldifst.StdArc(0, 0, no_sil_cost, loop))
    fst.add_arc(start_s, kaldifst.StdArc(0, 0, sil_cost, sil_s))
    fst.add_arc(sil_s, kaldifst.StdArc(sil_id, 0, 0.0, loop))
    word_ids = {}
    wid = 1
    for w, ps in zip(words, word_phones):
        if w in word_ids:
            continue
        word_ids[w] = wid
        wid += 1
        seq = [phone2id[p] for p in ps]
        cur = loop
        for k in range(len(seq) - 1):
            nxt = fst.add_state()
            fst.add_arc(cur, kaldifst.StdArc(seq[k], word_ids[w] if k == 0 else 0, 0.0, nxt))
            cur = nxt
        last = seq[-1]
        olab = word_ids[w] if len(seq) == 1 else 0
        fst.add_arc(cur, kaldifst.StdArc(last, olab, no_sil_cost, loop))
        fst.add_arc(cur, kaldifst.StdArc(last, olab, sil_cost, sil_s))
    kaldifst.arcsort(fst, sort_type="olabel")
    lex_fst = fst

    disambig = sorted(i for p, i in phone2id.items() if p.startswith("#"))
    gc = khg.TrainingGraphCompiler(
        trans_model=tm,
        ctx_dep=tree,
        lex_fst=lex_fst,
        disambig_syms=disambig,
        # compile-train-graphs semantics: transition probs are baked with
        # scale 0 here and added ONCE at alignment time (AddTransitionProbs
        # inside gmm_align_compiled); kaldi src/bin/compile-train-graphs.cc
        # sets both scales to 0.0 for the same reason.
        opts=khg.TrainingGraphCompilerOptions(transition_scale=0.0, self_loop_scale=0.0),
    )
    wids = [word_ids[w] for w in words]
    graph = gc.compile_graph_from_text(wids)

    # 5) align — follow MFA/kalpy exactly:
    #    transition_scale=1.0, acoustic_scale=0.1, self_loop_scale=0.1,
    #    boost_silence=1.0, beam=10, retry_beam=40, careful=False
    #    (kalpy/kalpy/aligner.py GmmAligner defaults; MFA
    #    montreal_forced_aligner/alignment/mixins.py AlignMixin same)
    from align import gmm_align_compiled
    from ali_to_phones import ali_to_phones

    cfg = khg.AlignConfig()
    cfg.beam = 10.0
    cfg.retry_beam = 40.0
    cfg.careful = False
    ans = gmm_align_compiled(
        am_gmm=am,
        transition_model=tm,
        utt="utt",
        fst=graph,
        feats=feat,
        align_config=cfg,
        acoustic_scale=0.1,
        transition_scale=1.0,
        self_loop_scale=0.1,
    )
    if ans["num_error"] != 0:
        raise SystemExit("alignment failed")
    ali = ans["alignment"]

    # 6) TextGrid
    n_frames = len(ali)
    frame_shift = meta.get("features", {}).get("frame_shift", 10) / 1000.0
    duration = n_frames * frame_shift
    phones = ali_to_phones(tm, ali, per_frame=False)
    # per-frame phone seq to build word boundaries
    per_frame = ali_to_phones(tm, ali, per_frame=True)
    id2phone = {i: p for p, i in phone2id.items()}

    # phone intervals: use SplitToPhones-based segments so that ADJACENT
    # INSTANCES of the same phone stay separate (e.g. here=...+e, and=e+...).
    from ali_to_phones import split_to_phones as _stp

    segs = _stp(tm, ali)  # list of transition-id segments, one per phone instance
    intervals = []
    t = 0.0
    for seg in segs:
        n = len(seg)
        sym = id2phone[ali_to_phones(tm, seg, per_frame=False)[0]]
        intervals.append((t, t + n * frame_shift, sym))
        t += n * frame_shift

    # word intervals: distribute phone segments among words in order
    # word tier (kalpy/MFA semantic: each word spans exactly its
    # pronunciation's phone instances; silence runs are unattributed gaps).
    n_phones = []
    for ps in word_phones:
        n_phones.append(len(ps))
    word_intervals = []
    seg_i = 0
    for w, need in zip(words, n_phones):
        # skip leading silence runs
        while seg_i < len(intervals) and intervals[seg_i][2] == "sil":
            seg_i += 1
        got = intervals[seg_i : seg_i + need]
        seg_i += need
        if got:
            word_intervals.append((got[0][0], got[-1][1], w))

    build_textgrid(intervals, word_intervals, duration, args.out)
    print(f"OK: {args.out}  frames={n_frames} dur={duration:.2f}s phones={len(intervals)} words={len(word_intervals)}")
    elapsed = time.perf_counter() - t0
    print(
        f"timing: audio duration {audio_seconds:.2f}s, "
        f"elapsed {elapsed:.2f}s, RTF {elapsed / audio_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
