#!/usr/bin/env bash
# Prepare the audio published with the documentation:
#
#   docs/source/_static/en.wav              page audio (English test wav)
#   docs/source/_static/zh.wav              page audio (Mandarin test wav)
#   docs/source/_static/words/{en,zh-aishell3,zh-large}/wNN.wav
#                                           one clip per word interval, cut
#                                           by tools/textgrid_to_word_wavs.py
#                                           from the TextGrid word tier
#
# The wav files are NOT committed; CI runs this script before `make html`
# and the gh-pages deploy publishes them with the site.  Idempotent: every
# download/unpack step is skipped when its output already exists.

set -euo pipefail
cd "$(dirname "$0")/.."  # repository root

WORK=docs/build/audio-work
STATIC=docs/source/_static
EN_TEXT="Ask not what your country can do for you, ask what you can do for your country."
ZH_TEXT="不要问你的国家能为你做什么，而要问你能为你的国家做什么。"

mkdir -p "$WORK" "$STATIC"

# 1) page audio
if [ ! -f "$STATIC/en.wav" ]; then
  wget -q -O "$STATIC/en.wav" \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav
fi
if [ ! -f "$STATIC/zh.wav" ]; then
  wget -q -O "$STATIC/zh.wav" \
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/zh.wav
fi

# 2) acoustic models + dictionaries
if [ ! -d "$WORK/english_mfa" ]; then
  wget -q -O "$WORK/english_mfa.zip" \
    https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-english_mfa-v3.1.0/english_mfa.zip
  unzip -q "$WORK/english_mfa.zip" -d "$WORK"
fi
if [ ! -f "$WORK/english_mfa.dict" ]; then
  wget -q -O "$WORK/english_mfa.dict" \
    https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-english_mfa-v3.1.0/english_mfa.dict
fi
if [ ! -d "$WORK/aishell3_pinyin/aishell3_pinyin_acoustic" ]; then
  mkdir -p "$WORK/aishell3_pinyin"
  wget -q -O "$WORK/aishell3_pinyin/aishell3_pinyin.zip" \
    https://github.com/lars76/forced-alignment-chinese/releases/download/model_files/aishell3_pinyin.zip
  unzip -q "$WORK/aishell3_pinyin/aishell3_pinyin.zip" -d "$WORK/aishell3_pinyin"
  unzip -q "$WORK/aishell3_pinyin/aishell3_pinyin_acoustic.zip" -d "$WORK/aishell3_pinyin"
fi
if [ ! -d "$WORK/large_pinyin/d_pinyin_acoustic" ]; then
  mkdir -p "$WORK/large_pinyin"
  wget -q -O "$WORK/large_pinyin/large_pinyin.zip" \
    https://github.com/lars76/forced-alignment-chinese/releases/download/model_files/large_pinyin.zip
  unzip -q "$WORK/large_pinyin/large_pinyin.zip" -d "$WORK/large_pinyin"
  unzip -q "$WORK/large_pinyin/large_pinyin_acoustic.zip" -d "$WORK/large_pinyin"
fi

# 3) align -> TextGrid (work files only) -> per-word clips under _static/words/
python3 tools/align_mfa.py --wav "$STATIC/en.wav" --text "$EN_TEXT" \
  --model-dir "$WORK/english_mfa" --dict "$WORK/english_mfa.dict" \
  --phones-txt "$WORK/english_mfa/phones.txt" --out "$WORK/en.TextGrid"
python3 tools/textgrid_to_word_wavs.py --wav "$STATIC/en.wav" \
  --textgrid "$WORK/en.TextGrid" --out-dir "$STATIC/words/en"

python3 tools/align_mfa.py --wav "$STATIC/zh.wav" --text "$ZH_TEXT" \
  --model-dir "$WORK/aishell3_pinyin/aishell3_pinyin_acoustic" \
  --dict "$WORK/aishell3_pinyin/aishell3_pinyin_dictionary.dict" \
  --phones-txt "$WORK/aishell3_pinyin/aishell3_pinyin_acoustic/phones.txt" \
  --tokenize pinyin --out "$WORK/zh-aishell3.TextGrid"
python3 tools/textgrid_to_word_wavs.py --wav "$STATIC/zh.wav" \
  --textgrid "$WORK/zh-aishell3.TextGrid" --out-dir "$STATIC/words/zh-aishell3"

python3 tools/align_mfa.py --wav "$STATIC/zh.wav" --text "$ZH_TEXT" \
  --model-dir "$WORK/large_pinyin/d_pinyin_acoustic" \
  --dict "$WORK/large_pinyin/large_pinyin_dictionary.dict" \
  --phones-txt "$WORK/large_pinyin/d_pinyin_acoustic/phones.txt" \
  --tokenize pinyin --out "$WORK/zh-large.TextGrid"
python3 tools/textgrid_to_word_wavs.py --wav "$STATIC/zh.wav" \
  --textgrid "$WORK/zh-large.TextGrid" --out-dir "$STATIC/words/zh-large"

echo "$0: audio ready under $STATIC/"
