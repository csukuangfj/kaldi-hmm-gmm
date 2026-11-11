# Introduction

[Kaldi][kaldi]'s HMM-GMM reimplementation with [Eigen][eigen]: diagonal-covariance
GMMs, HMM topology and transition model, decision trees / context dependency,
training-graph compilation, MLE/MAP estimation, and GMM decodables for
**forced alignment** and **decoding** — from C++ and Python, without any
kaldi binaries.

The decoding engine (lattice decoders, decoder wrappers, lattice best path)
lives in [kaldi-decoder][kaldi-decoder]; this library links its C++ core and
implements its `DecodableInterface` / `TransitionInformation`.

[kaldi]: https://github.com/kaldi-asr/kaldi
[eigen]: https://eigen.tuxfamily.org/index.php
[kaldi-decoder]: https://github.com/k2-fsa/kaldi-decoder

# Build

```bash
git clone https://github.com/csukuangfj/kaldi-hmm-gmm
cd kaldi-hmm-gmm
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
ctest --output-on-failure
```

All C++ dependencies (kaldi-decoder, kaldifst/openfst, Eigen, pybind11,
googletest) are fetched automatically. To build a Python wheel instead:

```bash
python setup.py bdist_wheel
pip install dist/kaldi_hmm_gmm-*.whl
```

The Python package `kaldi_hmm_gmm` is self-contained — it does not require
the `kaldi-decoder` Python package at runtime.

# Python API

```python
import kaldi_hmm_gmm as khg

tm, am = khg.read_model("final.alimdl")     # kaldi model format
tree = khg.read_tree("tree")

decodable = khg.DecodableAmDiagGmmScaled(am=am, tm=tm, feats=feats, scale=0.1)

cfg = khg.AlignConfig(beam=10.0, retry_beam=40.0)   # kalpy/MFA defaults
num_done, num_error, _, like, frames, alignment, words = khg.align_utterance(
    config=cfg, utt="utt", acoustic_scale=0.1,
    transition_scale=1.0, self_loop_scale=0.1,
    trans_model=tm, fst=training_graph, decodable=decodable)
```

See the documentation for the full API and two complete tutorials.

# Examples

- **yesno** ([`egs/yesno`](egs/yesno)) — the kaldi `egs/yesno/s5` recipe,
  stage by stage: train a monophone GMM (40 iterations), build HCLG, decode,
  score. Run it with `cd egs/yesno && ./run.sh`. The corpus is the OpenSLR
  [waves_yesno](http://www.openslr.org/resources/1/waves_yesno.tar.gz)
  release; the script fetches the same tarball from a
  [github.com mirror](https://github.com/csukuangfj/kaldi-hmm-gmm/releases/download/v1.1.4/waves_yesno.tar.gz)
  to reduce CI download time.
- **MFA forced alignment** ([`tools/align_mfa.py`](tools/align_mfa.py)) —
  align a single utterance to its transcript with pretrained MFA acoustic
  models (English `english_mfa`, Chinese `aishell3_pinyin` /
  `large_pinyin`) and produce Praat TextGrids.  Test audio: [en.wav](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav) /
  [zh.wav](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/zh.wav).
  Every run reports the audio duration, elapsed seconds and the real-time
  factor (RTF); the `.github/workflows/mfa.yaml` CI runs both languages and
  uploads the TextGrids next to the audio.

Both are documented in detail in the documentation (`docs/source/usage/`),
including environment setup (no C++ build needed — `pip install
kaldi-hmm-gmm`) and copy-paste runnable commands.

# Colab notebooks

Run the MFA examples in your browser:

- MFA forced alignment — English [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csukuangfj/kaldi-hmm-gmm/blob/master/mfa_english.ipynb)
- MFA forced alignment — Chinese [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csukuangfj/kaldi-hmm-gmm/blob/master/mfa_chinese.ipynb)

# Documentation

```bash
cd docs
pip install -r requirements.txt   # + doxygen
make html
```

Then open `docs/build/html/index.html`. The C++ API reference is generated
by doxygen (linked from the C++ API pages) and the Python API is rendered
with autodoc.

# See also

- [kaldi-decoder][kaldi-decoder] — kaldi's `src/decoder`/`src/lat` port
- [kaldifst](https://github.com/k2-fsa/kaldifst) — openfst Python bindings
  with zero-copy interop
- [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank) —
  kaldi-compatible feature extraction
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — uses this library
  (C++) for GMM forced alignment
