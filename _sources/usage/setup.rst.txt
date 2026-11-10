Environment setup
=================

This page sets up everything needed to run the examples.  **No C++ build is
required**: the ``kaldi-hmm-gmm`` wheel ships the compiled extension module
with the decoding engine linked in.

1. Install the packages
-----------------------

Python 3.8+ is required.  Then::

   pip install -U kaldi-hmm-gmm
   pip install kaldi-native-fbank kaldifst numpy soundfile librosa pypinyin

- ``kaldi-hmm-gmm`` — the GMM/HMM/tree library and the alignment/decoding
  API (compiled; self-contained, no ``kaldi-decoder`` package needed);
- ``kaldi-native-fbank`` — kaldi-compatible feature extraction (MFCC/pitch/
  CMVN/splice/LDA), used to compute the model features from the wav;
- ``kaldifst`` — FST construction (lexicon / training graphs);
- ``soundfile`` — wav reading;
- ``librosa`` — waveform resampling, only needed when a wav's sample rate
  differs from the model's (the test wavs are 24 kHz, the models expect
  16 kHz);
- ``pypinyin`` — pinyin lookup for the Mandarin examples (``--tokenize
  pinyin``).

2. Get the example scripts
--------------------------

The example drivers (``tools/align_mfa.py`` and the yesno recipe) live in
the repository — clone it (no build needed):

.. code-block:: bash

   git clone https://github.com/csukuangfj/kaldi-hmm-gmm
   cd kaldi-hmm-gmm

3. Verify
---------

.. code-block:: bash

   python3 -c "import kaldi_hmm_gmm as khg; print(khg.AlignConfig()); print(khg.TrainingGraphCompiler)"

You should see an ``AlignConfig`` object printed and the
``TrainingGraphCompiler`` class.  If ``TrainingGraphCompiler`` is missing,
your ``kaldi-hmm-gmm`` wheel is too old for these examples — install a
newer wheel or :doc:`build from source <../install>`.

Running with a from-source build
--------------------------------

If you are developing kaldi-hmm-gmm itself, :doc:`build it <../install>`
instead: the example scripts automatically prefer an in-repo build (they
fall back to the installed package when the repo has not been built).
