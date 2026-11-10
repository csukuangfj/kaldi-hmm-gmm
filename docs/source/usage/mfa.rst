MFA-style forced alignment
==========================

:khg:`tools/align_mfa.py` force-aligns a single utterance to its known
transcript using **pretrained MFA acoustic models** (kaldi format) and writes
a Praat **TextGrid** with phone and word tiers — no MFA installation and no
kaldi binaries involved.  The pipeline reproduces MFA/kalpy conventions
exactly: kaldi-compatible features (computed with `kaldi-native-fbank
<https://github.com/csukuangfj/kaldi-native-fbank>`_), a lexicon FST with
optional silence, a training graph compiled from the known text, and
``gmm-align-compiled``-style alignment.

Complete, copy-paste runnable tutorials:

.. toctree::
   :maxdepth: 2

   mfa-english
   mfa-mandarin

Each tutorial embeds the test audio so you can listen to it, and shows the
exact command, the expected output (including the real-time factor), and
what the resulting TextGrid contains.

What a model directory contains
-------------------------------

======================  =========================================================
File                    Role
======================  =========================================================
``final.alimdl``        alignment model used by the aligner (``read_model``)
``final.mdl``           full acoustic model
``tree``                context-dependency tree (monophone here)
``lda.mat``             LDA transform applied after CMVN + splicing
``meta.json``           feature configuration (MFCC/pitch/CMVN/LDA options)
``phones.txt``          phone symbol table (optional ``--phones-txt``)
======================  =========================================================

The feature chain is selected by ``meta.json``: MFCC → (processed pitch,
concatenated, if ``use_pitch``) → utterance CMVN → splice ±3 frames + LDA
(or deltas) → the model's feature dimension (usually 40).  If the wav's
sample rate differs from the model's ``sample_frequency``, the waveform is
resampled first (librosa; MFA's ``allow_downsample``/``allow_upsample``
semantics).

.. important::
   Waveforms are read at **raw int16 amplitude scale** (do NOT divide by
   32768) — kaldi's wav reader keeps integer magnitudes and the models were
   trained on that scale.  ``align_mfa.py``'s ``read_wave`` does this for you.

Command-line options
--------------------

===========================  ======================================================
Option                       Meaning
===========================  ======================================================
``--wav``                    input wav (mono, 16-bit PCM)
``--text``                   the transcript of the utterance
``--model-dir``              MFA model directory (see above)
``--dict``                   pronunciation dictionary (comma-separated list
                             allowed; earlier files win)
``--phones-txt``             optional phone symbol table (defaults to the
                             model's ``meta.json`` phone order)
``--tokenize``               ``space`` (default, whitespace words), ``char``
                             (longest-match word segmentation, Chinese), or
                             ``pinyin`` (pinyin lookup via ``pypinyin``,
                             display = hanzi)
``--pitch``                  force ``use_pitch`` on (needed for the official
                             MFA Mandarin model)
``--out``                    output TextGrid path
===========================  ======================================================

Unknown words fall back to the model's OOV phone (``spn``); a phone missing
from the phone table is an error.

Every run prints a timing line — the audio duration, the wall-clock seconds
of the pipeline (features + graph + alignment + TextGrid; one-time
initialization such as model/dictionary loading is **not** counted), and the
real-time factor (RTF = elapsed / audio duration; lower is faster):

.. code-block:: text

   timing: audio duration 4.76s, elapsed 0.75s, RTF 0.158

The TextGrid output
-------------------

The TextGrid has two interval tiers:

- **phones** — one interval per phone instance.  Adjacent instances of the
  same phone stay separate (e.g. ``here`` = ``...+e`` and ``and`` = ``e+...``
  produce distinct intervals);
- **words** — one interval per word, spanning exactly its pronunciation's
  phone instances (leading/between-word silence stays unattributed).

Open the file in `Praat <https://praat.org/>`_ (or any TextGrid viewer) to
inspect the alignment alongside the audio.

How it works
------------

The tool chains these steps (each mirroring an MFA/kalpy stage):

1. **Features** (``kaldi-native-fbank`` step operators, configured from
   ``meta.json``): ``compute_model_features`` in :khg:`tools/align_mfa.py`.
2. **Lexicon FST** from the dictionary, with optional silence ``sil`` at
   word boundaries (probability from the model's ``silence_probability``).
3. **Training graph** — ``khg.TrainingGraphCompiler`` compiles the known
   word sequence into an alignment graph (transition probabilities are added
   once, at alignment time, exactly like kaldi's ``compile-train-graphs`` +
   ``gmm-align-compiled`` pair).
4. **Alignment** — ``gmm_align_compiled`` (:khg:`tools/align.py`) with the
   kalpy/MFA defaults: ``transition_scale=1.0``, ``acoustic_scale=0.1``,
   ``self_loop_scale=0.1``, ``beam=10``, ``retry_beam=40``.
5. **TextGrid** from the transition-id alignment via
   ``khg.split_to_phones``.
