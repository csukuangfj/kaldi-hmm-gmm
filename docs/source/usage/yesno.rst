The yesno recipe — train a monophone GMM
========================================

:khg:`egs/yesno` mirrors kaldi's `egs/yesno/s5
<https://github.com/kaldi-asr/kaldi/tree/master/egs/yesno/s5>`_ stage by
stage: it trains a monophone GMM on the "yesno" corpus (60 recordings of
8-digit sequences), builds an HCLG decoding graph, decodes the test set and
reports WER — using only the ``kaldi_hmm_gmm`` Python API, **no kaldi
binaries**.

Prerequisites
-------------

1. :doc:`Build kaldi-hmm-gmm <../install>` (cmake build with tests passing).
2. Install the Python dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

   (``kaldifst``, ``kaldi_native_io``, ``numpy``, ``kaldilm``, ``kaldialign``)

Run everything
--------------

::

   cd egs/yesno
   ./run.sh

This runs stages 0–6 in order and prints kaldi-style log lines.  The final
lines look like::

   local/score.sh --cmd utils/run.pl data/test_yesno exp/mono0a/graph_tgpr exp/mono0a/decode_test_yesno
   local/score.sh: scoring with word insertion penalty=0.0,0.5,1.0
   %WER 0.43 [ 1 / 232, 1 ins, 0 del, 0 sub ] exp/mono0a/decode_test_yesno/wer_7_0.0

To run a single stage or a range::

   ./run.sh --stage 3 --stop-stage 3     # only the training loop
   ./run.sh --stage 6                    # only decoding + scoring

Stage by stage
--------------

**Stage 0 — prepare the data** (:khg:`local/prepare_data.sh <egs/yesno/local/prepare_data.sh>`).
Downloads the "yesno" corpus if ``waves_yesno/`` is missing, then splits it
into ``data/train_yesno`` (31 utterances) and ``data/test_yesno`` (29
utterances) with kaldi-style ``wav.scp`` / ``text`` / ``utt2spk`` /
``spk2utt``.  The wav file names encode the transcripts
(e.g. ``0_0_0_1_0_1_1_0.wav`` is the digit sequence ``zero zero zero one ...``).

The corpus is the OpenSLR "waves_yesno" release (the same one kaldi's
:kaldi:`egs/yesno/s5/run.sh <egs/yesno/s5/run.sh>` downloads):

- original URL (OpenSLR resource 1):
  http://www.openslr.org/resources/1/waves_yesno.tar.gz
- mirror actually fetched by the script (same tarball, hosted on github.com
  to reduce CI download time):
  https://github.com/csukuangfj/kaldi-hmm-gmm/releases/download/v1.1.4/waves_yesno.tar.gz

**Stage 1 — prepare the lang directory** (``prepare_lang.py``).
Builds ``data/lang``: the phone/word symbol tables (``phones.txt``,
``words.txt``), the lexicon FST ``L.fst`` with optional silence, its
disambiguated counterpart ``L_disambig.fst``, and the test-language G.fst
(``data/lang_test_tg``).  This is the kaldi ``utils/prepare_lang.sh``
equivalent.

**Stage 2 — compute features** (:khg:`local/compute_feats.py <egs/yesno/local/compute_feats.py>`).
Raw 13-dim MFCC (kaldi-compatible, ``mfcc/raw_mfcc_{train,test}_yesno.1.ark``)
plus per-speaker CMVN stats; ``data/{train,test}_yesno/feats.scp`` points at
the MFCC arks.  CMVN + deltas (→ 39-dim) are applied at load time
(``load_delta_feats`` in :khg:`tools/read_feats.py`), mirroring kaldi's
``apply-cmvn | add-deltas`` use-time pipeline.  The training/decoding code
itself is feature-agnostic — it just reads ``feats.scp``.

**Stage 3 — train the monophone GMM** (``train.py``).
The kaldi ``steps/train_mono.sh`` equivalent (40 iterations, ``--totgauss
400``, power 0.25): initialize the GMM from global stats, **align equally**
(kaldi ``align-equal-compiled`` semantics), then per iteration accumulate
statistics (``gmm-acc-stats-ali``), update (``gmm-est``), and realign on the
schedule used by kaldi.  Each iteration's model is written to
``exp/mono0a/<n>.mdl`` (kaldi model format: ``AmDiagGmm`` +
``TransitionModel`` + ``ContextDependency``) together with per-iteration
alignments ``ali.1.gz`` and Gaussian occupancies ``*.occs``.

**Stage 4 — final alignment** (``ali.py``).
Re-aligns the training set with the final model and exports the
"ali-to-phones" deliverables into ``exp/mono0a``:

===========================  ================================================================
File                         Content
===========================  ================================================================
``ali_full.ark``              the transition-id alignment itself
``ali_phones.ark``            per-frame phone-ids (``ark,t:`` format)
``ali_seq.ark``               collapsed phone sequence per utterance
``ali_phones_sym.txt``        per-frame phones with symbols (``SIL N Y ...``)
``ali_stats.txt``             log-likelihood / error summary
===========================  ================================================================

These match what kaldi produces with::

   ali-to-phones --per-frame=true exp/mono0a/final.mdl \
       "ark:gunzip -c exp/mono0a/ali.1.gz|" ark,t:- | utils/int2sym.pl -f 2-

**Stage 5 — build the decoding graph** (``mkgraph.py``).
The kaldi ``utils/mkgraph.sh`` equivalent: compose the lexicon with the test
grammar (``L·G``), expand context (``CLG``), compose with the HMM transducer
(``Ha``, transition probabilities baked), determinize (log) and minimize,
then add self-loops to obtain ``exp/mono0a/graph_tgpr/HCLG.fst`` — the same
chain kaldi's ``utils/mkgraph.sh`` runs.

**Stage 6 — decode and score** (``decode.py``).
Lattice-faster decoding (kaldi ``steps/decode.sh`` semantics, acoustic scale
0.083333 = kaldi's default acwt) of ``data/test_yesno`` with ``HCLG.fst``; the 1-best word sequence per
utterance is written to ``exp/mono0a/decode_test_yesno/text`` and scored
against the reference with ``kaldialign``.  The sweep over
word-insertion-penalties ``0,0.5,1.0`` (and lmwt) reproduces kaldi's
:kaldi:`local/score.sh <egs/yesno/s5/local/score.sh>`; the best result lands in ``exp/mono0a/decode_test_yesno/wer``.

Expected result
---------------

%WER 0.43 (1 insertion out of 232 words).  For reference, kaldi's own
recipe scores %WER 0.00 on this data; the residual error comes from
floating-point drift accumulated over the 40-iteration training loop (the
decode path itself is numerically exact: feeding kaldi's model, graph and
features through the decoders in this repo reproduces kaldi's output
exactly).  The recorded baseline
``egs/yesno/exp/parity/baseline-2026-11-07`` (kept locally; ``exp/`` is
git-ignored) holds the exact artifacts of this run for regression
comparisons.

Main outputs
------------

======================================  ==============================================
File                                    Content
======================================  ==============================================
``exp/mono0a/final.mdl``                transition model + trained GMM
``exp/mono0a/tree``                     context-dependency tree
``exp/mono0a/graph_tgpr/HCLG.fst``      decoding graph
``exp/mono0a/ali.1.gz``                 final transition-id alignments
``exp/mono0a/ali_phones.ark``           per-frame phone alignments
``exp/mono0a/decode_test_yesno/text``   decoded transcripts
``exp/mono0a/decode_test_yesno/wer``    WER summary
======================================  ==============================================

All of these are standard kaldi artifacts — ``final.mdl`` and ``HCLG.fst``
can be consumed by kaldi tools and by other kaldi-compatible libraries.
