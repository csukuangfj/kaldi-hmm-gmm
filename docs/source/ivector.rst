i-Vectors
=========

This page explains what an i-vector is, how it is computed, and how the
classes in this package (:class:`IvectorExtractor`,
:class:`OnlineIvectorEstimationStats`, :class:`OnlineIvectorExtractor`)
implement the computation.  The math follows kaldi's
:kaldi:`ivector/ivector-extractor.h <src/ivector/ivector-extractor.h>`,
:kaldi:`ivector/ivector-extractor.cc <src/ivector/ivector-extractor.cc>` and
:kaldi:`online2/online-ivector-feature.cc
<src/online2/online-ivector-feature.cc>` exactly; the port is validated
frame-by-frame against kaldi (see :khg:`tests/parity/parity_ivector.py`).

What problem i-Vectors solve
----------------------------

The acoustic features of the same word depend strongly on *who* is speaking
and *how* it was recorded (microphone, room, channel).  A speech recognizer
trained on many speakers therefore sees each phone smeared across many
slightly-different feature distributions.

An **i-vector** is a single, low-dimensional vector (typically 40–400
dimensions) that summarizes the speaker and channel of one utterance.  The
recognizer's acoustic model takes it as an extra input and can compensate for
the mismatch.  It is called *total variability* modeling: one latent space
carries both speaker and channel variability (Dehak et al., 2011).

The generative model
--------------------

Notation (kaldi uses SGMM-style letters throughout its code):

* :math:`D` — feature dimension (e.g. 40 MFCCs after transform)
* :math:`I` — number of Gaussians in the universal background model (UBM)
* :math:`S` — i-vector dimension
* :math:`\mathbf{x}` — one feature frame

Each UBM Gaussian :math:`g` has a mean :math:`\mathbf{m}_g` and
(diagonal) covariance :math:`\Sigma_g`.  When speaker :math:`\mathbf{y}` is
present, the mean of Gaussian :math:`g` is shifted linearly:

.. math::

    \boldsymbol{\mu}_g(\mathbf{y}) = \mathbf{m}_g + M_g \mathbf{y},

where :math:`M_g` is a :math:`D \times S` matrix ("mean subspace", ``M_`` in
the code) and :math:`\mathbf{y}` is the **i-vector**.  The prior over
i-vectors is standard normal except for the first dimension, which has offset
:math:`o` (``prior_offset_``, e.g. 33.1 — this encodes the global offset of
the adapted means so the subspace can stay zero-mean):

.. math::

    p(\mathbf{y}) = \mathcal{N}\!\left(\begin{bmatrix}o\\ \mathbf{0}\end{bmatrix},
    I\right).

At test time we never know :math:`\mathbf{y}`; we compute its **posterior
mean** from the features of an utterance — that estimate is the i-vector.

How it is computed
------------------

Step 1: Gaussian posteriors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every frame is assigned soft posteriors over the :math:`I` Gaussians using the
UBM (a diagonal-covariance GMM, e.g. ``final.dubm``):

.. math::

    \gamma_{tg} = \frac{p_g\,\mathcal{N}(\mathbf{x}_t; \mathbf{m}_g, \Sigma_g)}
                       {\sum_{g'} p_{g'}\,\mathcal{N}(\mathbf{x}_t; \mathbf{m}_{g'}, \Sigma_{g'})}.

In practice the posterior is pruned: keep the top-``num_gselect`` Gaussians
and/or drop entries below ``min_post`` of the total
(:func:`vector_to_posterior_entry`).

Step 2: sufficient statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accumulate per-Gaussian zeroth- and first-order statistics over the utterance
(:class:`IvectorExtractorUtteranceStats`):

.. math::

    \gamma_g = \sum_t \gamma_{tg}, \qquad
    \mathbf{X}_g = \sum_t \gamma_{tg}\, \mathbf{x}_t.

Step 3: the posterior over :math:`\mathbf{y}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Marginalizing the Gaussians, the posterior :math:`p(\mathbf{y}\,|\,\text{utt})`
is Gaussian with precision and linear term

.. math::

    \Lambda = I + \sum_g \gamma_g\, M_g^\top \Sigma_g^{-1} M_g, \qquad
    \boldsymbol{\ell} = \sum_g M_g^\top \Sigma_g^{-1} \mathbf{X}_g
                        + \begin{bmatrix}o\\ \mathbf{0}\end{bmatrix},

and therefore

.. math::

    \hat{\mathbf{y}} = \Lambda^{-1} \boldsymbol{\ell}.

The code precomputes :math:`U_g = M_g^\top \Sigma_g^{-1} M_g` and
:math:`\Sigma_g^{-1} M_g` once (``ComputeDerivedVars``) so that estimating an
i-vector is only a few matrix-vector products plus one linear solve.

Step 4: solving for :math:`\hat{\mathbf{y}}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two options:

* **Exact**: invert :math:`\Lambda` (or solve the linear system).  Used by
  :meth:`IvectorExtractor.compute_ivector`.
* **Conjugate gradient**, warm-started from the previous estimate — the
  streaming choice.  :meth:`OnlineIvectorEstimationStats.get_ivector` runs
  :kaldi:`LinearCgd <src/matrix/optimization.cc>` for ``num_cg_iters``
  iterations (kaldi suggests 15;
  pass a negative value to solve exactly).  Warm starts matter: every
  re-estimate begins near the solution, so a handful of iterations suffices.

Streaming estimation
--------------------

In online decoding the i-vector is re-estimated every
``ivector_period`` frames (e.g. 10) using **all frames seen so far**.  Rather
than re-accumulating statistics from scratch, :class:`OnlineIvectorEstimationStats`
maintains :math:`\Lambda` and :math:`\boldsymbol{\ell}` incrementally: each
frame adds :math:`\gamma_{tg} U_g` and :math:`M_g^\top \Sigma_g^{-1}
(\gamma_{tg}\mathbf{x}_t)`.

**Prior scaling (``max_count``).**  As the utterance grows, the data term
overwhelms the unit prior and long utterances produce atypical
("too confident") i-vectors.  With ``max_count`` (vosk uses 100) the prior
term is weighted by :math:`\max(n, c)/c`, where :math:`n` is the frame count
and :math:`c` is ``max_count``: up to :math:`c` frames nothing changes, beyond
it the prior pulls the estimate back toward typical.

**Output convention.**  Both kaldi's
:kaldi:`ivector-extract-online <src/ivectorbin/ivector-extract-online.cc>` and
its runtime class
:kaldi:`OnlineIvectorFeature <src/online2/online-ivector-feature.cc>` remove
the prior offset from the first
dimension of the output (``y[0] -= prior_offset``) so the vector is
approximately zero-mean — this is the vector the acoustic model expects.
:class:`OnlineIvectorExtractor` follows the same convention (the raw
estimate keeps the offset internally as the CG starting point).

The full pipeline of a vosk model
---------------------------------

:class:`OnlineIvectorExtractor` reproduces kaldi's
``OnlineIvectorFeature`` dataflow:

.. code-block:: text

   raw MFCC ──┬── running CMVN → splice(±3) → lda (final.mat) → UBM posteriors
              │                                              │
              └── splice(±3) → lda ──────────────────────────┴─→ stats → y-hat

Note the asymmetry (it is kaldi's, deliberately): the UBM **posteriors** are
computed on the CMVN-normalized chain, while the **first-order statistics**
accumulate the un-normalized transformed features (unless the
``online_cmvn_iextractor`` option is set; vosk's models leave it off).

Model files
-----------

A trained i-vector model directory (e.g. a vosk model's ``ivector/``):

==============================  =====================================================
File                            Role
==============================  =====================================================
``final.ie``                    The extractor: :math:`M_g`, :math:`\Sigma_g^{-1}`,
                                ``prior_offset`` (and optionally weight
                                projections :math:`\mathbf{w}_g`)
``final.dubm``                  The UBM (diagonal GMM) used for posteriors
``final.mat``                   LDA-like linear transform of spliced features
``global_cmvn.stats``           Global CMVN stats ``[2, D+1]``, seeds running CMVN
``splice.conf``                 Splice context (usually ``--left-context=3
                                --right-context=3``)
``online_cmvn.conf``            Running-CMVN options (may be empty ⇒ defaults)
==============================  =====================================================

Python example
--------------

.. code-block:: python

   import kaldi_hmm_gmm as khg
   import kaldi_native_io

   extractor = khg.IvectorExtractor.read("ivector/final.ie")
   ubm = khg.read_diag_gmm("ivector/final.dubm")
   lda = kaldi_native_io.FloatMatrix.read("ivector/final.mat").numpy()
   gs = kaldi_native_io.DoubleMatrix.read("ivector/global_cmvn.stats").numpy()

   cfg = khg.OnlineIvectorExtractorConfig(
       left_context=3, right_context=3,
       lda_mat=lda, global_cmvn_stats=gs,
       max_count=100, num_cg_iters=15, ivector_period=10)

   eng = khg.OnlineIvectorExtractor(cfg, extractor, ubm)
   eng.accept_features(mfcc)          # [T, 40] float32, from int16-scale audio
   eng.flush()                        # end of utterance
   print(eng.get_ivectors().shape)    # [ceil(T/10), ivector_dim]

Streaming, as kaldi's online2 decoders and vosk do it
-----------------------------------------------------

:class:`OnlineIvectorFeature` is a port of kaldi's
:kaldi:`OnlineIvectorFeature <src/online2/online-ivector-feature.h>` itself:
stats are accumulated lazily up to the frame asked for, and with
``use_most_recent_ivector`` (default) :meth:`OnlineIvectorFeature.get_frame`
returns the estimate at that frame.  Optional frame weights implement
kaldi's silence weighting: :class:`OnlineSilenceWeighting` turns the
decoder's current best path (one transition-id per decoder frame, e.g. from
kaldi-decoder's ``LatticeFasterOnlineDecoder.get_best_path_transition_ids``)
into delta-weights.  ``OnlineIvectorExtractorConfig`` defaults are kaldi's
(``num_gselect=5``, ``min_post=0.025``, ``posterior_scale=0.1``); vosk adds
``max_count=100`` and silence weight ``0.001``.

.. code-block:: python

   cfg = khg.OnlineIvectorExtractorConfig(
       left_context=3, right_context=3, lda_mat=lda, global_cmvn_stats=gs,
       max_count=100)
   ivector = khg.OnlineIvectorFeature(cfg, extractor, ubm)
   silence = khg.OnlineSilenceWeighting(
       trans_model,
       khg.OnlineSilenceWeightingConfig("1:2:3:4:5:6:7:8:9:10", 0.001),
       frame_subsampling_factor=3)

   for mfcc_chunk in stream:                       # every 0.2 s of audio
       ivector.accept_features(mfcc_chunk)
       if ivector.num_frames_ready > 0:
           silence.compute_current_traceback(
               decoder.get_best_path_transition_ids(False))
           ivector.update_frame_weights(
               silence.get_delta_weights(ivector.num_frames_ready))
       # for each nnet chunk: the latest i-vector
       iv = ivector.get_frame(ivector.num_frames_ready - 1)

Parity: ``tests/parity/parity_online_ivector_feature.py`` compares against
kaldi's ``ivector-extract-online2`` with and without frame weights.

For a batch/offline computation with explicit posteriors, see
:class:`IvectorExtractorUtteranceStats` and
:meth:`IvectorExtractor.compute_ivector`.

References
----------

* N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, P. Ouellet,
  *"Front-end factor analysis for speaker verification"*, IEEE TASLP, 2011.
* P. Kenny, *"A small-footprint i-vector extractor"*, Odyssey 2012.
* D. Povey et al., *"The subspace Gaussian mixture model — a structured model
  for speech recognition"*, Computer Speech & Language, 2011 (the SGMM
  notation used throughout kaldi's i-vector code).
