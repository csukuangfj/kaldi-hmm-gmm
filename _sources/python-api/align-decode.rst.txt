Alignment and decoding
======================

Forced alignment (kaldi ``gmm-align-compiled``)
-----------------------------------------------

.. autoclass:: kaldi_hmm_gmm.AlignConfig

.. autofunction:: kaldi_hmm_gmm.align_utterance

``align_utterance`` adds the transition probabilities to the training graph
(``AddTransitionProbs``) and runs the Viterbi alignment in a single C++ call.
It mutates the graph in place (same as kaldi's ``gmm-align-compiled``) and
returns ``(num_done, num_error, num_retried, tot_like, frame_count,
alignment, words)``; ``alignment`` holds transition-ids.

Typical use::

   cfg = khg.AlignConfig(beam=10.0, retry_beam=40.0)
   num_done, num_error, _, like, frames, alignment, words = khg.align_utterance(
       config=cfg, utt="utt", acoustic_scale=0.1,
       transition_scale=1.0, self_loop_scale=0.1,
       trans_model=tm, fst=graph, decodable=decodable)

Decoding (kaldi ``gmm-latgen-*``)
---------------------------------

.. autoclass:: kaldi_hmm_gmm.LatticeFasterDecoderConfig

.. autoclass:: kaldi_hmm_gmm.LatticeSimpleDecoderConfig

.. autofunction:: kaldi_hmm_gmm.decode_utterance_lattice_faster

.. autofunction:: kaldi_hmm_gmm.decode_utterance_lattice_simple

.. autofunction:: kaldi_hmm_gmm.lattice_best_path

``decode_utterance_*`` returns ``(ok, alignment, words, like, lat)`` where
``lat`` is the lattice exactly as kaldi writes it (acoustic costs divided by
the acoustic scale); feed it to ``lattice_best_path`` for 1-best under
lm/acoustic scaling (kaldi's ``lattice-best-path``).

Decodables
----------

.. autoclass:: kaldi_hmm_gmm.DecodableAmDiagGmmScaled
   :members:

.. autoclass:: kaldi_hmm_gmm.DecodableAmDiagGmmUnmapped

.. autoclass:: kaldi_hmm_gmm.DecodableInterface

.. autoclass:: kaldi_hmm_gmm.TransitionInformation

``DecodableInterface`` and ``TransitionInformation`` accept Python
subclasses (a pybind11 trampoline keeps virtual dispatch on the Python
side), so custom acoustic scorers or transition models can be plugged into
the alignment/decoding functions.
