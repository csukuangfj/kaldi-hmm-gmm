Graphs, HMM utilities and trees
===============================

Training-graph compiler
-----------------------

.. autoclass:: kaldi_hmm_gmm.TrainingGraphCompiler
   :members:

.. autoclass:: kaldi_hmm_gmm.TrainingGraphCompilerOptions

``TrainingGraphCompiler.compile_graph_from_text`` turns a word sequence into
a training graph (kaldi's ``compile-train-graphs``).  Compile with
``transition_scale=0.0, self_loop_scale=0.0`` and let the aligner add the
transition probabilities once (kaldi semantics).

HMM utilities (kaldi ``hmm-utils.h``)
-------------------------------------

.. autofunction:: kaldi_hmm_gmm.get_h_transducer

.. autoclass:: kaldi_hmm_gmm.HTransducerConfig

.. autofunction:: kaldi_hmm_gmm.add_transition_probs

.. autofunction:: kaldi_hmm_gmm.add_self_loops

.. autofunction:: kaldi_hmm_gmm.split_to_phones

.. autofunction:: kaldi_hmm_gmm.get_pdfs_for_phones

.. autofunction:: kaldi_hmm_gmm.get_pdf_to_phones_map

``split_to_phones`` splits a transition-id alignment into per-phone segments
(the core of kaldi's ``ali-to-phones``).

Decision trees / context dependency
-----------------------------------

.. autoclass:: kaldi_hmm_gmm.ContextDependency
   :members:

.. autoclass:: kaldi_hmm_gmm.ContextDependencyInterface

.. autofunction:: kaldi_hmm_gmm.monophone_context_dependency

.. autofunction:: kaldi_hmm_gmm.monophone_context_dependency_shared

Event maps
----------

.. autoclass:: kaldi_hmm_gmm.EventMap

.. autoclass:: kaldi_hmm_gmm.ConstantEventMap

Clusterable objects (used by tree building)
-------------------------------------------

.. autoclass:: kaldi_hmm_gmm.Clusterable

.. autoclass:: kaldi_hmm_gmm.ScalarClusterable

.. autoclass:: kaldi_hmm_gmm.GaussClusterable
