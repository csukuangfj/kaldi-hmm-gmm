Models, topology and trees
==========================

Reading and writing kaldi-format models
---------------------------------------

.. autofunction:: kaldi_hmm_gmm.read_model

.. autofunction:: kaldi_hmm_gmm.write_model

.. autofunction:: kaldi_hmm_gmm.read_tree

``read_model`` returns a ``(TransitionModel, AmDiagGmm)`` tuple — the same
two objects kaldi stores in ``final.mdl``.

Transition model
----------------

.. autoclass:: kaldi_hmm_gmm.TransitionModel
   :members:

.. autoclass:: kaldi_hmm_gmm.TransitionModelTuple

.. autoclass:: kaldi_hmm_gmm.MleTransitionUpdateConfig

HMM topology
------------

.. autoclass:: kaldi_hmm_gmm.HmmTopology
   :members:

Tree rendering
--------------

.. autofunction:: kaldi_hmm_gmm.draw_tree

``draw_tree`` renders a ``ContextDependency`` tree as graphviz dot text
(kaldi's ``draw-tree``).
