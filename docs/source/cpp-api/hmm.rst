HMM topology and transition model
=================================

Headers: :khg:`kaldi-hmm-gmm/csrc/hmm-topology.h`,
:khg:`kaldi-hmm-gmm/csrc/transition-model.h`,
:khg:`kaldi-hmm-gmm/csrc/hmm-utils.h` — kaldi's
:kaldi:`src/hmm/ <src/hmm>` port.  ``TransitionModel`` implements
``kaldi_decoder::TransitionInformation``.

.. code-block:: cpp

   khg::HmmTopology topo;
   topo.Read(is);                       // kaldi topology text format

   khg::ContextDependency tree;
   ReadKaldiObject("tree", &tree);

   khg::TransitionModel trans_model(tree, topo);
   trans_model.TransitionIdToPhone(17);
   trans_model.IsSelfLoop(17);

API (doxygen)
-------------

.. doxygenclass:: khg::HmmTopology
   :members:

.. doxygenclass:: khg::TransitionModel
   :members:

.. doxygenstruct:: khg::HTransducerConfig
   :members:

.. doxygenfunction:: khg::GetHTransducer

.. doxygenfunction:: khg::AddTransitionProbs

.. doxygenfunction:: khg::AddSelfLoops

.. doxygenfunction:: khg::SplitToPhones
