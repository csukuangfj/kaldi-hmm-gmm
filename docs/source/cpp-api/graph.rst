Trees and training graphs
=========================

Headers: :khg:`kaldi-hmm-gmm/csrc/context-dep.h`,
:khg:`kaldi-hmm-gmm/csrc/event-map.h`,
:khg:`kaldi-hmm-gmm/csrc/training-graph-compiler.h` — kaldi's
:kaldi:`src/tree/ <src/tree>` and
:kaldi:`src/decoder/training-graph-compiler.h
<src/decoder/training-graph-compiler.h>` port.

.. code-block:: cpp

   khg::TrainingGraphCompilerOptions opts;   // scales 0.0: add transition
   opts.transition_scale = 0.0;              // probs at alignment time
   opts.self_loop_scale = 0.0;

   khg::TrainingGraphCompiler gc(trans_model, tree, lex_fst,
                                 disambig_syms, opts);
   fst::VectorFst<fst::StdArc> graph;
   gc.CompileGraphFromText(word_ids, &graph);

API (doxygen)
-------------

.. doxygenclass:: khg::TrainingGraphCompiler
   :members:

.. doxygenclass:: khg::ContextDependency
   :members:

.. doxygenclass:: khg::EventMap
   :members:
