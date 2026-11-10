Python API
==========

The Python package ``kaldi_hmm_gmm`` exposes the C++ API through pybind11
bindings.  It is self-contained: the decoding engine is compiled in, so no
other kaldi packages need to be installed at runtime.

.. toctree::
   :maxdepth: 2

   model
   gmm
   graph
   align-decode
   reference
