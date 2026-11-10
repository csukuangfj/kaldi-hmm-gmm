Installation
============

Dependencies
------------

- CMake >= 3.13 and a C++17 compiler
- Python 3.8+ (for the Python bindings and the examples)
- Python packages::

   pip install -r requirements.txt

  (``kaldifst>=1.9.0``, ``kaldi_native_io``, ``kaldi-native-fbank`` for the
  MFA example, ``numpy``, ``kaldilm``, ``kaldialign``)

The C++ build fetches `kaldi-decoder <https://github.com/k2-fsa/kaldi-decoder>`_
(the decoding engine), `kaldifst <https://github.com/k2-fsa/kaldifst>`_,
Eigen, pybind11 and googletest automatically.  To use a local kaldi-decoder
checkout instead of the release tarball, pass
``-DKALDI_DECODER_SOURCE_DIR=/path/to/kaldi-decoder``.

Build from source
-----------------

::

   git clone https://github.com/csukuangfj/kaldi-hmm-gmm
   cd kaldi-hmm-gmm
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j4
   ctest --output-on-failure

The Python extension module is built into the package tree
(:khg:`kaldi-hmm-gmm/python/kaldi_hmm_gmm/lib/_kaldi_hmm_gmm.* <kaldi-hmm-gmm/python/kaldi_hmm_gmm/lib>`); add
:khg:`kaldi-hmm-gmm/python` to ``PYTHONPATH`` (the example scripts do this
themselves), or build a wheel::

   python setup.py bdist_wheel
   pip install dist/kaldi_hmm_gmm-*.whl

Note: ``kaldi_hmm_gmm`` is self-contained — it does **not** require the
``kaldi-decoder`` Python package at runtime (the decoder engine is compiled
in from C++).

Run the tests
-------------

From the build directory::

   ctest --output-on-failure

This runs the C++ gtests and the Python tests (``*_py``).  A single test can
be run with::

   ctest --output-on-failure -R <regexp>
