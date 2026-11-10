GMMs and MLE/MAP estimation
===========================

Headers: :khg:`kaldi-hmm-gmm/csrc/diag-gmm.h`,
:khg:`kaldi-hmm-gmm/csrc/am-diag-gmm.h`,
:khg:`kaldi-hmm-gmm/csrc/mle-diag-gmm.h`,
:khg:`kaldi-hmm-gmm/csrc/mle-am-diag-gmm.h` — kaldi's
:kaldi:`src/gmm/ <src/gmm>` port.

.. code-block:: cpp

   khg::DiagGmm gmm;
   gmm.Read(is);
   float log_like = gmm.LogLikelihood(data);

   khg::AmDiagGmm am;
   am.AddPdf(gmm);

   khg::MleDiagGmmOptions opts;
   opts.min_variance = 0.01;
   khg::AccumDiagGmm acc(gmm, kGmmAll);
   // ... acc.Accumulate(...) ...
   float objf, count;
   khg::MleDiagGmmUpdate(opts, acc, kGmmAll, &gmm, &objf, &count);

API (doxygen)
-------------

.. doxygenclass:: khg::DiagGmm
   :members:

.. doxygenclass:: khg::AmDiagGmm
   :members:

.. doxygenstruct:: khg::MleDiagGmmOptions
   :members:

.. doxygenclass:: khg::AccumDiagGmm
   :members:
