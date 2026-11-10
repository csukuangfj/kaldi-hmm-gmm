Topic notes
===========

Short notes on how the pieces fit together, with the math behind the GMM
utilities.

Model structure
---------------

- **DiagGmm** — for each pdf id there is a corresponding ``DiagGmm``.  A
  ``DiagGmm`` has 3 parameter sets: weights, means, variances.
- **AmDiagGmm** — a list of ``DiagGmm``, one per pdf id.  Use ``add_pdf()``
  to extend the list; it is initially empty.
- **AccumDiagGmm** — one per ``DiagGmm``; saves the accumulated statistics
  (weights, means, variances) used by the MLE/MAP updates.
- **AccumAmDiagGmm** — a list of ``AccumDiagGmm``; the list size equals the
  number of pdf ids.

.. toctree::
   :maxdepth: 2

   ./hmm
   ./diag-gmm
