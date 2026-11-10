HMM topology
============

- A non-emitting state has no pdf class associated with it: pdf class ``-1``
  (``kNoPdf``) means "no pdf class at all".  Valid pdf classes start from 0
  and are contiguous (``0 .. num_pdf_classes-1``).
- For an HMM topology, the forward pdf class and the self-loop pdf class are
  identical for every state (that is what makes it an HMM rather than a
  general transducer); see ``HmmTopology::IsHmm``.
