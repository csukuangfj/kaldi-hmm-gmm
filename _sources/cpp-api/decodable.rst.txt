GMM decodables
==============

Header: :khg:`kaldi-hmm-gmm/csrc/decodable-am-diag-gmm.h` — kaldi's
:kaldi:`src/gmm/decodable-am-diag-gmm.h
<src/gmm/decodable-am-diag-gmm.h>` port.  Both classes implement
``kaldi_decoder::DecodableInterface`` and can be passed directly to
kaldi-decoder's alignment/decoding functions.

.. code-block:: cpp

   khg::DecodableAmDiagGmmScaled decodable(am, trans_model, feats, 0.1f);

   kaldi_decoder::LatticeFasterDecoder decoder(fst, config);
   kaldi_decoder::DecodeUtteranceLatticeFaster(
       decoder, decodable, trans_model, utt, /*allow_partial=*/false,
       /*acoustic_scale=*/0.1f, &alignment, &words, &log_like, &lat);

API (doxygen)
-------------

.. doxygenclass:: khg::DecodableAmDiagGmmScaled
   :members:

.. doxygenclass:: khg::DecodableAmDiagGmmUnmapped
   :members:
