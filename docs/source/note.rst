Note
====

DiagGmm
-------

For each pdf id, there is a corresponding DiagGmm.

A DiagGmm has 3 parameters: weights, means, variances.


AmDiagGmm
---------

It contains a list of DiagGmm. We use ``add_pdf()`` to update the list.
The list is initially empty.

AccumDiagGmm
------------

There is an AccumDiagGmm For each DiagGmm.
An AccumDiagGmm saves the statistics for a DiagGmm, e.g., the weights, means,
and variances accumulated so far.


AccumAmDiagGmm
--------------

An AccumAmDiagGmm contains a list of AccumDiagGmm. The size of the list
equals to number of pdf ids.
