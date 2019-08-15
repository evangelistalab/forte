.. _`sec:methods:embedding`:

Active Space Embedding Theory
=======================================

.. codeauthor:: Nan He
.. sectionauthor:: Nan He

Basic EMBEDDING
^^^^^^^^^^

EMBEDDING Options
~~~~~~~~~~~~

**EMBEDDING**

Turn on/off embedding procedure.

* Type: bool
* Default: false

**EMBEDDING_CUTOFF_METHOD**
The choices of embedding cutoff methods.
THRESHOLD: simple threshold
CUM_THRESHOLD: cumulative threshold
NUM_OF_ORBITALS: fixed number of orbitals

* Type: string
* Options: THRESHOLD, CUM_THRESHOLD, NUM_OF_ORBITALS
* Default: THRESHOLD

**EMBEDDING_THRESHOLD**

The threshold :math:`t` of embedding cutoff.
Do nothing when EMBEDDING_CUTOFF_METHOD is NUM_OF_ORBITALS

* Type: double
* Default: 0.5

