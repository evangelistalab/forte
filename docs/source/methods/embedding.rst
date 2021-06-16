.. _`sec:methods:embedding`:

Active Space Embedding Theory
=======================================

.. codeauthor:: Nan He
.. sectionauthor:: Nan He

Simple active space frozen-orbital embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This embedding procedure provides an automatic way to embed one fragment into 
an environment, by an active space embedding theory that allows multireference method 
embedded in single-reference or multireference environment, for example, DSRG-MRPT2-in-CASSCF.

The input file should at least include two fragment::

    molecule {
      0 1 # Fragment 1, system A
      ...
      --
      0 1 # Fragment 2, environment or bath B
      ...

      symmetry c1 # Currently it is suggested to disable symmetry for embedding calculations
    }

In the forte options, turn on embedding procedure by adding options to forte::

    set forte{
      embedding    true
      embedding_cutoff_method    threshold # threshold/cum_threshold/num_of_orbitals
      embedding_threshold    0.5 # threshold t
    }

This is the minimum input required to run the embedding calculation. The embedding procedure will 
update the wavefunction coefficients and the MOSpaceInfo before running general forte calculations.

Four examples are available in test cases. Note that the program will by default semi-canonicalize 
frozen and active orbitals, if this is not intended, one can disable this semi-canonicalization with 
corresponding options.

EMBEDDING Options
~~~~~~~~~~~~~~~~~

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
* Default: 0

**EMBEDDING_REFERENCE**

The reference wavefunction, do not need to specify unless using special active space treatment.
Default is CASSCF with an well-defined active space including occupied and virtual orbitals.

* Type: string
* Default: CASSCF

**EMBEDDING_SEMICANONICALIZE_ACTIVE**

Turn on/off the semi-canonicalization of active space.

* Type: bool
* Default: true

**EMBEDDING_SEMICANONICALIZE_ACTIVE**

Turn on/off the semi-canonicalization of frozen core and virtual space. This will create a set of well-defined frozen orbitals.

* Type: bool
* Default: true

**NUM_A_DOCC**

The number of occupied orbitals fixed to system A, only function when EMBEDDING_CUTOFF_METHOD is NUM_OF_ORBITALS.

* Type: int
* Default: 0

**NUM_A_UOCC**

The number of virtual orbitals fixed to system A, only function when EMBEDDING_CUTOFF_METHOD is NUM_OF_ORBITALS.

* Type: int
* Default: 0

