.. _`sec:methods:avas`:

Atomic Valence Active Space (AVAS)
=======================================

.. codeauthor:: Chenxi Cai
.. sectionauthor:: Nan He

A simple and well-defined automated technique for constructing active orbital spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This AVAS procedure provides an automatic way to generate an active space for correlation 
computations by projecting MOs to an AO subspace, computing and sorting the overlaps for 
a new set of rotated MOs and a suitable active space.

In the forte options, turn on AVAS procedure by adding options to forte::

    set forte{
        subspace ["C(2px)","O(2px)"]                                                                                                                                                        avas True                                                                                                                                                                           avas_diagonalize true                                                                                                                                                               avas_sigma 1.0
    }

The subspace designate the AO subspace one wants to projected onto. This is an example input required 
to run the AVAS procedure for orbital projection. After AVAS, an automatically generated active space will 
be passed to following computations. Five examples are available in test cases.

AVAS Options
~~~~~~~~~~~~

