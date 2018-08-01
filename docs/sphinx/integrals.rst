
.. index::
   single: APIFCI
   pair: APIFCI; theory

.. _`sec:integrals`:

Selection of Integral
=====================

.. codeauthor:: Kevin P. Hannoni, Chenyang Li, Francesco A. Evangelista
.. sectionauthor:: Kevin P. Hannon and Francesco A. Evangelista

This section describes the THREE-DSRG-MRPT2 method in libadaptive.  This method is an efficient implementation of the DSRG-MRPT2 method.  The memory requirements are much less and this allows applications up to 1000 basis functions.

Conventional integrals
^^^^^^^^^^^^^^^^^^^^^^

Density Fitting (DF) and Cholesky Decomposition (CD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a reminder, Density-Fitting and Cholesky decomposition approaches seek to factorize the two integral integrals.

.. math:: \langle ij || ab \rangle = b_{ia}^{Q}b_{jb}^{Q} - b_{ib}^{Q}b_{ja}^{Q}

Note: The equations in implemented in this method use physicist notation for the two electron integrals, but the DF/CD literature use chemist notation.  The main difference between DF and CD is the formation of the B tensors.
In DF, the b tensor is defined as

.. math:: b_{pq}^{Q} = \sum_p (pq | P)[(P | Q)^{-1/2}]_{PQ}

and P and Q refer to the auxiliary basis set.

In the CD approach, the b tensor is formed by performing a cholesky decomposition of the two electron integrals.  The accuracy of this decomposition is determined by a user defined tolerance.  The accuracy of the two electron integral is directly determined by the tolerance.

Integral Selection Keywords
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These keywords are actually controlled by the integral class so they affect all areas of the code, not just THREE-DSRG-MRPT2.

**INT_TYPE**

If one is going to use THREE-DSRG-MRPT2, this keyword needs to be set to either DF or CHOLESKY.
The default value for integrals is actually conventional, but the code won't work if conventional is used.

INT_TYPE tells what type of integrals will be used in the calculation

* Type: string

* Possible Values: DF, CHOLESKY

* Default: CONVENTIONAL

**CHOLESKY_TOLERANCE**

The tolerance for the cholesky decomposition.  This keyword determines the accuracy of the computation.
A smaller tolerance is a more accurate computation.
The tolerance for the cholesky decomposition:

* Type: double in scientific notation (ie 1e-5 or 0.0001)
* Default:1.0e-6

**DF_BASIS_MP2**

The type of basis set used for DF.  This keyword needs to be placed in the globals section.
A common advice for these is to use the basis set designed for the primary basis set, ie cc-pVDZ should use cc-pVDZ-RI.

* Type: string specifing basis set

* Default: none


**ACI_SPIN_ANALYSIS**

Type: Boolean

Default value: False

Do spin correlation analysis



.. include:: autodoc_abbr_options_c.rst

.. index::
   single: CASSCF
   pair: CASSCF; theory

.. _`sec:CASSCF`:
