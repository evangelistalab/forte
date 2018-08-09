
.. index::
   single: Integrals
   pair: Integrals; theory

.. _`sec:integrals`:

Two-electron Integral Types
===========================

.. codeauthor:: Kevin P. Hannon, Chenyang Li, Francesco A. Evangelista
.. sectionauthor:: Kevin P. Hannon and Francesco A. Evangelista

Forte can handle different types of exact and approximate two-electron integrals.
This section describes the various options available and their properties/limitations.
The selection of different integral types is controlled by the option ``INT_TYPE``

Conventional integrals
^^^^^^^^^^^^^^^^^^^^^^

Conventional integrals are the default choice for Forte. When this option is selected,
Forte will compute and store the two-electron integrals in the molecular orbital (MO) basis :math:`\phi_p`.

.. math:: \langle pq | rs \rangle = \int dx_1 dx_2 \phi_p(x_1)^* \phi_q(x_2)^* r_{12}^{-1} \phi_r(x_1) \phi_s(x_2)

These integrals are computed with Psi4's ``IntegralTrasform`` class and written to disk.
Forte will store three copies of these integrals, the antisymmetrized alpha-alpha and beta-beta integrals

.. math:: \langle p_\alpha q_\alpha \| r_\alpha s_\alpha \rangle,  \langle p_\beta q_\beta \| r_\beta s_\beta \rangle,

and the alpha-beta integrals (not antisymmetrized)

.. math:: \langle p_\alpha q_\beta | r_\alpha s_\beta \rangle,

for all values of  :math:`p, q, r, s`.
Storage of these integrals has a memory cost equal to :math:`3 N^4`, where :math:`N` is the number of orbitals
that are correlated (frozen core and virtual orbitals excluded).
Therefore, conventional integrals are viable for computations with at most 100-200 orbitals.
For larger bases, density Fitting and Cholesky decomposition are instead recommended.

Density Fitting (DF) and Cholesky Decomposition (CD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The density fitting and Cholesky decomposition methods approximate
two-electron integrals as products of three-index tensors :math:`b_{pr}^{P}`

.. math:: \langle pq | rs \rangle = \sum_P^M b_{pr}^{P} b_{qs}^{P}

where :math:`M` is a quantity of the order :math:`3 N`.

.. caution:: The equations reported here use physicist notation for the two-electron integrals, but the DF/CD literature usually adopts chemist's notation.
The main difference between DF and CD is in the way the B tensors are defined.
In DF, the :math:`b` tensor is defined as

.. math:: b_{pq}^{Q} = \sum_p (pq | P)[(P | Q)^{-1/2}]_{PQ}

where the indices :math:`P` and :math:`Q` refer to the auxiliary basis set.
The auxiliary basis is defined via the Psi4 option ``DF_BASIS_MP2``.

In the CD approach, the :math:`b` tensor is formed via Cholesky decomposition of the
exact two-electron integrals in the atomic basis.
The accuracy of this decomposition (and the resulting two-electron integrals) is determined by a
user defined tolerance selected via the option ``CHOLESKY_TOLERANCE``.
Both the DF and CD algorithms store the :math:`b` tensor in memory, and therefore, they
require :math:`M N^2 \approx 3 N^3` memory for storage.
On a single node with 128 GB of memory, DF and CD computations allow to treat up to 1000 orbitals.


Disk-based Density Fitting (DiskDF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculations with more than 1000 basis functions quickly become unfeasible as the memory requirements
of density fitting grows as the cube of basis size.
In this case, it is possible to switch to a disk-based implementation of DF, which assumes that
the :math:`b` tensor can be fully stored on disk.


Integral Selection Keywords
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following keywords control the integral class and affect all computations that run in Forte:

**INT_TYPE**

``INT_TYPE`` selects the integral type used in the calculation

* Type: string

* Default: ``CONVENTIONAL``

* Possible Values:  ``CONVENTIONAL``, ``DF``, ``CHOLESKY``, ``DISKDF``

**CHOLESKY_TOLERANCE**

The tolerance for the cholesky decomposition.  This keyword determines the accuracy of the computation.
A smaller tolerance is a more accurate computation.
The tolerance for the cholesky decomposition:

* Type: double in scientific notation (ie 1e-5 or 0.0001)

* Default: ``1.0e-6``

**DF_BASIS_MP2**

The basis set used for DF.  This keyword needs to be placed in the globals section of a Psi4 input.
This basis should be one of the RI basis sets designed for a given primary basis, for example,
when using ``BASIS = cc-pVDZ`` you should use ``BASIS = cc-pVDZ-RI``.

* Type: string specifing basis set

* Default: none
