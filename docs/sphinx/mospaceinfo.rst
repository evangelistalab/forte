.. _`sec:mospaceinfo`:

Specifying a wave function in Forte
===================================

.. sectionauthor:: Francesco A. Evangelista

Number of electrons and spin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, Forte will determine the number of electrons from the atomic charges
and total molecular charge. The total molecular charge is assumed to be 0 unless
specified by the user in the input file::

    molecule {
      0 1 # <-- charge and multiplicity (see below)
      ...
    }

For certain computations, Forte allows the user to compute a solution with a
well defined value of total spin (:math:`\hat{S}^2`) and spin projection onto
the z axis (:math:`\hat{S}_z`).

The total spin is controlled by the option ``MULTIPLICITY``. This quantity is
related to the total spin quantum number ``S`` by the condition
``MULTIPLICITY = 2S + 1``.
If the input file does not specify the option ``MULTIPLICITY``, Forte will read
the multiplicity from the ``Wavefunction`` object passed by Psi4.

The projection of spin onto the z axis is controlled by the option ``MS``.
This **is an integer variable that is twice** the value of the spin z-projection
quantum number, ``MS`` = :math:`2 M_S`.
If the user does not specify the option ``MS``, Forte determines the **lowest**
value consistent with the value of ``MULTIPLICITY``.
For example, if ``MULTIPLICITY = 3`` and ``MS`` is not specified, Forte will
assume that the user is interested in the solution with :math:`M_S = 0`.

For example, the following input requests the :math:`M_S = 0` component of a
triplet state::

    # triplet, m_s = 0
    set forte{
        multiplicity = 3
        ms = 0
    }

while the following gives the :math:`M_S = 1` component::

    # triplet, m_s = 1
    set forte{
        multiplicity = 3
        ms = 2
    }

Definition of orbital spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a Forte computation requires specifying a partitioning of the molecular
orbitals.
Forte defines five orbital spaces:

1. Frozen doubly occupied orbitals (``FROZEN_DOCC``). These orbitals are always
doubly occupied.

2. Restricted doubly occupied orbitals (``RESTRICTED_DOCC``). Orbitals that are
treated as doubly occupied by method for static correlation.
Restricted doubly occupied orbitals are allowed to be excited in
in methods that add dynamic electron correlation.

3. Active orbitals (``ACTIVE``). Used to define active spaces for static
correlation methods. These orbitals are partially occupied.

4. Restricted unoccupied orbitals (``RESTRICTED_UOCC``). Also called virtuals,
these orbitals are ignored by methods for static correlation but considered by
dynamic correlation approaches.

5. Frozen unoccupied orbitals (``FROZEN_UOCC``). These orbitals are always
unoccupied.
