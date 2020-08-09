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
well defined value of total spin (:math:`{\hat{S}}^2`) and spin projection onto
the z axis (:math:`{\hat{S}}_z`).

The total spin is controlled by the option ``MULTIPLICITY``. This quantity is
related to the total spin quantum number ``S`` by the condition
``MULTIPLICITY = 2S + 1``.
If the input file does not specify the option ``MULTIPLICITY``, Forte will read
the multiplicity from the ``Wavefunction`` object passed by Psi4.

The projection of spin onto the z axis is controlled by the option ``MS``.
This option is of type ``double``, so it should be specified as ``0.0``, ``-1.5``, etc.
If the user does not specify the option ``MS``, Forte deduces a
value consistent with the option ``MULTIPLICITY``.
Modules will select either the lowest or highest value of ``MS`` compatible with
``MULTIPLICITY``, depending on internal details of the implementation.
For example, if ``MULTIPLICITY = 3`` and ``MS`` is not specified, the FCI code
in Forte will assume that the user is interested in the solution with
:math:`M_S = 0`.

For example, the following input requests the :math:`M_S = 0` component of a
triplet state::

    # triplet, m_s = 0
    set forte{
        multiplicity = 3
        ms = 0.0
    }

while the following gives the :math:`M_S = -1` component::

    # triplet, m_s = 1
    set forte{
        multiplicity = 3
        ms = -1.0
    }


Definition of orbital spaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running a Forte computation requires specifying a partitioning of the molecular
orbitals.
Forte defines five types of elementary orbital spaces:

1. Frozen doubly occupied orbitals (``FROZEN_DOCC``). These orbitals are always
doubly occupied.

2. Restricted doubly occupied orbitals (``RESTRICTED_DOCC``). Orbitals that are
treated as doubly occupied by method for static correlation.
Restricted doubly occupied orbitals are allowed to be excited in
in methods that add dynamic electron correlation.

3. Active/generalized active orbitals (``ACTIVE``/``GASn``).
Used to define active spaces or generalized active spaces for static correlation methods.
These orbitals are partially occupied.
Standard complete active spaces can be specified either via the
``ACTIVE`` or the ``GAS1`` orbital space.
For generalized active spaces, the user must provide the number of orbitals
in each irrep for all the GAS spaces reuired.
``GAS1`` through ``GAS6`` are currently supported.

4. Restricted unoccupied orbitals (``RESTRICTED_UOCC``). Also called virtuals,
these orbitals are ignored by methods for static correlation but considered by
dynamic correlation approaches.

5. Frozen unoccupied orbitals (``FROZEN_UOCC``). These orbitals are always
unoccupied.

The following table summarizes the properties of these orbital spaces:

+-----------------+------------+---------------+--------------------------------------+
| Space           | Occupation | Occupation    |  Description                         |
|                 | in CAS/GAS | in correlated |                                      |
|                 |            | methods       |                                      |
+=================+============+===============+======================================+
| FROZEN_DOCC     |     2      |     2         |  Frozen doubly occupied orbitals     |
+-----------------+------------+---------------+--------------------------------------+
| RESTRICTED_DOCC |     2      |    0-2        |  Restricted doubly occupied orbitals |
+-----------------+------------+---------------+--------------------------------------+
| GAS1, GAS2, ... |    0-2     |    0-2        |  Generalized active spaces           |
+-----------------+------------+---------------+--------------------------------------+
| RESTRICTED_UOCC |     0      |    0-2        |  Restricted unoccupied orbitals      |
+-----------------+------------+---------------+--------------------------------------+
| FROZEN_UOCC     |     0      |     0         |  Frozen unoccupied orbitals          |
+-----------------+------------+---------------+--------------------------------------+

.. Note::
  Forte makes a distinction between `elementary` and `composite` orbital spaces.
  The spaces defined above are all elementary, except for ``ACTIVE``, which is
  defined as the composite space of all the GAS spaces, that is,
  ``ACTIVE`` = ``GAS1 | GAS2 | GAS3 | GAS4 | GAS5 | GAS6``.
  When the user specifies the value of a composite space like ``ACTIVE``, then all the
  orbitals are by default assigned to the first space, which in the case of ``ACTIVE`` is ``GAS1``.
  It is important also to note that when there is more than one irrep, the orbitals withing a
  composite space are ordered **first** by irrep and then by elementary space.
  This important to keep in mind when plotting orbitals or for developers writing code in forte.
  

Orbital space specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Selecting the correct set of orbitals for a multireference computation is
perhaps one of the most important step in setting up an input file.

Forte takes advantage of symmetry, so for each orbital space the user must
provide the number of orbitals in each irrep. Forte cna handle only Abelian
groups, so each orbital space is a vector of integers with at most eight entries.
Irreps are arranged according to Cotton's book
(`Chemical Applications of Group Theory`).

The following is an example of a computation on BeH\ :sub:`2`. This system has 6
electrons. We freeze the Be 1s-like orbital, which has A\ :sub:`1` symmetry.
The 2a\ :sub:`1` orbital is restricted doubly occupied and the
3a\ :sub:`1`/1b\ :sub:`2` orbitals belong to the active space. The remaining
orbitals belong to the ``RESTRICTED_UOCC`` set and no virtual orbitals are
frozen::

    set forte{
        #                 A1 A2 B1 B2
        frozen_docc      [1 ,0 ,0 ,0]
        restricted_docc  [2 ,0 ,0 ,0]
        active           [1 ,0 ,0 ,1]
        restricted_uocc  [4 ,0 ,2 ,3]
        frozen_uocc      [0 ,0 ,0 ,0]
    }


Partial specification of orbital spaces and space priority
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifying all five orbital spaces for each computation is tedious and error prone.
Forte can help reduced the number of orbital spaces that the user needs to
specify by making certain assumptions.
The class that controls orbital spaces (``MOSpaceInfo``) assumes that orbital
spaces have the following priority::

    GAS1 (= ACTIVE) > RESTRICTED_UOCC > RESTRICTED_DOCC > FROZEN_DOCC > FROZEN_UOCC > GAS2 > ...

When the input does not contain all five orbital spaces, Forte will infer the
size of other orbital spaces. It first sums up all the orbitals specified by
the user, and then assigns any remaining orbital to the space not specified in
the input that has the highest priority.

In the case of the BeH\ :sub:`2` example, it is necessary to specify only the
``FROZEN_DOCC``, ``RESTRICTED_DOCC``, and ``ACTIVE`` orbital spaces::

    set forte{
        frozen_docc        [1 ,0 ,0 ,0]
        restricted_docc    [2 ,0 ,0 ,0]
        active             [1 ,0 ,0 ,1]

        # Forte will automatically assign the following:
        # restricted_uocc  [4 ,0 ,2 ,3]
        # frozen_uocc      [0 ,0 ,0 ,0]
        # gas2             [0 ,0 ,0 ,0]
        # gas3             [0 ,0 ,0 ,0]
        # gas4             [0 ,0 ,0 ,0]
        # gas5             [0 ,0 ,0 ,0]
        # gas6             [0 ,0 ,0 ,0]
}

the remaining 9 orbitals are automatically assigned to the ``RESTRICTED_UOCC``
space. This space, together with ``FROZEN_UOCC``, was not specified in the input.
However, ``RESTRICTED_UOCC`` has higher priority than the ``FROZEN_UOCC`` space,
so Forte will assign all the remaining orbitals to the ``RESTRICTED_UOCC`` set.

A Forte input with no orbital space specified will assign all orbitals to the
active space::

    set forte{
        # Forte will automatically assign the following:
        # frozen_docc      [0 ,0 ,0 ,0]
        # restricted_docc  [0 ,0 ,0 ,0]
        # active           [7 ,0 ,2 ,4]
        # restricted_uocc  [0 ,0 ,0 ,0]
        # frozen_uocc      [0 ,0 ,0 ,0]
    }

Note, that except for full CI computations with small basis sets, in all
other cases this computation might be unfeasible.

As a general rule, it is recommended that user run a SCF computations and
inspect the orbitals prior to selecting an active space.


