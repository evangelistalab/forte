.. _`sec:methods:avas`:

Atomic Valence Active Space (AVAS)
==================================

.. codeauthor:: Chenxi Cai and Chenyang Li
.. sectionauthor:: Chenyang Li and Nan He

Overview
^^^^^^^^

This AVAS procedure provides an automatic way to generate an active space for correlation
computations by projecting MOs to an AO subspace, computing and sorting the overlaps for
a new set of rotated MOs and a suitable active space.

Given a projector :math:`\hat{P}`, AVAS builds the projected overlap matrices for
doubly occupied and virtual orbitals separately from an restricted Hartree-Fock wave function

.. math::
    S_{ij} &= \langle i | \hat{P} | j \rangle = \sum_{\mu \nu} C_{\mu i} P_{\mu\nu} C_{\nu j},
   \quad i,j \in \{\text{DOCC}\}, \\
    \bar{S}_{ab} &= \langle a | \hat{P} | b \rangle = \sum_{\mu \nu} C_{\mu a} P_{\mu\nu} C_{\nu b},
   \quad a,b \in \{\text{UOCC}\},

where the projector matrix is given by

.. math::
    P_{\mu\nu} = \sum_{pq} \langle \mu | p \rangle (\rho^{-1})_{pq} \langle q | \nu \rangle,
    \quad p, q \in \{\text{Target Valence Atomic Orbitals}\}.

The matrix :math:`\rho^{-1}` is the inverse of target AO overlap matrix
:math:`\rho_{pq} = \langle p | q \rangle`.

.. note::
    Target AOs are selected from the MINAO basis.

If the option :code:`AVAS_DIAGONALIZE` is :code:`TRUE`, AVAS will diagonalize matrices
:math:`S_{ij}` and :math:`\bar{S}_{ab}` and rotate orbitals separately such that
the Hartree-Fock energy is unaffected:

.. math::
    \mathbf{S U} &= \mathbf{U \sigma_{\rm DOCC}}, \quad
    \tilde{C}_{\mu i} = \sum_{j} C_{\mu j} U_{ji}, \\
    \mathbf{\bar{S} \bar{U}} &= \mathbf{\bar{U} \sigma_{\rm UOCC}}, \quad
    \tilde{C}_{\mu a} = \sum_{b} C_{\mu b} \bar{U}_{ba}.

The two sets of eigenvalues are combined
:math:`\mathbf{\sigma = \sigma_{\rm DOCC} \oplus \sigma_{\rm UOCC}}`
and subsequently sorted in descending order.
If :code:`AVAS_DIAGONALIZE` is set to :code:`FALSE`,
the "eigenvalues" will be directly grabbed from the diagonal elements of the projected overlap matrices
and no orbital rotation is performed.

Depending on the selection scheme, part of the orbitals with nonzero eigenvalues
are selected as active orbitals.
We then semi-canonicalize all four subsets of orbitals separately.
The final orbitals are arranged such that those considered as active lie in between
the inactive occupied and inactive virtual orbitals.

.. warning::
    The code does not support UHF reference at present.
    For ROHF reference, our implementation does not touch any singly occupied orbitals,
    which are all considered as active orbitals and assumed in canonical form.

Input Example
^^^^^^^^^^^^^

In this example, we perform the AVAS procedure on formaldehyde
followed by a CASCI computation.::

    import forte
    molecule H2CO{
    0 1
    C           -0.000000000000    -0.000000000006    -0.599542970149
    O           -0.000000000000     0.000000000001     0.599382404096
    H           -0.000000000000    -0.938817812172    -1.186989139808
    H            0.000000000000     0.938817812225    -1.186989139839
    noreorient
    }

    set {
      basis         cc-pvdz
      reference     rhf
      scf_type      pk
      e_convergence 12
    }

    set forte {
      job_type            none                 # no energy computation
      subspace            ["C(2px)","O(2px)"]  # target AOs from 2px orbitals of C and O
      avas                true                 # turn on AVAS
      avas_diagonalize    true                 # diagonalize the projected overlaps
      avas_sigma          1.0                  # fraction of eigenvalues included as active
    }
    Escf, wfn = energy('forte', return_wfn=True)

    set forte {
      job_type            newdriver  # compute some forte energy
      active_space_solver fci        # use FCI solver
      print               1          # print level
      restricted_docc     [5,0,0,2]  # from AVAS
      active              [0,0,3,0]  # from AVAS
    }
    Ecasci = energy('forte', ref_wfn=wfn)

.. note::
    The keyword :code:`noreorient` in the :code:`molecule` section is very important
    if certain orientations of orbitals are selected in the subspace (e.g., 2pz of C).
    Otherwise, the subspace orbital selection may end up the wrong direction.

The AVAS procedure outputs::

    Sum of eigenvalues: 1.98526975

    ==> AVAS MOs Information <==

      ---------------------------------------
                         A1    A2    B1    B2
      ---------------------------------------
      DOCC INACTIVE       5     0     0     2
      DOCC ACTIVE         0     0     1     0
      SOCC ACTIVE         0     0     0     0
      UOCC ACTIVE         0     0     2     0
      UOCC INACTIVE      13     3     4     8
      ---------------------------------------
      RESTRICTED_DOCC     5     0     0     2
      ACTIVE              0     0     3     0
      RESTRICTED_UOCC    13     3     4     8
      ---------------------------------------

    ==> Atomic Valence MOs (Active Marked by *) <==

      ===============================
       Irrep    MO  Occ.  <phi|P|phi>
      -------------------------------
      *  B1      0    2      0.970513
      *  B1      1    0      0.992548
      *  B1      2    0      0.022209
      ===============================


The :code:`Sum of eigenvalues` is the sum of traces of projected overlap matrices
:math:`\mathbf{S}` and :math:`\mathbf{\bar{S}}`.
We see that AVAS generates three active orbitals of B1 symmetry.
We then use this guess of active orbitals to compute the CASCI energy:::

    ==> Root No. 0 <==

      200     -0.98014601
      020      0.18910986

      Total Energy:      -113.911667467206598

    ==> Energy Summary <==

      Multi.(2ms)  Irrep.  No.               Energy
      ---------------------------------------------
         1  (  0)    A1     0     -113.911667467207
      ---------------------------------------------

.. note::
    Currently, the procedure is not automated enough so that
    two Forte computations need to be carried out.
    First perform an AVAS and check the output guess of active orbitals.
    Then put :code:`RESTRICTED_DOCC` and :code:`ACTIVE` in the input for
    another round of Forte computation.

For more examples, see :code:`avas-1` to :code:`avas-6` in the :code:`tests/methods` folder.
In particular, :code:`avas-6` is a practical example on ferrocene.

Options
^^^^^^^

**AVAS**

Turn on the AVAS procedure or not.

* Type: Boolean
* Default: False

**AVAS_DIAGONALIZE**

Diagonalize the projected overlap matrices or not.

* Type: Boolean
* Default: True

**AVAS_EVALS_THRESHOLD**

Threshold smaller than which is considered as zero for
an eigenvalue of the projected overlap matrices.

* Type: double
* Default: 1.0e-6

**AVAS_SIGMA**

Cumulative threshold to the eigenvalues of the projected overlap matrices
to control the output number of active orbitals.
Orbitals will be added to the active subset starting from that of the largest
:math:`\sigma` value and stopped when
:math:`\sum_{u}^{\rm ACTIVE} \sigma_{u} / \sum_{p}^{\rm ALL} \sigma_{p}`
is larger than the threshold.

* Type: double
* Default: 0.98

**AVAS_CUTOFF**

The threshold greater than which to the eigenvalues of the projected overlap
matrices will be considered as active orbitals. If not equal to 1.0, it takes
priority over the sigma threshold selection.

* Type: double
* Default: 1.0

**AVAS_NUM_ACTIVE**

The total number of orbitals considered as active for
doubly occupied and virtual orbitals (singly occupied orbitals not included).
If not equal to 0, it will take priority over the sigma or cutoff selections.

* Type: int
* Default: 0

**AVAS_NUM_ACTIVE_OCC**

The number of doubly occupied orbitals considered as active.
If not equal to 0, it will take priority over the selection schemes based on
sigma and cutoff selections and the total number of active orbitals.

* Type: int
* Default: 0

**AVAS_NUM_ACTIVE_VIR**

The number of virtual orbitals considered as active.
If not equal to 0, it will take priority over the selection schemes based on
sigma and cutoff selections and the total number of active orbitals.

* Type: int
* Default: 0

Citation Reference
^^^^^^^^^^^^^^^^^^

Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals |br|
`J. Chem. Theory Comput. 13, 4063-4078 (2017) <https://pubs.acs.org/doi/10.1021/acs.jctc.7b00128>`_.

.. |br| raw:: html

   <br />
