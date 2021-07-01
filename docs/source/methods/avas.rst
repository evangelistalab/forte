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
followed by a CASCI computation. ::

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
We then use this guess of active orbitals to compute the CASCI energy: ::

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

Defining the molecular plane for π orbitals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the above example, the molecule is placed on the yz plane such that the directions of p orbitals
are aligned with the molecule xyz frame.
However, sometimes it is hard to make these two xyz frames aligned.
For example, the molecule contains multiple π subsystems or the plane is not perfect.
In such cases, if only pi orbitals are desired, we can linearly combine the px, py, and pz orbitals
to a single 'pz' orbital that perpendicular to the π subsystem plane.

We use the keyword :code:`SUBSPACE_PI_PLANES` to define all the π subsystems planes,
where each plane is given by a list of atoms' expressions.
Some valid expressions would be: ::

  - [['C', 'H', 'O']]  # only one plane consisting all C, H, and O atoms of the molecule.
  - [['C1-6'], ['N1-2', 'C9-11']]  # plane 1 with the first six C atoms of the molecule,
                                   # plane 2 with C9, C10, C11, N1 and N2 atoms.
  - [['C1-4'], ['C1-2', 'C5-6']]:  # plane 1 with the first four C atoms of the molecule,
                                   # plane 2 with C1, C2, C5, and C6 atoms.
                                   # Two planes share C1 and C2!

Take the formaldehyde example above.
We now reorient it such that the plane normal points at (1.0, 1.0, 1.0).
The same AVAS orbitals can be obtained using the following input: ::

  molecule H2CO{
  0 1
  C        0.346146295209737    0.126698337466632   -0.472844632676369
  O       -0.346053592352928   -0.126664405871036    0.472717998223964
  H        1.227335215970831   -0.489581944167415   -0.737753271803415
  H        0.143281782803090    0.991262584630455   -1.134544367433545
  noreorient
  }

  set forte {
    subspace           ["C(2p)", "O(2p)"]  # must include all p orbitals!
    subspace_pi_planes [["C", "O", "H"]]   # only one plane, defined by all C, O and H atoms
    avas               true
    avas_diagonalize   true
    avas_sigma         1.0
  }

If the :code:`SUBSPACE_PI_PLANES` option is not empty, the code will prune the subspace p orbitals
and linearly combine them to one p orbital per atom in the direction perpendicular to the plane.
In the above example, we will end up with only two p orbitals perpendicular to the molecular plane.
If :code:`SUBSPACE_PI_PLANES` is empty, we would have had 6 p orbitals in the subspace.

.. note::
    It is very important to include all p orbitals in :code:`SUBSPACE`.
    Otherwise, the code will follow the directions given by :code:`SUBSPACE`.

.. tip::
    The code is flexible enough to treat double active spaces (e.g., double-π or double-d-shell).
    For example, the double-π active space of formaldehyde can be obtained via ::

      set forte {
        minao_basis        double-shell
        subspace           ["C(2p)","C(3p)","O(2p)","O(3p)"]
        subspace_pi_planes [["C","O","H"]]
        avas               true
        avas_diagonalize   true
        avas_cutoff        0.5
      }

    You need to prepare your own MINAO basis (here I prepare a basis called "double-shell.gbs").
    This MINAO basis should include at least the 2p and 3p orbitals of C and O,
    which can be grabbed from the cc-pVTZ or ANO-RCC-VTZP basis sets.

For a more realistic example, consider the following iron porphyrin related molecule:

.. image:: images/FeP.png
    :width: 600
    :align: center
    :alt: An iron porphyrin complex.

This molecule contains two π systems, namely, porphyrin and imidazole.
Also, the porphyrin is not a perfect plane anymore.
The following input snippet selects all 3d orbitals of Fe, 3p orbitals of S,
and all 'pz' orbitals of porphyrin and imidazole rings. ::

  set forte {
    avas                true
    avas_diagonalize    true
    avas_cutoff         0.5
    minao_basis         cc-pvtz-minao
    subspace            ["Fe(3d)","C6-25(2p)","N(2p)","S(3p)","C1-3(2p)"]
    subspace_pi_planes  [["Fe","C6-25","N3-6"], ["N1-2","C1-3"]]
  }

The AVAS output selects exactly 37 orbitals we wanted. ::

  Sum of eigenvalues: 36.98863914
  AVAS covers 96.59% of the subspace.

  ==> AVAS MOs Information <==

    ---------------------
                        A
    ---------------------
    DOCC INACTIVE     106
    DOCC ACTIVE        22
    SOCC ACTIVE         0
    UOCC ACTIVE        15
    UOCC INACTIVE     462
    ---------------------
    RESTRICTED_DOCC   106
    ACTIVE             37
    RESTRICTED_UOCC   462
    ---------------------

  ==> Atomic Valence MOs (Active Marked by *) <==

    ===============================
     Irrep    MO  Occ.  <phi|P|phi>
    -------------------------------
    *   A      0    2      0.999085
    *   A      1    2      0.998642
    *   A      2    2      0.998359
    *   A      3    2      0.996035
    *   A      4    2      0.994644
    *   A      5    2      0.994278
    *   A      6    2      0.993868
    *   A      7    2      0.993659
    *   A      8    2      0.993108
    *   A      9    2      0.992442
    *   A     10    2      0.991897
    *   A     11    2      0.991522
    *   A     12    2      0.991168
    *   A     13    2      0.990619
    *   A     14    2      0.989037
    *   A     15    2      0.988792
    *   A     16    2      0.987373
    *   A     17    2      0.986867
    *   A     18    2      0.984205
    *   A     19    2      0.974919
    *   A     20    2      0.855068
    *   A     21    2      0.747171
        A     22    2      0.215276
        A     23    2      0.175599
        A     24    2      0.056342
        A     25    2      0.047345
        A     26    2      0.034783
        A     27    2      0.030997
        A     28    2      0.028569
        A     29    2      0.026469
        A     30    2      0.023365
        A     31    2      0.017892
        A     32    2      0.016921
        A     33    2      0.014212
        A     34    2      0.010871
        A     35    2      0.001703
        A     36    2      0.000408
    *   A    128    0      0.999163
    *   A    129    0      0.997849
    *   A    130    0      0.988687
    *   A    131    0      0.985388
    *   A    132    0      0.982652
    *   A    133    0      0.981676
    *   A    134    0      0.976224
    *   A    135    0      0.973079
    *   A    136    0      0.971042
    *   A    137    0      0.968590
    *   A    138    0      0.964765
    *   A    139    0      0.952259
    *   A    140    0      0.943277
    *   A    141    0      0.824388
    *   A    142    0      0.784721
        A    143    0      0.252635
        A    144    0      0.144740
        A    145    0      0.024759
        A    146    0      0.015490
        A    147    0      0.012840
        A    148    0      0.012333
        A    149    0      0.010860
        A    150    0      0.010643
        A    151    0      0.009008
        A    152    0      0.008557
        A    153    0      0.008096
        A    154    0      0.007851
        A    155    0      0.007346
        A    156    0      0.006517
        A    157    0      0.005974
        A    158    0      0.005795
        A    159    0      0.005433
        A    160    0      0.004976
        A    161    0      0.003642
        A    162    0      0.001632
        A    163    0      0.001346
        A    164    0      0.000898
    ===============================

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
