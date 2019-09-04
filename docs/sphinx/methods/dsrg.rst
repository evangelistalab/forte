.. _`sec:methods:ldsrg`:

Driven Similarity Renormalization Group
=======================================

.. codeauthor:: Francesco A. Evangelista, Chenyang Li, Kevin Hannon, Tianyuan Zhang
.. sectionauthor:: Chenyang Li, Tianyuan Zhang, Kevin P. Hannon

.. important::
  Any publication utilizing the DSRG code should acknowledge the following articles:

  * F. A. Evangelista, *J. Chem. Phys.* **141**, 054109 (2014).

  * C. Li and F. A. Evangelista, *Annu. Rev. Phys. Chem.* **70**, 245-273 (2019).

  Depending on the features used, the user is encouraged to cite the corresponding articles listed :ref:`here <dsrg_ref>`.

.. caution::
  The current implementation does not employ spin-adapted equations and it does not work for even multiplicities.
  For odd multiplicities, we assume low-spin configurations (by default, no need to set up).
  For those who are desperate to perform computations on doublet, an alternative way is to add a hydrogen atom at long distance away from the system and perform a singlet computation.
  Spin adaptation is on the TODO list.

Basic DSRG
^^^^^^^^^^

1. Overview of DSRG Theory
++++++++++++++++++++++++++

Driven similarity renormalization group (DSRG) is a numerically robust approach to consider
dynamical (or weak) electron correlation effects. Specifically, the DSRG performs a *continuous*
similarity transformation to the bare Born-Oppenheimer Hamiltonian :math:`\hat{H}`,

.. math:: \bar{H}(s) = e^{-\hat{S}(s)} \hat{H} e^{\hat{S}(s)},

where :math:`s` is the flow parameter defined in the range :math:`[0, +\infty)` that controls the
transformation (vaguely speaking).
The operator :math:`\hat{S}` can be any operator in general.
For example, if :math:`\hat{S} = \hat{T}` is the cluster substitution operator, the DSRG :math:`\bar{H}(s)`
is identical to coupled-cluster (CC) similarity transformed Hamiltonian except for the :math:`s`
dependence. See :ref:`Table I <table:dsrg_cc_connect>` for different flavours of :math:`\hat{S}`.

.. _`table:dsrg_cc_connect`:

.. table:: Table I. Connections of DSRG to CC theories when using different types of :math:`\hat{S}`.

    +-----------------+-----------------------------------------------+----------------+
    | :math:`\hat{S}` |                Explanation                    |   CC Theories  |
    +=================+===============================================+================+
    | :math:`\hat{T}` |             cluster operator                  | traditional CC |
    +-----------------+-----------------------------------------------+----------------+
    | :math:`\hat{A}` | :math:`\hat{A} = \hat{T} - \hat{T}^{\dagger}` | unitary CC, CT |
    +-----------------+-----------------------------------------------+----------------+
    | :math:`\hat{G}` |             general operator                  | generalized CC |
    +-----------------+-----------------------------------------------+----------------+

In the current implementation, we choose the **anti-hermitian** parametrization, i.e., :math:`\hat{S} = \hat{A}`.

The DSRG transformed Hamiltonian contains many-body (> 2-body) interactions in general.
We can express it as

.. math:: \bar{H} = \bar{h}_0 + \bar{h}^{p}_{q} \{ a^{q}_{p} \} + \frac{1}{4} \bar{h}^{pq}_{rs} \{ a^{rs}_{pq} \} + \frac{1}{36} \bar{h}^{pqr}_{stu} \{ a^{stu}_{pqr} \} + ...

where :math:`a^{pq...}_{rs...} = a_{p}^{\dagger} a_{q}^{\dagger} \dots a_s a_r` is a string of creation and annihilation operators
and :math:`\{\cdot\}` represents normal-ordered operators. In particular, we use Mukherjee-Kutzelnigg normal ordering
[see J. Chem. Phys. 107, 432 (1997)]. Here we also assume summations over repeated indices for brevity.
Also note that :math:`\bar{h}_0` is the energy dressed by dynamical correlation effects.

In DSRG, we require the off-diagonal components of :math:`\bar{H}` gradually go to zero (from :math:`\hat{H}`) as :math:`s` grows (from 0).
By off-diagonal components, we mean :math:`\bar{h}^{ij\dots}_{ab\dots}` and :math:`\bar{h}^{ab\dots}_{ij\dots}` where :math:`i,j,\dots`
indicates hole orbitals and :math:`a,b,\dots` labels particle orbitals.
There are in principle infinite numbers of ways to achieve this requirement.
The current implementation chooses the following parametrization,

.. math:: \bar{h}^{ij\dots}_{ab\dots} = [\bar{h}^{ij\dots}_{ab\dots} + \Delta^{ij\dots}_{ab\dots} t^{ij\dots}_{ab\dots}] e^{-s(\Delta^{ij\dots}_{ab\dots})^2},

where :math:`\Delta^{ij\dots}_{ab\dots} = \epsilon_{i} + \epsilon_{j} + \dots - \epsilon_{a} - \epsilon_{b} - \dots` is
the Møller-Plesset denominator defined by orbital energies :math:`\epsilon_{p}` and :math:`t^{ij\dots}_{ab\dots}` are the cluster amplitudes.
This equation is called the DSRG flow equation, which suggests a way how the off-diagonal Hamiltonian components evolves as :math:`s` changes.
We can now solve for the cluster amplitudes since :math:`\bar{H}` is a function of :math:`\hat{T}` using the Baker–Campbell–Hausdorff (BCH) formula.

Since we choose :math:`\hat{S} = \hat{A}`, the corresponding BCH expansion is thus non-terminating.
Approximations have to be introduced and different treatments to :math:`\bar{H}` leads to various levels of DSRG theories.
Generally, we can treat it either in a perturbative or non-perturbative manner.
For non-perturbative theories, the **only** widely tested scheme so far is the recursive single commutator (RSC) approach,
where every single commutator is truncated to contain at most two-body contributions for a nested commutator.
For example, a doubly nested commutator is computed as

.. math:: \frac{1}{2} [[\hat{H}, \hat{A}], \hat{A}] \approx \frac{1}{2} [[\hat{H}, \hat{A}]_{1,2}, \hat{A}]_{0,1,2},

where 0, 1, 2 indicate scalar, 1-body, and 2-body contributions.
We term the DSRG method that uses RSC as LDSRG(2).

Alternatively, we can perform a perturbative analysis on the **approximated** BCH equation of :math:`\bar{H}` and obtain
various DSRG perturbation theories [e.g., 2nd-order (PT2) or 3rd-order (PT3)].
Note we use the RSC approximated BCH equation for computational cost considerations.
As such, the implemented DSRG-PT3 is **not** a complete PT3 but a companion PT3 of the LDSRG(2) method.

To conclude this subsection, we discuss the computational cost and current implementation limit,
which are summarized in :ref:`Table II <table:dsrg_cost>`.

.. _`table:dsrg_cost`:

.. table:: Table II. Cost of the various implemented DSRG methods.

    +----------+-----------------------+----------------------------------+-----------------------------------+
    |  Method  |  Computational Cost   |  System Size (full 2e-ints)      |      System Size (DF/CD)          |
    +==========+=======================+==================================+===================================+
    |    PT2   | one-shot :math:`N^5`  | :math:`\sim 250` basis functions | :math:`\sim 1800` basis functions |
    +----------+-----------------------+----------------------------------+-----------------------------------+
    |    PT3   | one-shot :math:`N^6`  | :math:`\sim 250` basis functions | :math:`\sim 700` basis functions  |
    +----------+-----------------------+----------------------------------+-----------------------------------+
    | LDSRG(2) | iterative :math:`N^6` | :math:`\sim 200` basis functions | :math:`\sim 550` basis functions  |
    +----------+-----------------------+----------------------------------+-----------------------------------+


2. Input Examples
+++++++++++++++++

**Minimal Example**

Let us first see an example with minimal keywords.
In particular, we compute hydrogen fluoride using DSRG multireference (MR) PT2
with complete active space self-consistent field (CASSCF) reference.

::

    import forte

    molecule mol{
      0 1
      F
      H  1 R
    }
    mol.R = 1.50  # this is a neat way to specify H-F bond lengths

    set globals{
       basis                   cc-pvdz
       reference               rhf
       scf_type                pk
       d_convergence           8
       e_convergence           10
       restricted_docc         [2,0,1,1]
       active                  [2,0,0,0]
    }

    set forte{
       active_space_solver     fci
       correlation_solver      dsrg-mrpt2
       dsrg_s                  0.5
       frozen_docc             [1,0,0,0]
       restricted_docc         [1,0,1,1]
       active                  [2,0,0,0]
    }

    Emcscf, wfn = energy('casscf', return_wfn=True)
    energy('forte', ref_wfn=wfn)

There are three blocks in the input:

1. The :code:`molecule` block specifies the geometry, charge, multiplicity, etc. (see Psi4 manual for details).

2. The second block specifies Psi4 global options.

3. The last block shows options specifically for Forte.

In this example, we use Psi4 to compute CASSCF reference.
Psi4 provides the freedom to specify the core (a.k.a. internal) and active orbitals
using :code:`RESTRICTED_DOCC` and :code:`ACTIVE` options,
but *it is generally the user's responsibility for a correct orbital ordering*.
The :code:`RESTRICTED_DOCC` array :code:`[2,0,1,1]` indicates two :math:`a_1`,
zero :math:`a_2`, one :math:`b_1`, and one :math:`b_2` orbitals, because the computation is
performed in :math:`C_{2v}` point group.
The actual CASSCF computation is invoked by
:code:`Emcscf, wfn = energy('casscf', return_wfn=True)`, where we also ask for
wave function besides energy.
The wave function :code:`wfn` will be read by Forte via argument :code:`ref_wfn`.

Forte generally recomputes the reference using the provided wave function parameters.
To perform a DSRG computation, the user is expected to specify the following keywords:

* :code:`ACTIVE_SPACE_SOLVER`:
  Here we use :code:`FCI` to perform a CAS configuration interaction (CASCI)
  within the active orbitals.

* :code:`CORRELATION_SOLVER`:
  This option determines which code to run. The four well-tested DSRG solvers are:
  :code:`DSRG-MRPT2`, :code:`THREE-DSRG-MRPT2`, :code:`DSRG-MRPT3`, and :code:`MRDSRG`.
  The density-fitted DSRG-MRPT2 is implemented in :code:`THREE-DSRG-MRPT2`.
  The :code:`MRDSRG` is mainly designed to perform MR-LDSRG(2) computations.

* :code:`DSRG_S`:
  This keyword specify the DSRG flow parameter in a.u.
  For general MR-DSRG computations, the user should change the value to :math:`0.5 \sim 2` a.u.

  .. caution::
    By default, :code:`DSRG_S` is set to :math:`10^8` a.u.
    The user should always set this keyword by hand!

* Orbital spaces:
  Here we also specify frozen core orbitals besides core and active orbitals.
  Note that in this example, we optimize the 1s-like core orbital in CASSCF but
  later freeze for DSRG treatments for dynamical correlation.
  Details regarding to orbital spaces can be found :ref:`sec:mospaceinfo`.

  .. tip::
    To perform a single-reference (SR) DSRG computation, the user only needs to set
    :code:`ACTIVE` to zero. In the above example, the SR DSRG-PT2 energy can be obtained
    by modifying :code:`RESTRICTED_DOCC` to :code:`[2,0,1,1]`
    and :code:`ACTIVE` to :code:`[0,0,0,0]`. The MP2 energy can be reproduced
    if we further change :code:`DSRG_S` to :math:`10^8` a.u.


**A More Advanced Example**



**Other Examples**

There are plenty of examples in the tests/method folder.
A complete list of the DSRG test cases can be found here (TODO).

3. General DSRG Options
+++++++++++++++++++++++

**CORR_LEVEL**

Correlation level of MR-DSRG.

* Type: string
* Options: PT2, PT3, LDSRG2, LDSRG2_QC, LSRG2, SRG_PT2, QDSRG2
* Default: PT2

**DSRG_S**

The value of the flow parameter :math:`s`.

* Type: double
* Default: 1.0e10

**DSRG_MAXITER**

Max iterations for MR-DSRG amplitudes update.

* Type: int
* Default: 50


Theoretical Variants and Technical Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Reference Relaxation
+++++++++++++++++++++++

2. Orbital Rotations
++++++++++++++++++++

3. Sequential Transformation
++++++++++++++++++++++++++++

4. Non-Interacting Virtual Orbital Approximation
++++++++++++++++++++++++++++++++++++++++++++++++

5. Related Options
++++++++++++++++++

**RELAX_REF**

Relax the reference for MR-DSRG.

* Type: string
* Options: NONE, ONCE, TWICE, ITERATE
* Default: NONE

**DSRG_HBAR_SEQ**

Apply the sequential transformation algorithm in evaluating the transformed Hamiltonian :math:`\bar{H}(s)`, i.e.,

.. math:: \bar{H}(s) = e^{-\hat{A}_n(s)} \cdots e^{-\hat{A}_2(s)} e^{-\hat{A}_1(s)} \hat{H} e^{\hat{A}_1(s)} e^{\hat{A}_2(s)} \cdots e^{\hat{A}_n(s)}.

* Type: bool
* Default: false

**DSRG_NIVO**

Apply non-interacting virtual orbital (NIVO) approximation in evaluating the transformed Hamiltonian.

* Type: bool
* Default: false


Integral Factorization Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Theory
+++++++++

Integral factorization, as it suggests, factorizes the two-electron integrals into contractions of low-rank tensors.
In particular, we use density fitting (DF) or Cholesky decomposition (CD) technique to express two-electron integrals as

.. math:: \langle ij || ab \rangle = \sum_{Q}^{N_\text{aux}} ( B_{ia}^{Q} B_{jb}^{Q} - B_{ib}^{Q} B_{ja}^{Q} )

where :math:`Q` runs over auxiliary indices.
Note that we use physicists' notation here but the DF/CD literature use chemist notation.

The main difference between DF and CD is how the :math:`B` tensor is formed.
In DF, the :math:`B` tensor is defined as

.. math:: B_{pq}^{Q} = \sum_P^{N_\text{aux}} (pq | P)[(P | Q)^{-1/2}]_{PQ}.

In the CD approach, the :math:`B` tensor is formed by performing a pivoted incomplete Cholesky decomposition of the 2-electron integrals.
The accuracy of this decomposition is determined by a user defined tolerance, which directly determines the accuracy of the 2-electron integrals.

2. Limitations
++++++++++++++

There are several limitations of the current implementation.

We store the entire three-index integrals in memory by default.
Consequently, we can treat about 1000 basis functions.
For larger systems, please use the :code:`DiskDF` keyword where these integrals are loaded to memory only when necessary.
In general, we can treat about 2000 basis functions (with DiskDF) using DSRG-MRPT2.

Density fitting is more suited to spin-adapted equations while the current code uses spin-integrated equations.

We have a more optimized code of DF-DSRG-MRPT2.
The batching algorithms of DSRG-MRPT3 (manually tuned) and MR-LDSRG(2) (Ambit) are currently not ideal.

3. Examples
+++++++++++

.. tip::
  For DSRG-MRPT3 and MR-LDSRG(2), DF/CD will automatically turn on if
  :code:`INT_TYPE` is set to :code:`DF`, :code:`CD`, or :code:`DISKDF`.
  For DSRG-MRPT2 computations, please set the :code:`CORRELATION_SOLVER` keyword to
  :code:`THREE-DSRG-MRPT2` besides the :code:`INT_TYPE` option.

The following input performs a DF-DSRG-MRPT2 calculation on nitrogen molecule.
This example is modified from the df-dsrg-mrpt2-4 test case.

.. warning:: In test case df-dsrg-mrpt2-4, :code:`SCF_TYPE` is specified to :code:`PK`, which is incorrect for a real computation.

::

    import forte

    memory 500 mb

    molecule N2{
      0 1
      N
      N  1 R
      R = 1.1
    }

    set globals{
       reference               rhf
       basis                   cc-pvdz
       scf_type                df
       df_basis_mp2            cc-pvdz-ri
       df_basis_scf            cc-pvdz-jkfit
       d_convergence           8
       e_convergence           10
    }

    set forte {
       active_space_solver     cas
       int_type                df
       restricted_docc         [2,0,0,0,0,2,0,0]
       active                  [1,0,1,1,0,1,1,1]
       correlation_solver      three-dsrg-mrpt2
       dsrg_s                  1.0
    }

    Escf, wfn = energy('scf', return_wfn=True)
    energy('forte', ref_wfn=wfn)

To perform a DF computation, we need to specify the following options:


1. Psi4 options:
   :code:`SCF_TYPE`, :code:`DF_BASIS_SCF`, :code:`DF_BASIS_MP2`
2. Forte options:
   :code:`CORRELATION_SOLVER`, :code:`INT_TYPE`

.. attention::
  Here we use different basis sets for :code:`DF_BASIS_SCF` and :code:`DF_BASIS_MP2`.
  There is no consensus on what basis sets should be used for MR computations.
  However, there is one caveat of using inconsistent DF basis sets in Forte due to orbital canonicalization:
  Frozen orbitals are left unchanged (i.e., canonical for :code:`DF_BASIS_SCF`)
  while DSRG (and orbital canonicalization) only reads :code:`DF_BASIS_MP2`.
  This inconsistency leads to slight deviations to the frozen-core energies (:math:`< 10^{-4}` Hartree)
  comparing to using identical DF basis sets.

The output produced by this input: ::

    ==> DSRG-MRPT2 (DF/CD) Energy Summary <==

      E0 (reference)                 =   -109.023295547673101
      <[F, T1]>                      =     -0.000031933175984
      <[F, T2]>                      =     -0.000143067308999
      <[V, T1]>                      =     -0.000183596694872
      <[V, T2]> C_4 (C_2)^2 HH       =      0.003655752832132
      <[V, T2]> C_4 (C_2)^2 PP       =      0.015967613107776
      <[V, T2]> C_4 (C_2)^2 PH       =      0.017515091046864
      <[V, T2]> C_6 C_2              =     -0.000194156963250
      <[V, T2]> (C_2)^4              =     -0.265179563137787
      <[V, T2]>                      =     -0.228235263114265
      DSRG-MRPT2 correlation energy  =     -0.228593860294120
      DSRG-MRPT2 total energy        =   -109.251889407967226
      max(T1)                        =      0.002234583100143
      ||T1||                         =      0.007061738508652

To use Cholesky integrals, set :code:`INT_TYPE` to :code:`CHOLESKY` and specify :code:`CHOLESKY_TOLERANCE`.
For example, a CD equivalence of the above example is ::

    # same molecule input ...

    set globals{
       reference               rhf
       basis                   cc-pvdz
       scf_type                cd
       cholesky_tolerance      5
       d_convergence           8
       e_convergence           10
    }

    set forte {
       active_space_solver     cas
       int_type                cholesky
       cholesky_tolerance      1.0e-5
       restricted_docc         [2,0,0,0,0,2,0,0]
       active                  [1,0,1,1,0,1,1,1]
       correlation_solver      three-dsrg-mrpt2
       dsrg_s                  1.0
    }

    Escf, wfn = energy('scf', return_wfn=True)
    energy('forte', ref_wfn=wfn)

The output energies are: ::

    E0 (reference)                 =   -109.021897967354022
    DSRG-MRPT2 total energy        =   -109.250407455691658

The energies computed using conventional integrals are: ::

    E0 (reference)                 =   -109.021904986168678
    DSRG-MRPT2 total energy        =   -109.250416722481461

The energy error of using CD integrals (threshold = :math:`10^{-5}` a.u.) is thus around :math:`\sim 10^{-5}` Hartree.
In general, comparing to conventional 4-index 2-electron integrals, the use of CD integrals yields
energy errors to the same decimal points as :code:`CHOLESKY_TOLERANCE`.

.. caution:: The cholesky algorithm, as currently written, does not allow applications to large systems (> 1000 basis functions).

4. Options
++++++++++

General Keywords
''''''''''''''''

For basic options of factorized integrals, please check :ref:`sec:integrals`.

Advanced Keywords
'''''''''''''''''

**CCVV_BATCH_NUMBER**

Manually specify the number of batches for computing THREE-DSRG-MRPT2 energies.
By default, the number of batches are automatically computed using the remaining memory estimate.

* Type: integer
* Default: -1

State-Averaged and Multi-State Approaches for Excited States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


TODOs
^^^^^

1. Spin Adaptation
++++++++++++++++++

This is done for unrelaxed DSRG-MRPT2 but not complete for general LDSRG(2).

2. DSRG-MRPT2 Analytic Energy Gradients
+++++++++++++++++++++++++++++++++++++++

3. MR-DSRG(T) with Perturbative Triples
+++++++++++++++++++++++++++++++++++++++


.. _dsrg_ref:

References
^^^^^^^^^^

The seminal work of DSRG is given in:

* "A driven similarity renormalization group approach to quantum many-body problems",
  F. A. Evangelista, *J. Chem. Phys.* **141**, 054109 (2014).
  (doi: `10.1063/1.4890660 <http://dx.doi.org/10.1063/1.4890660>`_).

A general and pedagogical discussion of MR-DSRG is presented in:

* "Multireference Theories of Electron Correlation Based
  on the Driven Similarity Renormalization Group", C. Li and F. A. Evangelista,
  *Annu. Rev. Phys. Chem.* **70**, 245-273 (2019).
  (doi: `10.1146/annurev-physchem-042018-052416
  <http://dx.doi.org/10.1146/annurev-physchem-042018-052416>`_).

The theories of different DSRG correlation levels are discussed in the following articles:

    DSRG-MRPT2 (without reference relaxation):

    * "Multireference Driven Similarity Renormalization Group:
      A Second-Order Perturbative Analysis", C. Li and F. A. Evangelista,
      *J. Chem. Theory Compt.* **11**, 2097-2108 (2015).
      (doi: `10.1021/acs.jctc.5b00134 <http://dx.doi.org/10.1021/acs.jctc.5b00134>`_).

    DSRG-MRPT3 and variants of reference relaxations:

    * "Driven similarity renormalization group: Third-order multireference perturbation theory",
      C. Li and F. A. Evangelista, *J. Chem. Phys.* **146**, 124132 (2017).
      (doi: `10.1063/1.4979016 <http://dx.doi.org/10.1063/1.4979016>`_).
      Erratum: **148**, 079902 (2018).
      (doi: `10.1063/1.5023904 <http://dx.doi.org/10.1063/1.5023904>`_).

    MR-LDSRG(2):

    * "Towards numerically robust multireference theories: The driven similarity renormalization
      group truncated to one- and two-body operators", C. Li and F. A. Evangelista,
      *J. Chem. Phys.* **144**, 164114 (2016).
      (doi: `10.1063/1.4947218 <http://dx.doi.org/10.1063/1.4947218>`_).
      Erratum: **148**, 079903 (2018).
      (doi: `10.1063/1.5023493 <http://dx.doi.org/10.1063/1.5023493>`_).

The DSRG extensions for excited state are discussed in the following articles:

    SA-DSRG framework and its PT2 and PT3 applications:

    * "Driven similarity renormalization group for excited states:
      A state-averaged perturbation theory", C. Li and F. A. Evangelista,
      *J. Chem. Phys.* **148**, 124106 (2018).
      (doi: `10.1063/1.5019793 <http://dx.doi.org/10.1063/1.5019793>`_).

    MS-DSRG and DWMS-DSRG:

    * "Dynamically weighted multireference perturbation theory: Combining the advantages
      of multi-state and state- averaged methods", C. Li and F. A. Evangelista,
      *J. Chem. Phys.* **150**, 144107 (2019).
      (doi: `10.1063/1.5088120 <http://dx.doi.org/10.1063/1.5088120>`_).

The DSRG analytic energy gradients are described in the following series of papers:

    Single reference DSRG-PT2:

    * "Analytic gradients for the single-reference driven similarity renormalization group
      second-order perturbation theory", S. Wang, C. Li, and F. A. Evangelista,
      *J. Chem. Phys.* **151**, 044118 (2019).
      (doi: `10.1063/1.5100175 <http://dx.doi.org/10.1063/1.5100175>`_).

The integral-factorized implementation of DSRG is firstly achieved in:

* "An integral-factorized implementation of the driven similarity renormalization group
  second-order multireference perturbation theory", K. P. Hannon, C. Li, and F. A. Evangelista,
  *J. Chem. Phys.* **144**, 204111 (2016).
  (doi: `10.1063/1.4951684 <http://dx.doi.org/10.1063/1.4951684>`_).

The sequential variant of MR-LDSRG(2) and NIVO approximation are described in:

* "Improving the Efficiency of the Multireference Driven Similarity Renormalization Group
  via Sequential Transformation, Density Fitting, and the Noninteracting Virtual Orbital
  Approximation", T. Zhang, C. Li, and F. A. Evangelista,
  *J. Chem. Theory Compt.* **15**, 4399-4414 (2019).
  (doi: `10.1021/acs.jctc.9b00353 <http://dx.doi.org/10.1021/acs.jctc.9b00353>`_).

Combination between DSRG and adaptive configuration interaction with applications to acenes:

* "A Combined Selected Configuration Interaction and Many-Body Treatment of Static and Dynamical
  Correlation in Oligoacenes", J. B. Schriber, K. P. Hannon, C. Li, and F. A. Evangelista,
  *J. Chem. Theory Compt.* **14**, 6295-6305 (2018).
  (doi: `10.1021/acs.jctc.8b00877 <http://dx.doi.org/10.1021/acs.jctc.8b00877>`_).

Benchmark of state-specific unrelaxed DSRG-MRPT2 (tested 34 active orbitals):

* "A low-cost approach to electronic excitation energies based on the driven
  similarity renormalization group", C. Li, P. Verma, K. P. Hannon, and
  F. A. Evangelista, *J. Chem. Phys.* **147**, 074107 (2017).
  (doi: `10.1063/1.4997480 <http://dx.doi.org/10.1063/1.4997480>`_).

