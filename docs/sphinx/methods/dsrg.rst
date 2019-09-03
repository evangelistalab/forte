.. _`sec:methods:ldsrg`:

Driven Similarity Renormalization Group
=======================================

.. codeauthor:: Francesco A. Evangelista, Chenyang Li, Kevin Hannon, Tianyuan Zhang
.. sectionauthor:: Chenyang Li, Kevin P. Hannon, Tianyuan Zhang

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
dependence. See Table I for different flavours of :math:`\hat{S}`.

Table I. Connections of DSRG to CC theories when using different types of :math:`\hat{S}`:

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
various DSRG perturbation theories (e.g., 2nd-order or 3rd-order).
Note we use the RSC approximated BCH equation for computational cost considerations.
As such, the implemented DSRG-PT3 is **not** a complete PT3 but a companion PT3 of the LDSRG(2) method.

To conclude this subsection, we discuss the computational cost and current implementation limit.
These aspects are summarized in Table II.

Table II. Cost of the various implemented DSRG methods:

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


3. General DSRG Options
+++++++++++++++++++++++

**CORR_LEVEL**

Correlation level of MR-DSRG.

* Type: string
* Options: PT2, PT3, LDSRG2, LDSRG2_QC, LSRG2, SRG_PT2, QDSRG2, LDSRG2_P3, QDSRG2_P3
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


Integral Factorization
^^^^^^^^^^^^^^^^^^^^^^

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
For larger systems, please use the **DiskDF** approach where these integrals are loaded to memory only when necessary.
In general, we can treat about 2000 basis functions (with DiskDF) using DSRG-MRPT2.

Density fitting is more suited to spin-adapted equations while the current code uses spin-integrated equations.

We have a more optimized code of DF-DSRG-MRPT2.
The batching algorithms of DSRG-MRPT3 (manually tuned) and MR-LDSRG(2) (Ambit) are currently not ideal.

3. Examples
+++++++++++

.. caution::
  For DSRG-MRPT3 and MR-LDSRG(2), DF/CD will automatically turn on if **INT_TYPE** is set to **DF**, **CD**, or **DISKDF**.
  For DSRG-MRPT2 computations, please set the **CORRELATION_SOLVER** keyword to **THREE-DSRG-MRPT2** besides the **INT_TYPE** option.

The following input performs a DF-DSRG-MRPT2 calculation on nitrogen molecule.
This example is modified from the df-dsrg-mrpt2-4 test case.

.. note:: In test case df-dsrg-mrpt2-4, **SCF_TYPE** is specified to **PK**, which is incorrect for a real computation.

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
   **SCF_TYPE**, **DF_BASIS_SCF**, **DF_BASIS_MP2**
2. Forte options:
   **CORRELATION_SOLVER**, **INT_TYPE**
We recommend using consistent **DF_BASIS_SCF** and **DF_BASIS_MP2** due to orbital canonicalization.
Otherwise, the orbitals will be re-canonicalized before DSRG-MRPT2 computaionis.

.. note::
  Frozen orbitals will be left unchanged (in the original DF_BASIS_SCF basis) for orbital canonicalization.
  Thus, using inconsistent DF basis sets leads to inconsistent frozen-core energies.

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

To use Cholesky integrals, set **INT_TYPE** to **CHOLESKY** and specify **CHOLESKY_TOLERANCE**.
Running cholesky_tolerance with 1e-5 provides energies accurate to this tolerance comparing to conventional four-index 2-electron integrals.
The cholesky algorithm, as currently written, does not allow application to large systems (> 1000 basis functions).

4. Options
++++++++++

Please check the basic options in :ref:`sec:integrals`.

Advanced Keywords
'''''''''''''''''

**CCVV_BATCH_NUMBER**

Manually specify the number of batches for computing THREE-DSRG-MRPT2 energies.
By default, the number of batches are automatically computed using the remaining memory estimate.

* Type: integer
* Default: -1

State-Averaged and Multi-State Approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


TODOs
^^^^^

1. Spin Adaptation
++++++++++++++++++

This is done for unrelaxed DSRG-MRPT2 but not complete for general LDSRG(2).

2. DSRG-MRPT2 Analytic Energy Gradients
+++++++++++++++++++++++++++++++++++++++

3. MR-DSRG(T) with Perturbative Triples
+++++++++++++++++++++++++++++++++++++++
