.. _`sec:methods:ldsrg`:

Driven Similarity Renormalization Group
=======================================

.. codeauthor:: Chenyang Li, Kevin P. Hannon, Tianyuan Zhang, Francesco A. Evangelista

.. sectionauthor:: Chenyang Li, Francesco A. Evangelista, Tianyuan Zhang, Kevin P. Hannon

.. important::
  Any publication utilizing the DSRG code should acknowledge the following articles:

  * F. A. Evangelista, *J. Chem. Phys.* **141**, 054109 (2014).

  * C. Li and F. A. Evangelista, *Annu. Rev. Phys. Chem.* **70**, 245-273 (2019).

  Depending on the features used, the user is encouraged to cite the corresponding articles listed :ref:`here <dsrg_ref>`.

.. caution::
  The examples used in this manual are written based on the spin-integrated code.
  To make the spin-integrated code work properly for molecules with **even** multiplicities [S \* (S + 1) = 2, 4, 6, ...],
  the user should specify the following keyword:
  ::

     spin_avg_density    true       # use spin-summed reduced density matrices

  to invoke the use of spin-free densities.
  The spin-free densities are computed by averaging all spin multiplets (e.g., Ms = 1/2 or -1/2 for doublets).
  For odd multiplicities [S \* (S + 1) = 1, 3, 5, ...], there is no need to do so.
  Please check test case :ref:`dsrg-mrpt2-13 <dsrg_example>` for details.

.. Note::
  The latest version of Forte also has the spin-adapted MR-DSRG implemented for
  DSRG-MRPT2, DSRG-MRPT3, and MR-LDSRG(2) (and its variants).
  To invoke the spin-adated implementation, the user needs to specify the following keywords:
  ::

     correlation_solver  sa-mrdsrg  # spin-adapted DSRG computation
     corr_level          ldsrg2     # spin-adapted theories: PT2, PT3, LDSRG2_QC, LDSRG2

  The spin-adapted version should be at least 2-3 times faster than the corresponding spin-integrated code,
  and it also saves some memory.
  Note that the spin-adapted code will ignore the ``spin_avg_density`` keyword and always treat it as ``true``.

.. _`basic_dsrg`:

Basics of  DSRG
^^^^^^^^^^^^^^^

1. Overview of DSRG Theory
++++++++++++++++++++++++++

Driven similarity renormalization group (DSRG) is a numerically robust approach to treat dynamical (or weak) electron correlation.
Specifically, the DSRG performs a *continuous* similarity transformation of the bare Born-Oppenheimer Hamiltonian :math:`\hat{H}`,

.. math:: \bar{H}(s) = e^{-\hat{S}(s)} \hat{H} e^{\hat{S}(s)},

where :math:`s` is the flow parameter defined in the range :math:`[0, +\infty)`. The value of :math:`s` controls the amount of dynamical correlation included in :math:`\bar{H}(s)`, with :math:`s = 0` corresponding to no correlation included.
The operator :math:`\hat{S}` can be any operator in general.
For example, if :math:`\hat{S} = \hat{T}` is the coupled cluster substitution operator, the DSRG :math:`\bar{H}(s)`
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

The DSRG transformed Hamiltonian :math:`\bar{H}(s)` contains many-body (> 2-body) interactions in general.
We can express it as

.. math:: \bar{H} = \bar{h}_0 + \bar{h}^{p}_{q} \{ a^{q}_{p} \} + \frac{1}{4} \bar{h}^{pq}_{rs} \{ a^{rs}_{pq} \} + \frac{1}{36} \bar{h}^{pqr}_{stu} \{ a^{stu}_{pqr} \} + ...

where :math:`a^{pq...}_{rs...} = a_{p}^{\dagger} a_{q}^{\dagger} \dots a_s a_r` is a string of creation and annihilation operators
and :math:`\{\cdot\}` represents normal-ordered operators. In particular, we use Mukherjee-Kutzelnigg normal ordering
[see J. Chem. Phys. 107, 432 (1997)] with respect to a general multideterminantal reference :math:`\Psi_0`. Here we also assume summations over repeated indices for brevity.
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
As such, the implemented DSRG-PT3 is **not** a formally complete PT3, but a numerically efficient companion theory to the LDSRG(2) method.

To conclude this subsection, we discuss the computational cost and current implementation limit,
which are summarized in :ref:`Table II <table:dsrg_cost>`.

.. _`table:dsrg_cost`:

.. table:: Table II. Cost and maximum system size for the DSRG methods implemented in Forte.

    +----------+-----------------------+----------------------------------+-----------------------------------+
    |  Method  |  Computational Cost   |   Conventional 2-el. integrals   |   Density-fitted/Cholesky (DF/CD) |
    +==========+=======================+==================================+===================================+
    |    PT2   | one-shot :math:`N^5`  | :math:`\sim 250` basis functions | :math:`\sim 1800` basis functions |
    +----------+-----------------------+----------------------------------+-----------------------------------+
    |    PT3   | one-shot :math:`N^6`  | :math:`\sim 250` basis functions | :math:`\sim 700` basis functions  |
    +----------+-----------------------+----------------------------------+-----------------------------------+
    | LDSRG(2) | iterative :math:`N^6` | :math:`\sim 200` basis functions | :math:`\sim 550` basis functions  |
    +----------+-----------------------+----------------------------------+-----------------------------------+

.. _`basic_dsrg_example`:

2. Input Examples
+++++++++++++++++

**Minimal Example - DSRG-MPT2 energy of HF**

Let us first see an example with minimal keywords.
In particular, we compute the energy of hydrogen fluoride using DSRG multireference (MR) PT2
using a complete active space self-consistent field (CASSCF) reference.

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

1. The :code:`molecule` block specifies the geometry, charge, multiplicity, etc.

2. The second block specifies Psi4 options (see Psi4 manual for details).

3. The last block shows options specifically for Forte.

In this example, we use Psi4 to compute CASSCF reference.
Psi4 provides the freedom to specify the core (a.k.a. internal) and active orbitals
using :code:`RESTRICTED_DOCC` and :code:`ACTIVE` options,
but *it is generally the user's responsibility to select and verify correct orbital ordering*.
The :code:`RESTRICTED_DOCC` array :code:`[2,0,1,1]` indicates two :math:`a_1`,
zero :math:`a_2`, one :math:`b_1`, and one :math:`b_2` doubly occupied orbitals.
There are four irreps because the computation is performed using :math:`C_{2v}` point group symmetry.

The computation begins with the execution of Psi4's CASSCF code, invoked by
:code:`Emcscf, wfn = energy('casscf', return_wfn=True)`. This function call returns the energy and CASSCF wave function. In the second call to the energy function, :code:`energy('forte', ref_wfn=wfn)`, we ask the Psi4 driver to call Forte. The wave function stored in :code:`wfn` will is passed to Forte via argument :code:`ref_wfn`.

Forte generally recomputes the reference using the provided wave function parameters.
To perform a DSRG computation, the user is expected to specify the following keywords:

* :code:`ACTIVE_SPACE_SOLVER`:
  Here we use :code:`FCI` to perform a CAS configuration interaction (CASCI),
  i.e., a full CI within the active orbitals.

* :code:`CORRELATION_SOLVER`:
  This option determines which code to run. The four well-tested DSRG solvers are:
  :code:`DSRG-MRPT2`, :code:`THREE-DSRG-MRPT2`, :code:`DSRG-MRPT3`, and :code:`MRDSRG`.
  The density-fitted DSRG-MRPT2 is implemented in :code:`THREE-DSRG-MRPT2`.
  The :code:`MRDSRG` is mainly designed to perform MR-LDSRG(2) computations.

* :code:`DSRG_S`:
  This keyword specifies the DSRG flow parameter in a.u.
  For general MR-DSRG computations, the user should change the value to :math:`0.5 \sim 1.5` a.u.
  Most of our computations in :ref:`dsrg_ref` are performed using 0.5 or 1.0 a.u.

  .. caution::
    By default, :code:`DSRG_S` is set to :math:`0.5` a.u.
    The user should always set this keyword by hand!
    Non-perturbative methods may not converge for large values of flow parameter.

* Orbital spaces:
  Here we also specify frozen core orbitals besides core and active orbitals.
  Note that in this example, we optimize the 1s-like core orbital in CASSCF but
  later freeze it in the DSRG treatments of dynamical correlation.
  Details regarding to orbital spaces can be found in the section :ref:`sec:mospaceinfo`.

  .. tip::
    To perform a single-reference (SR) DSRG computation, set the array :code:`ACTIVE` to zero.
    In the above example, the SR DSRG-PT2 energy can be obtained
    by modifying :code:`RESTRICTED_DOCC` to :code:`[2,0,1,1]`
    and :code:`ACTIVE` to :code:`[0,0,0,0]`. The MP2 energy can be reproduced
    if we further change :code:`DSRG_S` to very large values (e.g., :math:`10^8` a.u.).

The output of the above example consists of several parts:

* The active-space FCI computation: ::

    ==> Root No. 0 <==

      20     -0.95086442
      02      0.29288371

      Total Energy:       -99.939316382616340

    ==> Energy Summary <==

      Multi.  Irrep.  No.               Energy
      -----------------------------------------
         1      A1     0       -99.939316382616
      -----------------------------------------

  Forte prints out the largest determinants in the CASCI wave function and its energy.
  Since we read orbitals from Psi4's CASSCF, this energy should coincide with Psi4's CASSCF energy.

* The computation of 1-, 2-, and 3-body reduced density matrices (RDMs) of the CASCI reference: ::

    ==> Computing RDMs for Root No. 0 <==

      Timing for 1-RDM: 0.000 s
      Timing for 2-RDM: 0.000 s
      Timing for 3-RDM: 0.000 s

* Canonicalization of the orbitals: ::

    ==> Checking Fock Matrix Diagonal Blocks <==

      Off-Diag. Elements       Max           2-Norm
      ------------------------------------------------
      Fa actv              0.0000000000   0.0000000000
      Fb actv              0.0000000000   0.0000000000
      ------------------------------------------------
      Fa core              0.0000000000   0.0000000000
      Fb core              0.0000000000   0.0000000000
      ------------------------------------------------
      Fa virt              0.0000000000   0.0000000000
      Fb virt              0.0000000000   0.0000000000
      ------------------------------------------------
    Orbitals are already semicanonicalized.

  All DSRG procedures require the orbitals to be canonicalized. In this basis, the core, active, and virtual diagonal blocks of the average Fock matrix are diagonal.
  Forte will test if the orbitals provided are canonical, and if not it will perform a canonicalization.
  In this example, since Psi4's CASSCF orbitals are already canonical, Forte just tests the Fock matrix
  but does not perform an actual orbital rotation.

* Computation of the DSRG-MRPT2 energy:

  - The output first prints out a summary of several largest amplitudes and possible intruders: ::

      ==> Excitation Amplitudes Summary <==

      Active Indices:    1    2
      ...  # ommit output for T1 alpha, T1 beta, T2 alpha-alpha, T2 beta-beta
      Largest T2 amplitudes for spin case AB:
             _       _                  _       _                  _       _
         i   j   a   b              i   j   a   b              i   j   a   b
      --------------------------------------------------------------------------------
      [  1   2   2   4] 0.055381 [  0   0   1   1]-0.053806 [  1   2   1   4] 0.048919
      [  1  14   1  15] 0.047592 [  1  10   1  11] 0.047592 [  2   2   4   4]-0.044138
      [  2  14   1  15] 0.042704 [  2  10   1  11] 0.042704 [  1  10   1  12]-0.040985
      [  1  14   1  16]-0.040985 [  2   2   1   4] 0.040794 [  1   1   1   5] 0.040479
      [  1  14   2  15] 0.036004 [  1  10   2  11] 0.036004 [  2  10   2  12]-0.035392
      --------------------------------------------------------------------------------
      Norm of T2AB vector: (nonzero elements: 1487)                 0.369082532477979.
      --------------------------------------------------------------------------------

    Here, {i, j} are generalized hole indices and {a, b} indicate generalized particle indices.
    The active indices are given at the beginning of this printing block.
    Thus, the largest amplitude in this case [(1,2) -> (2,4)] is a semi-internal excitation
    from (active, active) to (active, virtual).
    In general, semi-internal excitations tend to be large and they are suppressed by DSRG.

  - An energy summary is given later in the output: ::

      ==> DSRG-MRPT2 Energy Summary <==

        E0 (reference)                 =    -99.939316382616383
        <[F, T1]>                      =     -0.010942204196708
        <[F, T2]>                      =      0.011247157867728
        <[V, T1]>                      =      0.010183611834684
        <[V, T2]> (C_2)^4              =     -0.213259856801491
        <[V, T2]> C_4 (C_2)^2 HH       =      0.002713363798054
        <[V, T2]> C_4 (C_2)^2 PP       =      0.012979097502477
        <[V, T2]> C_4 (C_2)^2 PH       =      0.027792466274407
        <[V, T2]> C_6 C_2              =     -0.003202673882957
        <[V, T2]>                      =     -0.172977603109510
        DSRG-MRPT2 correlation energy  =     -0.162489037603806
        DSRG-MRPT2 total energy        =   -100.101805420220188
        max(T1)                        =      0.097879100308377
        max(T2)                        =      0.055380911136950
        ||T1||                         =      0.170534584213259
        ||T2||                         =      0.886328961933259

   Here we show all contributions to the energy. Specifically, those labeled by C_4
   involve 2-body density cumulants, and those labeled by C_6 involve 3-body cumulants.


**A More Advanced Example - MR-LDSRG(2) energy of HF**

Here we look at a more advanced example of MR-LDSRG(2) using the same molecule. ::

    # We just show the input block of Forte here.
    # The remaining input is identical to the previous example.

    set forte{
       active_space_solver     fci
       correlation_solver      mrdsrg
       corr_level              ldsrg2
       frozen_docc             [1,0,0,0]
       restricted_docc         [1,0,1,1]
       active                  [2,0,0,0]
       dsrg_s                  0.5
       e_convergence           1.0e-8
       dsrg_rsc_threshold      1.0e-9
       relax_ref               iterate
    }

.. warning::
  This example takes a long time to finish (~3 min on a 2018 15-inch MacBook Pro).

There are several things to notice.

1. To run a MR-LDSRG(2) computation, we need to change :code:`CORRELATION_SOLVER` to :code:`MRDSRG`.
   Additionally, the :code:`CORR_LEVEL` should be specified as :code:`LDSRG2`.
   There are other choices of :code:`CORR_LEVEL` but they are mainly for testing new ideas.

2. We specify the energy convergence keyword :code:`E_CONVERGENCE` and the RSC threshold :code:`DSRG_RSC_THRESHOLD`,
   which controls the truncation of the recursive single commutator (RSC) approximation of the DSRG Hamiltonian.
   In general, the value of :code:`DSRG_RSC_THRESHOLD` should be smaller than that of :code:`E_CONVERGENCE`.
   Making :code:`DSRG_RSC_THRESHOLD` larger will stop the BCH series earlier and thus saves some time.
   It is OK to leave :code:`DSRG_RSC_THRESHOLD` as the default value, which is :math:`10^{-12}` a.u.

3. The MR-LDSRG(2) method includes reference relaxation effects.
   There are several variants of reference relaxation levels (see :ref:`dsrg_variants`).
   Here we use the fully relaxed version, which is done by setting :code:`RELAX_REF` to :code:`ITERATE`.

.. note::
  The reference relaxation procedure is performed in a tick-tock way (see :ref:`dsrg_variants`),
  by alternating the solution of the DSRG amplitude equations and the diagonalization of the DSRG Hamiltonian.
  This procedure may not monotonically converge and is potentially numerically unstable.
  We therefore suggest using a moderate energy threshold (:math:`\geq 10^{-8}` a.u.) for the iterative reference relaxation,
  which is controlled by the option :code:`RELAX_E_CONVERGENCE`.

For a given reference wave function, the output prints out a summary of:

1. The iterations for solving the amplitudes, where each step involves building a DSRG transformed Hamiltonian.

2. The MR-LDSRG(2) energy: ::

    ==> MR-LDSRG(2) Energy Summary <==

      E0 (reference)                 =     -99.939316382616383
      MR-LDSRG(2) correlation energy =      -0.171613035562048
      MR-LDSRG(2) total energy       =    -100.110929418178429

3. The MR-LDSRG(2) converged amplitudes: ::

    ==> Final Excitation Amplitudes Summary <==

      Active Indices:    1    2
      ...  # ommit output for T1 alpha, T1 beta, T2 alpha-alpha, T2 beta-beta
      Largest T2 amplitudes for spin case AB:
             _       _                  _       _                  _       _
         i   j   a   b              i   j   a   b              i   j   a   b
      --------------------------------------------------------------------------------
      [  0   0   1   1]-0.060059 [  1   2   2   4] 0.046578 [  1  10   1  11] 0.039502
      [  1  14   1  15] 0.039502 [  0   0   1   2]-0.038678 [  1   1   1   5] 0.037546
      [  2   2   4   4]-0.033871 [  1   2   1   4] 0.033125 [  1  14   2  15] 0.032868
      [  1  10   2  11] 0.032868 [  1  10   1  12]-0.032602 [  1  14   1  16]-0.032602
      [ 14  14  15  15]-0.030255 [ 10  10  11  11]-0.030255 [  2  14   1  15] 0.029241
      --------------------------------------------------------------------------------
      Norm of T2AB vector: (nonzero elements: 1487)                 0.330204946109119.
      --------------------------------------------------------------------------------

At the end of the computation, Forte prints a summary of the energy during the reference relaxation iterations: ::

    => MRDSRG Reference Relaxation Energy Summary <=

                           Fixed Ref. (a.u.)                  Relaxed Ref. (a.u.)
             -----------------------------------  -----------------------------------
      Iter.          Total Energy          Delta          Total Energy          Delta
      -------------------------------------------------------------------------------
          1     -100.110929418178 (a) -1.001e+02     -100.114343552853 (b) -1.001e+02
          2     -100.113565563124 (c) -2.636e-03     -100.113571036112      7.725e-04
          3     -100.113534597590      3.097e-05     -100.113534603824      3.643e-05
          4     -100.113533334887      1.263e-06     -100.113533334895      1.269e-06
          5     -100.113533290863      4.402e-08     -100.113533290864      4.403e-08
          6     -100.113533289341      1.522e-09     -100.113533289341 (d)  1.522e-09
      -------------------------------------------------------------------------------

Let us introduce the nomenclature for reference relaxation.

   ====================  =========================  =============================
          Name              Example Value               Description
   ====================  =========================  =============================
   a) Unrelaxed          :code:`-100.110929418178`  1st iter.; fixed CASCI ref.
   b) Partially Relaxed  :code:`-100.114343552853`  1st iter.; relaxed CASCI ref.
   c) Relaxed            :code:`-100.113565563124`  2nd iter.; fixed ref.
   d) Fully Relaxed      :code:`-100.113533289341`  last iter.; relaxed ref.
   ====================  =========================  =============================

   The unrelaxed energy is a diagonalize-then-perturb scheme,
   while the partially relaxed energy corresponds to a diagonalize-then-perturb-then-diagonalize method.
   In this example, the fully relaxed energy is well reproduced by
   the relaxed energy with a small error (:math:`< 10^{-4}` a.u.).

**Other Examples**

There are plenty of examples in the tests/method folder.
A complete list of the DSRG test cases can be found :ref:`here <dsrg_example>`.

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
* Default: 0.5

**DSRG_MAXITER**

Max iterations for MR-DSRG amplitudes update.

* Type: integer
* Default: 50

**DSRG_RSC_NCOMM**

The maximum number of commutators in the recursive single commutator approximation to the BCH formula.

* Type: integer
* Default: 20

**DSRG_RSC_THRESHOLD**

The threshold of considering the BCH expansion converged based on the recursive single commutator approximation.

* Type: double
* Default: 1.0e-12

**R_CONVERGENCE**

The convergence criteria for the amplitudes.

* Type: double
* Default: 1.0e-6

**NTAMP**

The number of largest amplitudes printed in the amplitudes summary.

* Type: integer
* Default: 15

**INTRUDER_TAMP**

A threshold for amplitudes that are considered as intruders for printing.

* Type: double
* Default: 0.1

**TAYLOR_THRESHOLD**

A threshold for small energy denominators that are computed using Taylor expansion
(instead of direct reciprocal of the energy denominator).
For example, 3 means Taylor expansion is performed if denominators are smaller than 1.0e-3.

* Type: integer
* Default: 3

**DSRG_DIIS_START**

The minimum iteration to start storing DIIS vectors for MRDSRG amplitudes.
Any number smaller than 1 will turn off the DIIS procedure.

* Type: int
* Default: 2

**DSRG_DIIS_FREQ**

How often to do a DIIS extrapolation in MRDSRG iterations.
For example, 1 means do DIIS every iteration and 2 is for every other iteration, etc.

* Type: int
* Default: 1

**DSRG_DIIS_MIN_VEC**

Minimum number of error vectors stored for DIIS extrapolation in MRDSRG.

* Type: int
* Default: 3

**DSRG_DIIS_MAX_VEC**

Maximum number of error vectors stored for DIIS extrapolation in MRDSRG.

* Type: int
* Default: 8

.. _dsrg_variants:

Theoretical Variants and Technical Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Reference Relaxation
+++++++++++++++++++++++

For MR methods, it is necessary to consider reference relaxation effects
due to coupling between static and dynamical correlation.
This can be introduced by requiring the reference wave function,
:math:`\Psi_0` to be the eigenfunction of :math:`\bar{H}(s)`.
The current implementation uses the uncoupled two-step (tick-tock) approach, where
the DSRG transformed Hamiltonian :math:`\bar{H}(s)` is built using the RDMs of a given :math:`\Psi_0`,
and then diagonalize :math:`\bar{H}(s)` within the active space yielding a new :math:`\Psi_0`.
These two steps can be iteratively performed until convergence.

Denoting the :math:`i`-th iteration of reference relaxation by superscript :math:`[i]`,
the variants of reference relaxation procedure introduced above can be expressed as

   =================  ===============================================================================
          Name                               Energy Expression
   =================  ===============================================================================
   Unrelaxed          :math:`\langle \Psi_0^{[0]} | \bar{H}^{[0]} (s) | \Psi_0^{[0]} \rangle`
   Partially Relaxed  :math:`\langle \Psi_0^{[1]} (s) | \bar{H}^{[0]} (s) | \Psi_0^{[1]} (s) \rangle`
   Relaxed            :math:`\langle \Psi_0^{[1]} (s) | \bar{H}^{[1]} (s) | \Psi_0^{[1]} (s) \rangle`
   Fully Relaxed      :math:`\langle \Psi_0^{[n]} (s) | \bar{H}^{[n]} (s) | \Psi_0^{[n]} (s) \rangle`
   =================  ===============================================================================

where :math:`[0]` uses the original reference wave function and :math:`[n]` suggests converged results.

By default, :code:`MRDSRG` only performs an unrelaxed computation.
To obtain partially relaxed energy, the user needs to change :code:`RELAX_REF` to :code:`ONCE`.
For relaxed energy, :code:`RELAX_REF` should be switched to :code:`TWICE`.
For fully relaxed energy, :code:`RELAX_REF` should be set to :code:`ITERATE`.

For other DSRG solvers aimed for perturbation theories, only the unrelaxed and partially relaxed energies are available.
In the literature, we term the partially relaxed version as the default DSRG-MRPT,
while the unrelaxed version as uDSRG-MRPT.

.. tip::
  These energies can be conveniently obtained in the input file.
  For example, :code:`Eu = variable("UNRELAXED ENERGY")` puts unrelaxed energy to a variable :code:`Eu`.
  The available keys are :code:`"UNRELAXED ENERGY"`, :code:`PARTIALLY RELAXED ENERGY`,
  :code:`"RELAXED ENERGY"`, and :code:`"FULLY RELAXED ENERGY"`.

2. Orbital Rotations
++++++++++++++++++++

The DSRG equations are defined in the semicanonical orbital basis,
and thus it is not generally orbital invariant.
All DSRG solvers, except for :code:`THREE-DSRG-MRPT2`, automatically rotates the integrals to semicanonical basis
even if the input integrals are not canonicalized (if keyword :code:`SEMI_CANONICAL` is set to :code:`FALSE`).
However, it is recommended a careful inspection to the printings regarding to the semicanonical orbitals.
An example printing of orbital canonicalization can be found in :ref:`Minimal Example <basic_dsrg_example>`.

3. Sequential Transformation
++++++++++++++++++++++++++++

In the sequential transformation ansatz, we compute :math:`\bar{H}` sequentially as

.. math:: \bar{H}(s) = e^{-\hat{A}_n} \cdots e^{-\hat{A}_2} e^{-\hat{A}_1} \hat{H} e^{\hat{A}_1} e^{\hat{A}_2} \cdots e^{\hat{A}_n}

instead of the traditional approach:

.. math:: \bar{H}(s) = e^{-\hat{A}_1 - \hat{A}_2 - \cdots - \hat{A}_n} \hat{H} e^{\hat{A}_1 + \hat{A}_2 + \cdots + \hat{A}_n}

For clarity, we ignore the indication of :math:`s` dependence on :math:`\bar{H}(s)` and :math:`\hat{A}(s)`.
In the limit of :math:`s \rightarrow \infty` and no truncation of :math:`\hat{A}(s)`,
both the traditional and sequential MR-DSRG methods can approach the full configuration interaction limit.
The difference between their truncated results are also usually small.

In the sequential approach, :math:`e^{-\hat{A}_1} \hat{H} e^{\hat{A}_1}` is computed as a unitary transformation to the bare Hamiltonian,
which is very efficient when combined with integral factorization techniques (scaling reduction).

4. Non-Interacting Virtual Orbital Approximation
++++++++++++++++++++++++++++++++++++++++++++++++

In the non-interacting virtual orbital (NIVO) approximation,
we neglect the operator components of all rank-4 intermediate tensors and
:math:`\bar{H}` with three or more virtual orbital indices
(:math:`\mathbf{VVVV}`, :math:`\mathbf{VCVV}`, :math:`\mathbf{VVVA}`, etc.).
Consequently, the number of elements in the intermediates are reduced from :math:`{\cal O}(N^4)` to :math:`{\cal O}(N^2N_\mathbf{H}^2)`,
which is of similar size to the :math:`\hat{T}_2` amplitudes.
As such, the memory requirement of MR-LDSRG(2) is significantly reduced when we apply NIVO approximation
and combine with integral factorization techniques with a batched algorithm for tensor contractions.

Since much less number of tensor elements are involved, NIVO approximation dramatically reduces computation time.
However, the overall time scaling of MR-LDSRG(2) remain unchanged (prefactor reduction).
The error introduced by the NIVO approximation is usually negligible.

.. note::
  If conventional two-electron integrals are used, NIVO starts from the bare Hamiltonian term
  (i.e., :math:`\hat{H}` and all the commutators in the BCH expansion of :math:`\bar{H}` are approximated).
  For DF or CD intregrals, however, NIVO will start from the first commutator :math:`[\hat{H}, \hat{A}]`.

5. Zeroth-order Hamiltonian of DSRG-MRPT2 in MRDSRG Class
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

DSRG-MRPT2 is also implemented in the MRDSRG class for testing other zeroth-order Hamiltonian.
The general equation for all choices is to compute the summed second-order Hamiltonian:

.. math:: \bar{H}^{[2]} = \hat{H} + [\hat{H}, \hat{A}^{(1)}] + [\hat{H}^{(0)}, \hat{A}^{(2)}] + \frac{1}{2} [[\hat{H}^{(0)}, \hat{A}^{(1)}], \hat{A}^{(1)}]

where for brevity the :math:`(s)` notation is ignored and the superscripts of parentheses indicate the orders of perturbation.
We have implemented the following choices for the zeroth-order Hamiltonian.

**Diagonal Fock operator (Fdiag)**

  This choice contains the three diagonal blocks of the Fock matrix,
  that is, core-core, active-active, and virtual-virtual.
  Due to its simplicity, :math:`\bar{H}^{[2]}` can be obtained in a non-iterative manner in the semicanonical basis.

**Fock operator (Ffull)**

  This choice contains all the blocks of the Fock matrix.
  Since Fock matrix contains non-diagonal contributions, :math:`[\hat{H}^{(0)}, \hat{A}^{(2)}]` can contribute to the energy.
  As such, both first- and second-order amplitudes are solved iteratively.

**Dyall Hamiltonian (Fdiag_Vactv)**

  This choice contains the diagonal Fock matrix and the part of V labeled only by active indices.
  We solve the first-order amplitudes iteratively.
  However, :math:`[\hat{H}^{(0)}, \hat{A}]` will neither contribute to the energy nor the active part of the :math:`\bar{H}^{[2]}`.

**Fink Hamiltonian (Fdiag_Vdiag)**

  This choice contains all the blocks of Dyall Hamiltonian plus other parts of V that do not change the excitation level.
  For example, these additional blocks include: cccc, aaaa, vvvv, caca, caac, acac, acca,
  cvcv, cvvc, vcvc, vccv, avav, avva, vava, and vaav.
  The computation procedure is similar to that of Dyall Hamiltonian.

To use different types of zeroth-order Hamiltonian, the following options are needed
::

    correlation_solver      mrdsrg
    corr_level              pt2
    dsrg_pt2_h0th           Ffull

.. warning::
  The implementation of DSRG-MRPT2 in ``correlation_solver mrdsrg`` is different from the one in ``correlation_solver dsrg-mrpt2``.
  For the latter, the :math:`\hat{H}^{(0)}` is **assumed** being Fdiag and diagonal such that
  :math:`[\hat{H}^{(0)}, \hat{A}^{(1)}]` can be written in a compact form using semicanonical orbital energies.
  For ``mrdsrg``, :math:`[\hat{H}^{(0)}, \hat{A}^{(1)}]` is evaluated without any assumption to the form of :math:`\hat{H}^{(0)}`.
  These two approaches are equivalent for DSRG based on a CASCI reference.

  However, they will give different energies when there are multiple GAS spaces
  (In DSRG, all GAS orbitals are treated as ACTIVE).
  In this case, semicanonical orbitals are defined as those that make the diagonal blocks of the Fock matrix diagonal: core-core, virtual-virtual, GAS1-GAS1, GAS2-GAS2, ..., GAS6-GAS6.
  Then it is equivalent to say that ``dsrg-mrpt2`` uses all the diagonal blocks of the Fock matrix as zeroth-order Hamiltonian.
  In order to correctly treat the GAS :math:`m` - GAS :math:`n` (:math:`m \neq n`) part of Fock matrix as first-order Hamiltonian, one need to invoke internal excitations (i.e., active-active excitations).
  Contrarily, ``mrdsrg`` takes the entire active-active block of Fock matrix as zeroth-order Hamiltonian, that is all blocks of GAS :math:`m` - GAS :math:`n` (:math:`m, n \in \{1,2,\cdots,6\}`).

  The spin-adapted code ``correlation_solver sa-mrdsrg`` with ``corr_level pt2`` has the same behavior to the ``dsrg-mrpt2`` implementaion.

6. Restart iterative MRDSRG from a previous computation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

The convergence of iterative MRDSRG [e.g., MR-LDSRG(2)] can be greatly improved if it starts from good initial guesses
(e.g., from loosely converged amplitudes or those of a near-by geometry).
The amplitudes can be dumped to the current working directory on disk for later use by turning on the ``DSRG_DUMP_AMPS`` keyword.
These amplitudes are stored in a binary file using Ambit (version later than 06/30/2020).
For example, T1 amplitudes are stored as ``forte.mrdsrg.spin.t1.bin`` for the spin-integrated code
and ``forte.mrdsrg.adapted.t1.bin`` for spin-adapted code (i.e., `correlation_solver` set to `sa-mrdsrg`).
To read amplitudes in the current directory (must follow the same file name convention),
the user needs to invoke the ``DSRG_READ_AMPS`` keyword.

.. note::
  In general, we should make sure the orbital phases are consistent between reading and writing amplitudes.
  For example, the following shows part of the input to ensure the coefficient of the first AO being positive for all MOs. ::

    ...
    Escf, wfn = energy('scf', return_wfn=True)

    # fix orbital phase
    Ca = wfn.Ca().clone()
    nirrep = wfn.nirrep()
    rowdim, coldim = Ca.rowdim(), Ca.coldim()
    for h in range(nirrep):
        for i in range(coldim[h]):
            v = Ca.get(h, 0, i)
            if v < 0:
                for j in range(rowdim[h]):
                    Ca.set(h, j, i, -1.0 * Ca.get(h, j, i))
    wfn.Ca().copy(Ca)

    energy('forte', ref_wfn=wfn)

For reference relaxation, initial amplitudes are obtained from the previous converged values by default.
To turn this feature off (not recommended), please set ``DSRG_RESTART_AMPS`` to ``False``.

7. Examples
+++++++++++

Here we slightly modify the more advanced example in :ref:`General DSRG Examples <basic_dsrg_example>`
to adopt the sequential transformation and NIVO approximation. ::

    # We just show the input block of Forte here.

    set forte{
       active_space_solver     fci
       correlation_solver      mrdsrg
       corr_level              ldsrg2
       frozen_docc             [1,0,0,0]
       restricted_docc         [1,0,1,1]
       active                  [2,0,0,0]
       dsrg_s                  0.5
       e_convergence           1.0e-8
       dsrg_rsc_threshold      1.0e-9
       relax_ref               iterate
       dsrg_nivo               true
       dsrg_hbar_seq           true
    }

.. note::
  Since the test case is very small, invoking these two keywords does not make the computation faster.
  A significant speed improvement can be observed for a decent amout of basis functions (:math:`\sim 100`).

8. Related Options
++++++++++++++++++

**RELAX_REF**

Different approaches for MR-DSRG reference relaxation.

* Type: string
* Options: NONE, ONCE, TWICE, ITERATE
* Default: NONE

**RELAX_E_CONVERGENCE**

The energy convergence criteria for MR-DSRG reference relaxation.

* Type: double
* Default: 1.0e-8

**MAXITER_RELAX_REF**

Max macro iterations for MR-DSRG reference relaxation.

* Type: integer
* Default: 15

**DSRG_DUMP_RELAXED_ENERGIES**

Dump the energies after each reference relaxation step to JSON.
The energies include all computed states and the averaged DSRG "Fixed"
and "Relaxed" energies for every reference relaxation step.

* Type: Boolean
* Default: False

**DSRG_RESTART_AMPS**

Use converged amplitudes from the previous step as initial guesses of the current amplitudes.

* Type: Boolean
* Default: True

**SEMI_CANONICAL**

Semicanonicalize orbitals after solving the active-space eigenvalue problem.

* Type: Boolean
* Default: True

**DSRG_HBAR_SEQ**

Apply the sequential transformation algorithm in evaluating the transformed Hamiltonian :math:`\bar{H}(s)`, i.e.,

.. math:: \bar{H}(s) = e^{-\hat{A}_n(s)} \cdots e^{-\hat{A}_2(s)} e^{-\hat{A}_1(s)} \hat{H} e^{\hat{A}_1(s)} e^{\hat{A}_2(s)} \cdots e^{\hat{A}_n(s)}.

* Type: Boolean
* Default: False

**DSRG_NIVO**

Apply non-interacting virtual orbital (NIVO) approximation in evaluating the transformed Hamiltonian.

* Type: Boolean
* Default: False

**DSRG_PT2_H0TH**

The zeroth-order Hamiltonian used in the MRDSRG code for computing DSRG-MRPT2 energy.

* Type: string
* Options: FDIAG, FFULL, FDIAG_VACTV, FDIAG_VDIAG
* Default: FDIAG

**DSRG_DUMP_AMPS**

Dump amplitudes to the current directory for a MRDSRG method.
File names for T1 and T2 amplitudes are ``forte.mrdsrg.CODE.t1.bin``
and ``forte.mrdsrg.CODE.t2.bin``, respectively.
Here, ``CODE`` will be ``adapted`` if using the spin-adapted implementation,
while ``spin`` if using the spin-integrated code.

* Type: Boolean
* Default: False

**DSRG_READ_AMPS**

Read amplitudes from the current directory for iterative MRDSRG methods.
File format and content should match those with ``DSRG_DUMP_AMPS``.

* Type: Boolean
* Default: False


Density Fitted (DF) and Cholesky Decomposition (CD) Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Theory
+++++++++

Integral factorization, as it suggests, factorizes the two-electron integrals into contractions of low-rank tensors.
In particular, we use density fitting (DF) or Cholesky decomposition (CD) technique to express two-electron integrals as

.. math:: \langle ij || ab \rangle = \sum_{Q}^{N_\text{aux}} ( B_{ia}^{Q} B_{jb}^{Q} - B_{ib}^{Q} B_{ja}^{Q} )

where :math:`Q` runs over auxiliary indices.
Note that we use physicists' notation here but the DF/CD literature use chemist notation.

The main difference between DF and CD is how the :math:`B` tensor is formed.
In DF, the :math:`B` tensor is defined as

.. math:: B_{pq}^{Q} = \sum_P^{N_\text{aux}} (pq | P) (P | Q)^{-1/2}.

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

.. warning:: In test case df-dsrg-mrpt2-4, :code:`SCF_TYPE` is specified to :code:`PK`, which is incorrect for a real computation.

2. Forte options:
   :code:`CORRELATION_SOLVER`, :code:`INT_TYPE`

.. attention::
  Here we use different basis sets for :code:`DF_BASIS_SCF` and :code:`DF_BASIS_MP2`.
  There is no consensus on what basis sets should be used for MR computations.
  However, there is one caveat of using inconsistent DF basis sets in Forte due to orbital canonicalization:
  Frozen orbitals are left unchanged (i.e., canonical for :code:`DF_BASIS_SCF`)
  while DSRG (and orbital canonicalization) only reads :code:`DF_BASIS_MP2`.
  This inconsistency leads to slight deviations to the frozen-core energies (:math:`< 10^{-4}` a.u.)
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

.. note:: :code:`THREE-DSRG-MRPT2` currently does not print a summary for the largest amplitudes.

To use Cholesky integrals, set :code:`INT_TYPE` to :code:`CHOLESKY` and specify :code:`CHOLESKY_TOLERANCE`.
For example, a CD equivalence of the above example is ::

    # same molecule input ...

    set globals{
       reference               rhf
       basis                   cc-pvdz
       scf_type                cd                  # <=
       cholesky_tolerance      5                   # <=
       d_convergence           8
       e_convergence           10
    }

    set forte {
       active_space_solver     cas
       int_type                cholesky           # <=
       cholesky_tolerance      1.0e-5             # <=
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

The energy error of using CD integrals (threshold = :math:`10^{-5}` a.u.) is thus around :math:`\sim 10^{-5}` a.u..
In general, comparing to conventional 4-index 2-electron integrals, the use of CD integrals yields
energy errors to the same decimal points as :code:`CHOLESKY_TOLERANCE`.

.. caution:: The cholesky algorithm, as currently written, does not allow applications to large systems (> 1000 basis functions).

4. Related Options
++++++++++++++++++

For basic options of factorized integrals, please check :ref:`sec:integrals`.

**CCVV_BATCH_NUMBER**

Manually specify the number of batches for computing :code:`THREE-DSRG-MRPT2` energies.
By default, the number of batches are automatically computed using the remaining memory estimate.

* Type: integer
* Default: -1

MR-DSRG Approaches for Excited States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several MR-DSRG methods available for computing excited states.

.. warning::
  The current only supports SA-DSRG due to the revamp of Forte structure.
  MS-, XMS-, DWMS-DSRG will be available soon.

1. State-Averaged Formalism
+++++++++++++++++++++++++++

In state-averaged (SA) DSRG, the MK vacuum is an ensemble of electronic states,
which are typically obtained by an SA-CASSCF computation.
For example, we want to study two states, :math:`\Phi_1` and :math:`\Phi_2`,
described qualitatively by a CASCI with SA-CASSCF orbitals.
The ensemble of states (assuming equal weights) is characterized by the density operator

.. math:: \hat{\rho} = \frac{1}{2} | \Phi_1 \rangle \langle \Phi_1 | + \frac{1}{2} | \Phi_2 \rangle \langle \Phi_2 |

Note that :math:`\Phi_1` and :math:`\Phi_2` are just two of the many states (say, :math:`n`) in CASCI.

The bare Hamiltonian and cluster operators are normal ordered with respect to this ensemble,
whose information is embedded in the state-averaged densities.
An effective Hamiltonian :math:`\bar{H}` is then built by solving the DSRG cluster amplitudes.
In this way, the dynamical correlation is described for all the states lying in the ensemble.
Here, the DSRG solver and correlation levels remain the same to those of state-specific cases.
For example, we use :code:`DSRG-MRPT3` to do SA-DSRG-PT3.

Now we have many ways to proceed and obtain the excited states, two of which have been implemented.

- One approach is to diagonalize :math:`\bar{H}` using :math:`\Phi_1` and :math:`\Phi_2`.
  As such, the new states are just linear combinations of states in the ensemble and
  the CI coefficients are then constrained to be combined using :math:`\Phi_1` and :math:`\Phi_2`.
  We term this approach constrained SA, with a letter "c" appended at the end of a method name (e.g., SA-DSRG-PT2c).
  and in Forte we use the option :code:`SA_SUB` to specify this SA variant.

- The other approach is to diagonalize :math:`\bar{H}` using all configurations in CASCI,
  which allows all CI coefficients to relax.
  This approach is the default SA-DSRG approach, which is also the default in Forte.
  The corresponding option is :code:`SA_FULL`.

For both approaches, one could iterate these two-step (DSRG + diagoanlization) procedure
till convergence is reached.

.. note::
  For SA-DSRG, a careful inspection of the output CI coefficients is usually necessary.
  This is because the ordering of states may change after dynamical correlation is included.
  When that happens, a simple fix is to include more states in the ensemble,
  which may reduce the accuracy yet usually OK if only a few low-lying states are of interest.

.. tip::
   When the ground state is averaged, the three-body density cumulants can be safely ignored
   without affecting the vertical excitation energies.

.. tip::
   For spin-adapted implementations, it is possible to compute the oscillator strengths of
   dipole-allowed transitions using the DSRG-transformed dipole integrals by specifying the
   option :code:`DSRG_MAX_DIPOLE_LEVEL`, which indicates the max body of integrals kept.

2. Multi-State, Extended Multi-State Formalisms
+++++++++++++++++++++++++++++++++++++++++++++++

.. warning:: Not available at the moment.
.. note:: Only support at the PT2 level of theory.

In multi-state (MS) DSRG, we adopt the single-state parametrization where the effective Hamiltonian is built as

.. math:: H^{\rm eff}_{MN} = \langle \Phi_M | \hat{H} | \Phi_N \rangle + \frac{1}{2} \left[ \langle \Phi_M | \hat{T}_{M}^\dagger \hat{H} | \Phi_N \rangle + \langle \Phi_M | \hat{H} \hat{T}_N | \Phi_N \rangle \right],

where :math:`\hat{T}_{M}` is the state-specific cluster amplitudes for state :math:`M`,
that is, we solve DSRG-PT2 amplitudes :math:`\hat{T}_{M}` normal ordered to :math:`| \Phi_M \rangle`.
The MS-DSRG-PT2 energies are then obtained by diagonalizing this effective Hamiltonian.
However, it is known this approach leaves wiggles on the potential energy surface (PES) near
the strong coupling region of the reference wave functions.

A simple way to cure these artificial wiggles is to use the extended MS (XMS) approach.
In XMS DSRG, the reference states :math:`\tilde{\Phi}_M` are linear combinations of CASCI states
:math:`\Phi_M` such that the Fock matrix is diagonal.
Specifically, the Fock matrix is built according to

.. math:: F_{MN} = \langle \Phi_M | \hat{F} | \Phi_N \rangle,

where :math:`\hat{F}` is the state-average Fock operator.
Then in the mixed state basis, we have :math:`\langle \tilde{\Phi}_M | \hat{F} | \tilde{\Phi}_N \rangle = 0`, if :math:`M \neq N`.
The effective Hamiltonian is built similarly to that of MS-DSRG-PT2, except that :math:`\tilde{\Phi}_M` is used.

3. Dynamically Weighted Multi-State Formalism
+++++++++++++++++++++++++++++++++++++++++++++

.. warning:: Not available at the moment.
.. note:: Only support at the PT2 level of theory.

As shown by the XMS approach, mixing states is able to remove the wiggles on the PES.
Dynamically weighted MS (DWMS) approach provides an alternative way to mix zeroth-order states.
The idea of DWMS is closely related to SA-DSRG.
In DWMS, we choose an ensemble of zeroth-order reference states,
where the weights are automatically determined according to the energy separations between these reference states.
Specifically, the weight for target state :math:`M` is given by

.. math:: \omega_{MN} (\zeta) = \frac{e^{-\zeta (E_M^{(0)} - E_N^{(0)})^2}}{\sum_{P=1}^{n} e^{-\zeta(E_M^{(0)} - E_P^{(0)})^2}},

where :math:`E_M^{(0)} = \langle \Phi_M| \hat{H} | \Phi_M \rangle` is the zeroth-order energy of state :math:`M`
and :math:`\zeta` is a parameter to be set by the user.
Then we follow the MS approach to form an effective Hamiltonian
where the amplitudes are solved for the ensemble tuned to that particular state.

For a given value of :math:`zeta`, the weights of two reference states :math:`\Phi_M` and :math:`\Phi_N` will be equal
if they are degenerate in energy.
On the other limit where they are energetically far apart,
the ensemble used to determine :math:`\hat{T}_M` mainly consists of :math:`\Phi_M` with a little weight on :math:`\Phi_N`,
and vice versa.

For two non-degenerate states, by sending :math:`\zeta` to zero,
both states in the ensemble have equal weights (general for :math:`n` states),
which is equivalent to the SA formalism.
If we send :math:`\zeta` to :math:`\infty`, then the ensemble becomes state-specific.
Thus, parameter :math:`\zeta` can be understood as how drastic between the transition from MS to SA schemes.

.. caution::
  It is not guaranteed that the DWMS energy (for one adiabatic state) lies in between the MS and SA values.
  When DWMS energies go out of the bounds of MS and SA,
  a small :math:`\zeta` value is preferable to avoid rather drastic energy changes in a small geometric region.

4. Examples
+++++++++++

A simple example is to compute the lowest two states of :math:`\text{LiF}` molecule using SA-DSRG-PT2. ::

  import forte

  molecule {
    0 1
    Li
    F  1 R
    R = 10.000

    units bohr
  }

  basis {
    assign Li Li-cc-pvdz
    assign F  aug-cc-pvdz
  [ Li-cc-pvdz ]
  spherical
  ****
  Li     0
  S   8   1.00
     1469.0000000              0.0007660
      220.5000000              0.0058920
       50.2600000              0.0296710
       14.2400000              0.1091800
        4.5810000              0.2827890
        1.5800000              0.4531230
        0.5640000              0.2747740
        0.0734500              0.0097510
  S   8   1.00
     1469.0000000             -0.0001200
      220.5000000             -0.0009230
       50.2600000             -0.0046890
       14.2400000             -0.0176820
        4.5810000             -0.0489020
        1.5800000             -0.0960090
        0.5640000             -0.1363800
        0.0734500              0.5751020
  S   1   1.00
        0.0280500              1.0000000
  P   3   1.00
        1.5340000              0.0227840
        0.2749000              0.1391070
        0.0736200              0.5003750
  P   1   1.00
        0.0240300              1.0000000
  D   1   1.00
        0.1239000              1.0000000
  ****
  }

  set globals{
    reference           rhf
    scf_type            pk
    maxiter             300
    e_convergence       10
    d_convergence       10
    docc                [4,0,1,1]
    restricted_docc     [3,0,1,1]
    active              [2,0,0,0]
    mcscf_r_convergence 7
    mcscf_e_convergence 10
    mcscf_maxiter       250
    mcscf_diis_start    25
    num_roots           2
    avg_states          [0,1]
  }

  set forte{
    active_space_solver cas
    correlation_solver  dsrg-mrpt2
    frozen_docc        [2,0,0,0]
    restricted_docc    [1,0,0,0]
    active             [3,0,2,2]
    dsrg_s             0.5
    avg_state          [[0,1,2]]
    dsrg_multi_state   sa_full
    calc_type          sa
  }

  Emcscf, wfn = energy('casscf', return_wfn=True)
  energy('forte',ref_wfn=wfn)

Here, we explicitly specify the cc-pVDZ basis set of Li since Psi4 uses seg-opt basis (at least at some time).
For simplicity, we do an SA-CASSCF(2,2) computation in Psi4 but the active space in Forte is CASCI(8e,7o),
which should be clearly stated in the publication if this kind of special procedure is used.

To perform an SA-DSRG-PT2 computation, the following keywords should be specified
(besides those already mentioned in the state-specific DSRG-MRPT2):

- :code:`CALC_TYPE`:
  The type of computation should be set to state averaging, i.e., SA.
  Multi-state and dynamically weighted computations should be set correspondingly.

- :code:`AVG_STATE`:
  This specifies the states to be averaged, given in arrays of triplets [[A1, B1, C1], [A2, B2, C2], ...].
  Each triplet corresponds to the *state irrep*, *state multiplicity*, and the *nubmer of states*, in sequence.
  The number of states are counted from the lowest energy one in the given symmetry.

- :code:`DSRG_MULTI_STATE`:
  This options specifies the methods used in DSRG computations.
  By default, it will use :code:`SA_FULL`.

The output of this example will print out the CASCI(8e,7o) configurations ::

  ==> Root No. 0 <==

    ba0 20 20         -0.6992227471
    ab0 20 20         -0.6992227471
    200 20 20         -0.1460769052

    Total Energy:   -106.772573855919561


  ==> Root No. 1 <==

    200 20 20          0.9609078151
    b0a 20 20          0.1530225853
    a0b 20 20          0.1530225853
    ba0 20 20         -0.1034194675
    ab0 20 20         -0.1034194675

    Total Energy:   -106.735798144523812

Then the 1-, 2-, and 3-RDMs for each state are computed and then sent to orbital canonicalizer.
The DSRG-PT2 computation will still print out the energy contributions,
which now correspond to the corrections to the average of the ensemble. ::

  ==> DSRG-MRPT2 Energy Summary <==

    E0 (reference)                 =   -106.754186000221665
    <[F, T1]>                      =     -0.000345301150943
    <[F, T2]>                      =      0.000293904835970
    <[V, T1]>                      =      0.000300892512596
    <[V, T2]> (C_2)^4              =     -0.246574892923286
    <[V, T2]> C_4 (C_2)^2 HH       =      0.000911300780649
    <[V, T2]> C_4 (C_2)^2 PP       =      0.002971830422787
    <[V, T2]> C_4 (C_2)^2 PH       =      0.010722949661906
    <[V, T2]> C_6 C_2              =      0.000099208259233
    <[V, T2]>                      =     -0.231869603798710
    DSRG-MRPT2 correlation energy  =     -0.231620107601087
    DSRG-MRPT2 total energy        =   -106.985806107822754

Finally, a CASCI is performed using DSRG-PT2 dressed integrals. ::

  ==> Root No. 0 <==

    200 20 20          0.8017660337
    ba0 20 20          0.4169816393
    ab0 20 20          0.4169816393

    Total Energy:   -106.990992362637314


  ==> Root No. 1 <==

    200 20 20         -0.5846182713
    ba0 20 20          0.5708699624
    ab0 20 20          0.5708699624

    Total Energy:   -106.981903302649229

Here we observe the ordering of states changes by comparing the configurations.
In fact, it is near the avoided crossing region and we see the CI coefficients
between these two states are very similar (comparing to the original CASCI coefficients).
An automatic way to correspond states before and after DSRG treatments for dynamical correlation is not implemented.
A simple approach is to compute the overlap, which should usually suffice.

At the end, we print the energy summary of the states of interest. ::

  ==> Energy Summary <==

    Multi.  Irrep.  No.               Energy
    -----------------------------------------
       1      A1     0      -106.990992362637
       1      A1     1      -106.981903302649
    -----------------------------------------

.. tip::
  It is sometimes cumbersome to grab the energies of all the computed states from
  the output file, especially when multiple reference relaxation steps are performed.
  Here, one could use the keyword **DSRG_DUMP_RELAXED_ENERGIES** where a JSON file
  :code:`dsrg_relaxed_energies.json` is created.
  In the above example, the file will read ::

      {
          "0": {
              "ENERGY ROOT 0 1A1": -106.7725738559195,
              "ENERGY ROOT 1 1A1": -106.7357981445238
          },
          "1": {
              "DSRG FIXED": -106.98580610782275,
              "DSRG RELAXED": -106.98644783264328,
              "ENERGY ROOT 0 1A1": -106.99099236263731,
              "ENERGY ROOT 1 1A1": -106.98190330264923
          }
      }

The printing for SA-DSRG-PT2c (set :code:`DSRG_MULTI_STATE` to :code:`SA_SUB`) is slightly different from above.
After the DSRG-PT2 computation, we build the effective Hamiltonian using the original CASCI states. ::

  ==> Building Effective Hamiltonian for Singlet A1 <==

  Computing  1RDMs (0 Singlet A1 - 0 Singlet A1) ... Done. Timing        0.001090 s
  Computing  2RDMs (0 Singlet A1 - 0 Singlet A1) ... Done. Timing        0.001884 s
  Computing 1TrDMs (0 Singlet A1 - 1 Singlet A1) ... Done. Timing        0.001528 s
  Computing 2TrDMs (0 Singlet A1 - 1 Singlet A1) ... Done. Timing        0.002151 s
  Computing  1RDMs (1 Singlet A1 - 1 Singlet A1) ... Done. Timing        0.001114 s
  Computing  2RDMs (1 Singlet A1 - 1 Singlet A1) ... Done. Timing        0.001757 s

  ==> Effective Hamiltonian for Singlet A1 <==

  ## Heff Singlet A1 (Symmetry 0) ##
  Irrep: 1 Size: 2 x 2

                 1                   2

    1  -106.98637816344888     0.00443421124030
    2     0.00443421124030  -106.98523405219674

  ## Eigen Vectors of Heff for Singlet A1 with eigenvalues ##

           1           2

    1  -0.7509824  -0.6603222
    2   0.6603222  -0.7509824

     -106.9902771-106.9813351

Here, we see a strong coupling between the two states at this geometry:
The SA-DSRG-PT2c ground state is :math:`0.75 |\Phi_1\rangle - 0.66 |\Phi2\rangle`.

5. Related Options
++++++++++++++++++

**DSRG_MULTI_STATE**

Algorithms to compute excited states.

* Type: string
* Options: SA_FULL, SA_SUB, MS, XMS
* Default: SA_FULL

**DWMS_ZETA**

Automatic Gaussian width cutoff for the density weights.

* Type: double
* Default: 0.0

.. note:: Add options when DWMS is re-enabled.

Frozen-Natural-Orbital Truncated MR-DSRG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Theory
+++++++++

The MRPT3 and LDSRG(2) computations can be accelerated by constructing a
compact set of virtual orbitals based on the quasi-natural orbitals of DSRG-MRPT2.
The natural occupations smaller than the user-defined threshold will be discarded
for MRPT3 or LDSRG(2) computations.
A second-order correction to the discarded virtual orbitals are considered by default,
but this correction can be disabled.

The FNO MR-DSRG procedure add the following additional steps before a regular MR-DSRG computation:

  (1) Build natural virtual orbitals by diagonalizing the virtual-virtual block of
  the unrelaxed DSRG-MRPT2 one-particle reduced density matrix.
  Integrals are subsequently updated to the natural-virtual-orbital basis.

  (2) Read natural virtual occupations from :code:`NAT_OCC_VIRT` file.
  Throw away virtual orbitals whose natural occupations are smaller than the
  user-defined threshold.

  (3) Compute the MRPT2 corrections due to FNO truncation.
  Perform two MRPT2 computations: one with complete virtual orbitals and
  the other with FNO truncated virtual orbitals.

.. note::
  - If the :code:`NAT_OCC_VIRT` file is available in the same directory as the input file,
    we assume the orbitals are already in DSRG-MRPT2 quasi-natural-orbital basis such that
    step (1) is skipped.

  - In step (1), the DSRG-MRPT2 ccvv amplitudes have the same expressions to those of MP2.
    This behavior is hard coded in :code:`proc/dsrg_fno.py` by the option :code:`CCVV_SOURCE`.

  - If option :code:`THREEPDC` is set to :code:`ZERO`, 3-RDM will also be ignored for the
    1-RDM build of DSRG-MRPT2.
    The resulting quasi-natural orbitals are thus approximated, but the error is negligible.
    This behavior can be changed by modifying :code:`max_rdm_level` in :code:`proc/dsrg_fno.py`.

  - Because the recommended flow parameter is different between MRPT2 and others,
    the flow parameter for MRPT2 related steps [i.e., (1) and (3)] is controlled by option
    :code:`DSRG_FNO_PT2_S`.
    The default value of :code:`DSRG_FNO_PT2_S` is 0.5.

2. Examples
+++++++++++

The following is an example of FNO SA-DSRG-PT3 to compute the vertical excitation energy of
acetaldehyde from ground state to the first singlet A'' state (test case "fno-2"). ::

  import forte
  memory 4 gb
  molecule acetaldehyde{
  C -0.00234503  0.00000000  0.87125063
  C -1.75847785  0.00000000 -1.34973671
  O  2.27947397  0.00000000  0.71968028
  H -0.92904537  0.00000000  2.73929404
  H -2.97955463  1.66046488 -1.25209463
  H -2.97955463 -1.66046488 -1.25209463
  H -0.70043433  0.00000000 -3.11066412
  units bohr
  nocom
  noreorient
  }

  set globals{
  scf_type      df
  reference     rhf
  basis         aug-cc-pvtz
  df_basis_scf  aug-cc-pvtz-jkfit
  df_basis_mp2  aug-cc-pvtz-jkfit
  maxiter       100
  d_convergence 1.0e-6
  e_convergence 1.0e-8
  }
  escf, wfn = energy('scf', return_wfn=True)
  compare_values(ref_escf, escf, 7, "SCF energy")
  wfn_cas = wfn.from_file("../fno-1/wfn_casscf.npy")
  wfn.Ca().copy(wfn_cas.Ca())

  set forte{
  int_type            df
  active_space_solver fci
  correlation_solver  sa-mrdsrg
  corr_level          pt3
  frozen_docc         [3,0]
  restricted_docc     [5,1]
  active              [3,2]
  avg_state           [[0,1,1],[1,1,1]]
  dsrg_s              2.0
  calc_type           sa
  dl_maxiter          500
  threepdc            zero
  dsrg_fno            true
  dsrg_fno_cutoff     1.0e-4
  dsrg_fno_pt2_s      0.5
  }
  energy('forte', ref_wfn=wfn)

Here, we read converged SA-CASSCF orbitals from test case "fno-1".
The FNO procedure is activated by :code:`dsrg_fno` and the occupation truncation
is managed by the option :code:`dsrg_fno_cutoff`.
Using 1.0e-4 as the FNO cutoff, 82 (out of 134) A' and 49 (out of 82) A'' orbitals are discarded.

.. tip::
  When computing vertical transition energies using SA-DSRG-PT2, -PT3, or LDSRG(2),
  there is no need to compute the SA-3RDM.

The PT2 corrected FNO SA-DSRG-PT3 energies are ::

  Multi.(2ms)  Irrep.  No.               Energy      <S^2>
  --------------------------------------------------------
     1  (  0)    Ap     0     -153.578330831999   0.000000
  --------------------------------------------------------
     1  (  0)   App     0     -153.415039030319   0.000000
  --------------------------------------------------------

For comparison, the PT2 uncorrected (by setting :code:`DSRG_FNO_PT2_CORRECTION` to :code:`False`)
FNO SA-DSRG-PT3 energies read as ::

  Multi.(2ms)  Irrep.  No.               Energy      <S^2>
  --------------------------------------------------------
     1  (  0)    Ap     0     -153.561250184637   0.000000
  --------------------------------------------------------
     1  (  0)   App     0     -153.398282723341  -0.000000
  --------------------------------------------------------

and the untruncated SA-DSRG-PT3 gives ::

  Multi.(2ms)  Irrep.  No.               Energy      <S^2>
  --------------------------------------------------------
     1  (  0)    Ap     0     -153.576522878154   0.000000
  --------------------------------------------------------
     1  (  0)   App     0     -153.413287040548   0.000000
  --------------------------------------------------------

The resulting SA-DSRG-PT3 vertical excitation energies are:

============================================  =====================
Method                                         :math:`\Delta E` / eV
============================================  =====================
FNO (PT2 uncorrected, w/o :math:`\lambda_3`)         4.4346
FNO (PT2 uncorrected, w/  :math:`\lambda_3`)         4.4336
FNO (PT2 corrected, w/o :math:`\lambda_3`)           4.4434
FNO (PT2 corrected, w/  :math:`\lambda_3`)           4.4425
untruncated                                          4.4418
============================================  =====================

where the PT2 corrected FNO result is in excellent agreement with that of the complete SA-DSRG-PT3.

3. Related Options
++++++++++++++++++

**DSRG_FNO**

Perform frozen-natural-orbital truncated MR-DSRG based on DSRG-MRPT2 unrelaxed 1-RDM.

* Type: Boolean
* Default: False

**DSRG_FNO_PT2_CORRECTION**

Perform PT2 corrections to the discarded natural virtual orbitals.

* Type: Boolean
* Default: True

**DSRG_FNO_CUTOFF**

The virtual orbitals with natural occupations smaller than this cutoff will be discarded.

* Type: double
* Default: 1.0e-5

**DSRG_FNO_PT2_S**

Flow parameter for DSRG-MRPT2 related steps in the FNO procedure.

* Type: double
* Default: 0.5

TODOs
^^^^^

0. Re-enable MS, XMS, and DWMS
++++++++++++++++++++++++++++++

These are disabled due to an infrastructure change.

1. DSRG-MRPT2 Analytic Energy Gradients
+++++++++++++++++++++++++++++++++++++++

This is an ongoing project.

2. MR-DSRG(T) with Perturbative Triples
+++++++++++++++++++++++++++++++++++++++

This is an ongoing project.

.. _`dsrg_example`:

A Complete List of DSRG Teset Cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Acronyms used in the following text:

* Integrals

  DF: density fitting;
  DiskDF: density fitting (disk algorithm);
  CD: Cholesky decomposition;

* Reference Relaxation

  U: unrelaxed;
  PR: partially relaxed;
  R: relaxed;
  FR: fully relaxed;

* Single-State / Multi-State

  SS: state-specific;
  SA: state-averaged;
  SAc: state-averaged with constrained reference;
  MS: multi-state;
  XMS: extended multi-state;
  DWMS: dynamically weighted multi-state;

* Theoretical Variants

  QC: commutator truncated to doubly nested level (i.e., :math:`\bar{H} = \hat{H} + [\hat{H}, \hat{A}] + \frac{1}{2} [[\hat{H}, \hat{A}], \hat{A}]`);
  SQ: sequential transformation;
  NIVO: non-interacting virtual orbital approximation;

* Run Time:

  long: > 30 s to finish;
  Long: > 5 min to finish;
  LONG: > 20 min to finish;

1. DSRG-MRPT2 Test Cases
++++++++++++++++++++++++

  ============================  =========  ============================================  =================================================
              Name               Variant     Molecule                                      Notes
  ============================  =========  ============================================  =================================================
  dsrg-mrpt2-1                   SS, U     :math:`\text{BeH}_{2}`                        large :math:`s` value, user defined basis set
  dsrg-mrpt2-2                   SS, U     :math:`\text{HF}`
  dsrg-mrpt2-3                   SS, U     :math:`\text{H}_4` (rectangular)
  dsrg-mrpt2-4                   SS, U     :math:`\text{N}_2`
  dsrg-mrpt2-5                   SS, U     benzyne :math:`\text{C}_6 \text{H}_4`
  dsrg-mrpt2-6                   SS, PR    :math:`\text{N}_2`
  dsrg-mrpt2-7-casscf-natorbs    SS, PR    :math:`\text{N}_2`                            CASSCF natural orbitals
  dsrg-mrpt2-8-sa                SA, SAc   :math:`\text{LiF}`                            lowest two singlet states, user defined basis set
  dsrg-mrpt2-9-xms               MS, XMS   :math:`\text{LiF}`                            lowest two singlet states
  dsrg-mrpt2-10-CO               SS, PR    :math:`\text{CO}`                             dipole moment (not linear response)
  dsrg-mrpt2-11-C2H4             SA        ethylene :math:`\text{C}_2\text{H}_4`         lowest three singlet states
  dsrg-mrpt2-12-localized-actv   SA        butadiene :math:`\text{C}_4\text{H}_6`        long, localized active orbitals
  dsrg-mrpt2-13                  SS        :math:`\text{N}_2` and N atom                 size-consistency check
  aci-dsrg-mrpt2-1               SS, U     :math:`\text{N}_2`                            ACI(:math:`\sigma=0`)
  aci-dsrg-mrpt2-2               SS, U     :math:`\text{H}_4` (rectangular)              ACI(:math:`\sigma=0`)
  aci-dsrg-mrpt2-3               SS, PR    :math:`\text{H}_4` (rectangular)              ACI(:math:`\sigma=0`)
  aci-dsrg-mrpt2-4               SS, U     octatetraene :math:`\text{C}_8\text{H}_{10}`  DF, ACI(:math:`\sigma=0.001`), ACI batching
  aci-dsrg-mrpt2-5               SS, PR    octatetraene :math:`\text{C}_8\text{H}_{10}`  long, DF, ACI(:math:`\sigma=0.001`), ACI batching
  ============================  =========  ============================================  =================================================

2. DF/CD-DSRG-MRPT2 Test Cases
++++++++++++++++++++++++++++++

  ================================  =========  ============================================  =================================================
              Name                   Variant     Molecule                                      Notes
  ================================  =========  ============================================  =================================================
  cd-dsrg-mrpt2-1                    SS, U      :math:`\text{BeH}_{2}`                        CD(:math:`\sigma=10^{-14}`)
  cd-dsrg-mrpt2-2                    SS, U      :math:`\text{HF}`                             CD(:math:`\sigma=10^{-14}`)
  cd-dsrg-mrpt2-3                    SS, U      :math:`\text{H}_4` (rectangular)              CD(:math:`\sigma=10^{-14}`)
  cd-dsrg-mrpt2-4                    SS, U      :math:`\text{N}_2`                            CD(:math:`\sigma=10^{-12}`)
  cd-dsrg-mrpt2-5                    SS, U      benzyne :math:`\text{C}_6 \text{H}_4`         CD(:math:`\sigma=10^{-11}`)
  cd-dsrg-mrpt2-6                    SS, PR     :math:`\text{BeH}_{2}`                        CD(:math:`\sigma=10^{-14}`)
  cd-dsrg-mrpt2-7-sa                 SA         :math:`\text{LiF}`                            CD(:math:`\sigma=10^{-14}`)
  df-dsrg-mrpt2-1                    SS, U      :math:`\text{BeH}_{2}`
  df-dsrg-mrpt2-2                    SS, U      :math:`\text{HF}`
  df-dsrg-mrpt2-3                    SS, U      :math:`\text{H}_4` (rectangular)
  df-dsrg-mrpt2-4                    SS, U      :math:`\text{N}_2`
  df-dsrg-mrpt2-5                    SS, U      benzyne :math:`\text{C}_6 \text{H}_4`
  df-dsrg-mrpt2-6                    SS, PR     :math:`\text{N}_2`
  df-dsrg-mrpt2-7-localized-actv     SA         butadiene :math:`\text{C}_4\text{H}_6`        long, localized active orbitals
  df-dsrg-mrpt2-threading1           SS, U      benzyne :math:`\text{C}_6 \text{H}_4`
  df-dsrg-mrpt2-threading2           SS, U      benzyne :math:`\text{C}_6 \text{H}_4`
  df-dsrg-mrpt2-threading4           SS, U      benzyne :math:`\text{C}_6 \text{H}_4`
  diskdf-dsrg-mrpt2-1                SS, U      :math:`\text{BeH}_{2}`
  diskdf-dsrg-mrpt2-2                SS, U      :math:`\text{HF}`
  diskdf-dsrg-mrpt2-3                SS, U      :math:`\text{H}_4` (rectangular)
  diskdf-dsrg-mrpt2-4                SS, PR     :math:`\text{N}_2`
  diskdf-dsrg-mrpt2-5                SS, U      benzyne :math:`\text{C}_6 \text{H}_4`
  diskdf-dsrg-mrpt2-threading1       SS, U      benzyne :math:`\text{C}_6 \text{H}_4`
  diskdf-dsrg-mrpt2-threading4       SS, U      benzyne :math:`\text{C}_6 \text{H}_4`
  df-aci-dsrg-mrpt2-1                SS, U      benzyne :math:`\text{C}_6 \text{H}_4`         ACI(:math:`\sigma=0`)
  df-aci-dsrg-mrpt2-2                SS, U      :math:`\text{HF}`                             ACI(:math:`\sigma=0.0001`)
  ================================  =========  ============================================  =================================================

3. DSRG-MRPT3 Test Cases
++++++++++++++++++++++++

  ============================  =========  ============================================  =================================================
              Name               Variant     Molecule                                      Notes
  ============================  =========  ============================================  =================================================
   dsrg-mrpt3-1                  SS, PR     :math:`\text{HF}`
   dsrg-mrpt3-2                  SS, PR     :math:`\text{HF}`                             CD(:math:`\sigma=10^{-8}`)
   dsrg-mrpt3-3                  SS, PR     :math:`\text{N}_2`                            CD(:math:`\sigma=10^{-8}`), long, time printing
   dsrg-mrpt3-4                  SS, PR     :math:`\text{N}_2`
   dsrg-mrpt3-5                  SA         :math:`\text{LiF}`                            CAS(2e,2o), default cc-pVDZ of Li is seg-opt
   dsrg-mrpt3-6-sa               SA         :math:`\text{LiF}`                            CAS(8e,7o), user defined cc-pVDZ for Li
   dsrg-mrpt3-7-CO               SS, PR     :math:`\text{CO}`                             dipole moment (not linear response)
   dsrg-mrpt3-8-sa-C2H4          SA         ethylene :math:`\text{C}_2\text{H}_4`         long, lowest three singlet states
   dsrg-mrpt3-9                  SS, PR     :math:`\text{HF}`                             CD(:math:`\sigma=10^{-14}`), batching
   aci-dsrg-mrpt3-1              SS, PR     :math:`\text{N}_2`                            ACI(:math:`\sigma=0`)
  ============================  =========  ============================================  =================================================

4. MR-DSRG Test Cases
+++++++++++++++++++++

  =================================  =======================  ============================================  =================================================
              Name                           Variant            Molecule                                      Notes
  =================================  =======================  ============================================  =================================================
  mrdsrg-pt2-1                        SS, U                    :math:`\text{BeH}_{2}`                        PT2
  mrdsrg-pt2-2                        SS, PR                   :math:`\text{BeH}_{2}`                        PT2
  mrdsrg-pt2-3                        SS, FR                   :math:`\text{BeH}_{2}`                        long, PT2
  mrdsrg-pt2-4                        SS, FR                   :math:`\text{HF}`                             PT2
  mrdsrg-pt2-5                        SS, R                    :math:`\text{HF}`                             long, PT2, DIIS, 0th-order Hamiltonian
  mrdsrg-srgpt2-1                     SS, U                    :math:`\text{BeH}_{2}`                        Long, SRG_PT2
  mrdsrg-srgpt2-2                     SS, U                    :math:`\text{BeH}_{2}`                        LONG, SRG_PT2, Dyall Hamiltonian
  mrdsrg-ldsrg2-1                     SS, U                    :math:`\text{N}_{2}`                          long, read amplitudes
  mrdsrg-ldsrg2-df-1                  SS, R                    :math:`\text{BeH}_{2}`                        CD, long
  mrdsrg-ldsrg2-df-2                  SS, R                    :math:`\text{HF}`                             CD, long
  mrdsrg-ldsrg2-df-3                  SS, U                    :math:`\text{H}_4` (rectangular)              CD, long
  mrdsrg-ldsrg2-df-4                  SS, PR                   :math:`\text{H}_{2}`                          CD
  mrdsrg-ldsrg2-df-seq-1              SS, PR, SQ               :math:`\text{BeH}_{2}`                        CD, Long
  mrdsrg-ldsrg2-df-seq-2              SS, R, SQ                :math:`\text{HF}`                             CD, Long
  mrdsrg-ldsrg2-df-seq-3              SS, U, SQ                :math:`\text{H}_4` (rectangular)              CD, long
  mrdsrg-ldsrg2-df-seq-4              SS, FR, SQ               :math:`\text{H}_4` (rectangular)              CD, Long
  mrdsrg-ldsrg2-df-nivo-1             SS, PR, NIVO             :math:`\text{BeH}_{2}`                        CD, long
  mrdsrg-ldsrg2-df-nivo-2             SS, R, NIVO              :math:`\text{HF}`                             CD, long
  mrdsrg-ldsrg2-df-nivo-3             SS, U, NIVO              :math:`\text{H}_4` (rectangular)              CD, long
  mrdsrg-ldsrg2-df-seq-nivo-1         SS, PR, SQ, NIVO         :math:`\text{BeH}_{2}`                        CD, long
  mrdsrg-ldsrg2-df-seq-nivo-2         SS, R, SQ, NIVO          :math:`\text{HF}`                             CD, Long
  mrdsrg-ldsrg2-df-seq-nivo-3         SS, U, SQ, NIVO          :math:`\text{H}_4` (rectangular)              CD, long
  mrdsrg-ldsrg2-qc-1                  SS, FR, QC               :math:`\text{HF}`                             long
  mrdsrg-ldsrg2-qc-2                  SS, U, QC                :math:`\text{HF}`                             long
  mrdsrg-ldsrg2-qc-df-2               SS, U, QC                :math:`\text{HF}`                             CD, long
  =================================  =======================  ============================================  =================================================

5. DWMS-DSRG-PT2 Test Cases
+++++++++++++++++++++++++++

Add test cases when DWMS is back to life.

6. Spin-Adapted MR-DSRG Test Cases
++++++++++++++++++++++++++++++++++

  ============================  ==================  ===========================  =================================================
              Name              Variants            Molecule                     Notes
  ============================  ==================  ===========================  =================================================
  mrdsrg-spin-adapted-1         SS, U               :math:`\text{HF}`            LDSRG(2) truncated to 2-nested commutator
  mrdsrg-spin-adapted-2         SS, PR              :math:`\text{HF}`            long, LDSRG(2), non-semicanonical orbitals
  mrdsrg-spin-adapted-3         SS, R, SQ, NIVO     :math:`\text{HF}`            long, CD, LDSRG(2)
  mrdsrg-spin-adapted-4         SS, U               :math:`\text{N}_2`           long, CD, LDSRG(2), non-semicanonical, zero ccvv
  mrdsrg-spin-adapted-5         SS, U               :math:`\text{N}_2`           long, read/dump amplitudes
  mrdsrg-spin-adapted-6         SA                  benzene                      long
  mrdsrg-spin-adapted-7         SA                  ethylene                     short, read/dump amplitudes, multipole integrals
  mrdsrg-spin-adapted-pt2-1     SS, U               :math:`\text{HF}`            CD
  mrdsrg-spin-adapted-pt2-2     SS, U               :math:`\text{HF}`            CD, non-semicanonical orbitals, zero ccvv source
  mrdsrg-spin-adapted-pt2-3     SS, PR              p-benzyne                    DiskDF
  mrdsrg-spin-adapted-pt2-4     SS, R               :math:`\text{O}_2`           triplet ground state, CASSCF(8e,6o)
  mrdsrg-spin-adapted-pt2-5     SA, R               :math:`\text{C}_2`           CASSCF(8e,8o), zero 3 cumulant
  mrdsrg-spin-adapted-pt2-6     SA                  benzene                      Exotic state-average weights
  mrdsrg-spin-adapted-pt2-7     SA                  ethylene                     general orbitals, dipole level 2
  mrdsrg-spin-adapted-pt3-1     SS, PR              :math:`\text{HF}`            CD
  mrdsrg-spin-adapted-pt3-2     SA                  ethylene                     lowest three singlet states
  ============================  ==================  ===========================  =================================================

.. _`dsrg_ref`:

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

    The spin-adapted implementation of the above MR-DSRG methods is reported in:

    * "Spin-free formulation of the multireference driven similarity renormalization group:
      A benchmark study of first-row diatomic molecules and spin-crossover energetics", C. Li
      and F. A. Evangelista, *J. Chem. Phys.* **155**, 114111 (2021).
      (doi: `10.1063/5.0059362 <http://dx.doi.org/10.1063/5.0059362>`_).

The DSRG extensions for excited state are discussed in the following articles:

    SA-DSRG framework and its PT2 and PT3 applications:

    * "Driven similarity renormalization group for excited states:
      A state-averaged perturbation theory", C. Li and F. A. Evangelista,
      *J. Chem. Phys.* **148**, 124106 (2018).
      (doi: `10.1063/1.5019793 <http://dx.doi.org/10.1063/1.5019793>`_).

    SA-DSRG benchmarks

    * "Assessment of State-Averaged Driven Similarity Renormalization Group on Vertical
      Excitation Energies: Optimal Flow Parameters and Applications to Nucleobases",
      M. Wang, W.-H. Fang, and C. Li, *J. Chem. Theory Comput.* **19**, 122-136 (2023).
      (doi: `10.1021/acs.jctc.2c00966 <http://dx.doi.org/10.1021/acs.jctc.2c00966>`_).

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

    Multireference DSRG-MRPT2:

    * "Analytic Energy Gradients for the Driven Similarity Renormalization Group
      Multireference Second-Order Perturbation Theory", S. Wang, C. Li, and F. A. Evangelista,
      *J. Chem. Theory Comput.* **17**, 7666-7681 (2021).
      (doi: `10.1021/acs.jctc.1c00980 <http://dx.doi.org/10.1021/acs.jctc.1c00980>`_).

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
