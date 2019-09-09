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

.. _`basic_dsrg`:

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

.. _`basic_dsrg_example`:

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

1. The :code:`molecule` block specifies the geometry, charge, multiplicity, etc.

2. The second block specifies Psi4 options (see Psi4 manual for details).

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
    By default, :code:`DSRG_S` is set to :math:`10^{10}` a.u.
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
    if we further change :code:`DSRG_S` to very large values (e.g., :math:`10^8` a.u.).

The output of the above example consists of several parts:

* Perform a active-space computation: ::

    ==> Root No. 0 <==

      20     -0.95086442
      02      0.29288371

      Total Energy:       -99.939316382616340

    ==> Energy Summary <==

      Multi.  Irrep.  No.               Energy
      -----------------------------------------
         1      A1     0       -99.939316382616
      -----------------------------------------

  Here we print out the CASCI configurations and its energy.
  Since we read orbitals from Psi4's CASSCF, this energy should coincide with Psi4's CASSCF energy.

* Compute 1-, 2-, and 3-body reduced density matrices (RDMs): ::

    ==> Computing RDMs for Root No. 0 <==

      Timing for 1-RDM: 0.000 s
      Timing for 2-RDM: 0.000 s
      Timing for 3-RDM: 0.000 s

* Canonicalize orbitals: ::

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

  Since Psi4's CASSCF will canonicalize orbitals at the end, here Forte just tests the Fock matrix
  but does not perform an actual orbital rotation.

* Compute DSRG-MRPT2 energy:

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
   involves 2-body density cumulants, and those of C_6 are of 3-body cumnulants.


**A More Advanced Example**

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
  This example takes a long time to finish (~30 min on a laptop using 8 threads).

There are several things to notice.

1. To run a MR-LDSRG(2) computation, we need to change :code:`CORRELATION_SOLVER` to :code:`MRDSRG`.
   Additionally, the :code:`CORR_LEVEL` should be specified as :code:`LDSRG2`.
   There are other choices of :code:`CORR_LEVEL` but they are mainly for testing ideas.

2. We specify the energy convergence keyword :code:`E_CONVERGENCE` and the RSC threshold :code:`DSRG_RSC_THRESHOLD`.
   In general, the value of :code:`DSRG_RSC_THRESHOLD` should be smaller than that of :code:`E_CONVERGENCE`.
   Making :code:`DSRG_RSC_THRESHOLD` larger will stop the BCH series earlier and thus saves some time.
   It is OK to leave :code:`DSRG_RSC_THRESHOLD` as the default value, which is :math:`10^{-12}` a.u.

3. The MR-LDSRG(2) method includes reference relaxation effects.
   There are several variants of reference relaxation levels (see :ref:`dsrg_variants`).
   Here we use the fully relaxed version, which is done by setting :code:`RELAX_REF` to :code:`ITERATE`.

.. note::
  The reference relaxation procedure is performed in a tick-tock way (see :ref:`dsrg_variants`).
  This procedure is potentially not numerically stable for a strict energy convergence.
  We therefore suggest using a moderate the energy threshold for iterative reference relaxation,
  which is controlled by :code:`RELAX_E_CONVERGENCE` (:math:`\geq 10^{-8}` a.u.).

For a given reference wave function, the output prints out:

1. The iterations of amplitudes, where each step involves building a DSRG transformed Hamiltonian.

2. A summary of the MR-LDSRG(2) energy: ::

    ==> MR-LDSRG(2) Energy Summary <==

      E0 (reference)                 =     -99.939316382616383
      MR-LDSRG(2) correlation energy =      -0.171613035562048
      MR-LDSRG(2) total energy       =    -100.110929418178429

3. A summary of the MR-LDSRG(2) converged amplitudes: ::

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

4. The reference relaxation summary at the end: ::

    => MRDSRG Reference Relaxation Energy Summary <=

                           Fixed Ref. (a.u.)              Relaxed Ref. (a.u.)
             -------------------------------  -------------------------------
      Iter.          Total Energy      Delta          Total Energy      Delta
      -----------------------------------------------------------------------
          1     -100.110929418178 -1.001e+02     -100.114343552853 -1.001e+02
          2     -100.113565563124 -2.636e-03     -100.113571036112  7.725e-04
          3     -100.113534597590  3.097e-05     -100.113534603824  3.643e-05
          4     -100.113533334887  1.263e-06     -100.113533334895  1.269e-06
          5     -100.113533290863  4.402e-08     -100.113533290864  4.403e-08
          6     -100.113533289341  1.522e-09     -100.113533289341  1.522e-09
      -----------------------------------------------------------------------

   Let us introduce the nomenclature for reference relaxation.

   =================  =========================  ========================
          Name              Example Value               Description
   =================  =========================  ========================
   Unrelaxed          :code:`-100.110929418178`  1st iter.; fixed ref.
   Partially Relaxed  :code:`-100.114343552853`  1st iter.; relaxed ref.
   Relaxed            :code:`-100.113565563124`  2nd iter.; fixed ref.
   Fully Relaxed      :code:`-100.113533289341`  last iter.; relaxed ref.
   =================  =========================  ========================

   In the example, and usually, the fully relaxed energy is well reproduced by
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
* Default: 1.0e10

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
  The avaible keys are :code:`"UNRELAXED ENERGY"`, :code:`PARTIALLY RELAXED ENERGY`,
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

In sequential transformation ansatz, we compute :math:`\bar{H}` sequentially as

.. math:: \bar{H}(s) = e^{-\hat{A}_n(s)} \cdots e^{-\hat{A}_2(s)} e^{-\hat{A}_1(s)} \hat{H} e^{\hat{A}_1(s)} e^{\hat{A}_2(s)} \cdots e^{\hat{A}_n(s)}

instead of traditionally

.. math:: \bar{H}(s) = e^{-\hat{A}_1(s)-\hat{A}_2(s) - \cdots - \hat{A}_n(s)} \hat{H} e^{\hat{A}_1(s)+\hat{A}_2(s)+\cdots+\hat{A}_n(s)}

In the limit of :math:`s \rightarrow \infty` and no truncation of :math:`\hat{A}(s)`, both the traditional and sequential MR-DSRG can approach the full configuration interaction limit. The difference between their truncated results are also usually small.

Computationally, sequential transformation simplifies the evaluation of one-body contribution as a unitary transformation rather than conventional BCH expansion. If combined with integral factorization, the unitary transformation is further accelerated (scaling reduction).

4. Non-Interacting Virtual Orbital Approximation
++++++++++++++++++++++++++++++++++++++++++++++++

In the non-interacting virtual orbital (NIVO) approximation, we neglect the operator components of all rank-4 intermediate tensors and :math:`\bar{H}` with three or more virtual orbital indices (:math:`\mathbf{VVVV}`, :math:`\mathbf{VCVV}`, :math:`\mathbf{VVVA}`, etc.).

Removing these blocks, the number of elements in each NIVO-approximated tensor is reduced from :math:`{\cal O}(N^4)` to :math:`{\cal O}(N^2N_\mathbf{H}^2)`, a size comparable to that of the $\hat{A}_2(s)$ tensor.
Thus, the memory scaling of :code:`LDSRG(2)` can be reduced to be lower than :math:`\mathcal{O}(N^4)`, where :math:`N` is the number of correlated orbitals, when we apply NIVO approximation and combine with integral factorization and batched algorithm of tensor contraction.

Since much less number of tensor elements are involved, NIVO approximation dramatically reduces computation time. However, the overall time scaling of :code:`LDSRG(2)` remain unchanged (prefector reduction).

Despite a significant reduction on the tensor size, the error introduced is usually negligible.

5. Examples
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

6. Related Options
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

**SEMI_CANONICAL**

Semicanonicalize orbitals after solving the active-space eigenvalue problem.

* Type: boolean
* Default: True

**DSRG_HBAR_SEQ**

Apply the sequential transformation algorithm in evaluating the transformed Hamiltonian :math:`\bar{H}(s)`, i.e.,

.. math:: \bar{H}(s) = e^{-\hat{A}_n(s)} \cdots e^{-\hat{A}_2(s)} e^{-\hat{A}_1(s)} \hat{H} e^{\hat{A}_1(s)} e^{\hat{A}_2(s)} \cdots e^{\hat{A}_n(s)}.

* Type: boolean
* Default: false

**DSRG_NIVO**

Apply non-interacting virtual orbital (NIVO) approximation in evaluating the transformed Hamiltonian.

* Type: boolean
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

2. Multi-State and Extended Formalisms
++++++++++++++++++++++++++++++++++++++

3. Dynamically Weighted Formalism
+++++++++++++++++++++++++++++++++

4. Examples
+++++++++++

5. Related Options
++++++++++++++++++

TODOs
^^^^^

1. Spin Adaptation
++++++++++++++++++

This is done for unrelaxed DSRG-MRPT2 but not complete for general LDSRG(2).

2. DSRG-MRPT2 Analytic Energy Gradients
+++++++++++++++++++++++++++++++++++++++

This is an ongoing project.

3. MR-DSRG(T) with Perturbative Triples
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

* Misc.
  
  QC: quadratic convergence;
  SQ: sequential transformation;
  NIVO: non-interacting virtual orbital approximation;

* Run Time:

  long: more than 30 s to finish;
  Long: more than 5 min to finish;
  LONG: more than 20 min to finish;

1. DSRG-MRPT2 Test Cases
++++++++++++++++++++++++

  ============================  =========  ============================================  =================================================
              Name               Variant     Molecule                                      Notes
  ============================  =========  ============================================  =================================================
  dsrg-mrpt2-1                   U, SS     :math:`\text{BeH}_{2}`                        large :math:`s` value, user defined basis set
  dsrg-mrpt2-2                   U, SS     :math:`\text{HF}`
  dsrg-mrpt2-3                   U, SS     :math:`\text{H}_4` (rectangular)
  dsrg-mrpt2-4                   U, SS     :math:`\text{N}_2`
  dsrg-mrpt2-5                   U, SS     benzyne :math:`\text{C}_6 \text{H}_4`
  dsrg-mrpt2-6                   PR, SS    :math:`\text{N}_2`
  dsrg-mrpt2-7-casscf-natorbs    PR, SS    :math:`\text{N}_2`                            CASSCF natural orbitals
  dsrg-mrpt2-8-sa                SA, SAc   :math:`\text{LiF}`                            lowest two singlet states, user defined basis set
  dsrg-mrpt2-9-xms               MS, XMS   :math:`\text{LiF}`                            lowest two singlet states
  dsrg-mrpt2-10-CO               PR, SS    :math:`\text{CO}`                             dipole moment (not linear response)
  dsrg-mrpt2-11-C2H4             SA        ethylene :math:`\text{C}_2\text{H}_4`         lowest three singlet states
  dsrg-mrpt2-12-localized-actv   SA        butadiene :math:`\text{C}_4\text{H}_6`        long, localized active orbitals
  aci-dsrg-mrpt2-1               U, SS     :math:`\text{N}_2`                            ACI(:math:`\sigma=0`)
  aci-dsrg-mrpt2-2               U, SS     :math:`\text{H}_4` (rectangular)              ACI(:math:`\sigma=0`)
  aci-dsrg-mrpt2-3               PR, SS    :math:`\text{H}_4` (rectangular)              ACI(:math:`\sigma=0`)
  aci-dsrg-mrpt2-4               U, SS     octatetraene :math:`\text{C}_8\text{H}_{10}`  DF, ACI(:math:`\sigma=0.001`), ACI batching
  aci-dsrg-mrpt2-5               PR, SS    octatetraene :math:`\text{C}_8\text{H}_{10}`  long, DF, ACI(:math:`\sigma=0.001`), ACI batching
  ============================  =========  ============================================  =================================================

2. DF/CD-DSRG-MRPT2 Test Cases
++++++++++++++++++++++++++++++

   - cd-dsrg-mrpt2-1
   - cd-dsrg-mrpt2-2
   - cd-dsrg-mrpt2-3
   - cd-dsrg-mrpt2-4
   - cd-dsrg-mrpt2-5
   - cd-dsrg-mrpt2-6
   - cd-dsrg-mrpt2-7-sa
   - df-dsrg-mrpt2-1
   - df-dsrg-mrpt2-2
   - df-dsrg-mrpt2-3
   - df-dsrg-mrpt2-4
   - df-dsrg-mrpt2-5
   - df-dsrg-mrpt2-6, LONG
   - df-dsrg-mrpt2-7-localized-actv, LONG
   - df-dsrg-mrpt2-threading1
   - df-dsrg-mrpt2-threading2
   - df-dsrg-mrpt2-threading4
   - diskdf-dsrg-mrpt2-1
   - diskdf-dsrg-mrpt2-2
   - diskdf-dsrg-mrpt2-3
   - diskdf-dsrg-mrpt2-4
   - diskdf-dsrg-mrpt2-5
   - diskdf-dsrg-mrpt2-threading1
   - diskdf-dsrg-mrpt2-threading4
   - df-aci-dsrg-mrpt2-1
   - df-aci-dsrg-mrpt2-2

3. DSRG-MRPT3 Test Cases
++++++++++++++++++++++++

   - dsrg-mrpt3-1
   - dsrg-mrpt3-2
   - dsrg-mrpt3-3, LONG
   - dsrg-mrpt3-4, LONG
   - dsrg-mrpt3-5
   - dsrg-mrpt3-6-sa, LONG
   - dsrg-mrpt3-8-sa-C2H4, LONG
   - dsrg-mrpt3-7-CO
   - dsrg-mrpt3-9
   - aci-dsrg-mrpt3-1

4. MR-DSRG Test Cases
+++++++++++++++++++++

  =================================  =======================  ============================================  =================================================
              Name                           Variant            Molecule                                      Notes
  =================================  =======================  ============================================  =================================================
  mrdsrg-pt2-1                        U, SS                    :math:`\text{BeH}_{2}`                        PT2
  mrdsrg-pt2-2                        PR, SS                   :math:`\text{BeH}_{2}`                        PT2
  mrdsrg-pt2-3                        FR, SS                   :math:`\text{BeH}_{2}`                        long, PT2
  mrdsrg-pt2-4                        FR, SS                   :math:`\text{HF}`                             PT2
  mrdsrg-srgpt2-1                     U, SS                    :math:`\text{BeH}_{2}`                        Long, SRG_PT2
  mrdsrg-srgpt2-2                     U, SS                    :math:`\text{BeH}_{2}`                        LONG, SRG_PT2, h0th=fdiag_vactv
  mrdsrg-ldsrg2-df-1                  CD, R, SS                :math:`\text{BeH}_{2}`                        long
  mrdsrg-ldsrg2-df-2                  CD, R, SS                :math:`\text{HF}`                             long
  mrdsrg-ldsrg2-df-3                  CD, U, SS                :math:`\text{H}_4` (rectangular)              long
  mrdsrg-ldsrg2-df-4                  CD, PR, SS               :math:`\text{H}_{2}`
  mrdsrg-ldsrg2-df-seq-1              CD, PR, SS, SQ           :math:`\text{BeH}_{2}`                        Long
  mrdsrg-ldsrg2-df-seq-2              CD, R, SS, SQ            :math:`\text{HF}`                             Long
  mrdsrg-ldsrg2-df-seq-3              CD, U, SS, SQ            :math:`\text{H}_4` (rectangular)              long
  mrdsrg-ldsrg2-df-seq-4              CD, FR, SS, SQ           :math:`\text{H}_4` (rectangular)              Long
  mrdsrg-ldsrg2-df-nivo-1             CD, PR, SS, NIVO         :math:`\text{BeH}_{2}`                        long
  mrdsrg-ldsrg2-df-nivo-2             CD, R, SS, NIVO          :math:`\text{HF}`                             long
  mrdsrg-ldsrg2-df-nivo-3             CD, U, SS, NIVO          :math:`\text{H}_4` (rectangular)              long
  mrdsrg-ldsrg2-df-seq-nivo-1         CD, PR, SS, SQ, NIVO     :math:`\text{BeH}_{2}`                        long
  mrdsrg-ldsrg2-df-seq-nivo-2         CD, R, SS, SQ, NIVO      :math:`\text{HF}`                             Long
  mrdsrg-ldsrg2-df-seq-nivo-3         CD, U, SS, SQ, NIVO      :math:`\text{H}_4` (rectangular)              long
  mrdsrg-ldsrg2-qc-1                  FR, QC, SS               :math:`\text{HF}`                             long
  mrdsrg-ldsrg2-qc-2                  U, QC, SS                :math:`\text{HF}`                             long
  mrdsrg-ldsrg2-qc-df-2               CD, U, QC, SS            :math:`\text{HF}`                             long
  =================================  =======================  ============================================  =================================================

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

