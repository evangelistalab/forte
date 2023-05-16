FCI: Full Configuration Interaction
===================================

Theory
------

Forte contains a determinant-based implementation of the Full
Configuration Interaction (FCI) method. The FCI method belongs to the
category of active space solvers and can be used on its own or to
produce reference states for subsequent multireference treatments of
dynamical electron correlation. In FCI, the user specifies a set of
active orbitals and all Slater determinants :math:`\Phi_I` with the
correct symmetry and value of :math:`M_S` are generated. The FCI wave
function :math:`\Psi` is a linear combination of the determinants

.. math::


   |\Psi\rangle = \sum_{I}^{N_\mathrm{FCI}} C_I |\Phi_I\rangle

where :math:`N_\mathrm{FCI}` is the number of determinants in the FCI
space and :math:`C_I` are the expansion coefficients.

The FCI wave function is variationally optimized by solving the
eigenvalue problem

.. math::


   \hat{H} |\Psi\rangle = E_\mathrm{FCI} |\Psi\rangle

where :math:`\hat{H}` is the Hamiltonian operator.

In Forte, the FCI wave function is always assumed to be normalized, that
is, the coefficients :math:`C_I` are normalized such that

.. math::


   \sum_{I}^{N_\mathrm{FCI}} |C_I|^2 = 1

The FCI energy is given by the expectation value of the Hamiltonian
operator :math:`\hat{H}` with respect to the FCI wave function

.. math::


   E_\mathrm{FCI} = \langle \Psi | \hat{H} | \Psi \rangle

Implementation details
----------------------

To solve the FCI equations, Forte’s implementation uses the Davidson–Liu
solver, which requires computing only the action of the Hamiltonian onto
a general FCI vector :math:`\sum_I b_I |\Phi_I\rangle`:

.. math::


   \hat{H} \sum_I b_I |\Phi_I\rangle = \sum_I \sigma_I |\Phi_I\rangle

In this case, the final state can be written in terms of the coefficient
vector :math:`\boldsymbol{\sigma} = \mathbf{H} \mathbf{b}`, where here
we use matrix/vector notation.

The algorithm that computes
:math:`\boldsymbol{\sigma} = \mathbf{H} \mathbf{b}` is called the
**sigma algorithm**. Forte implements the string-based sigma algorithm
Bendazzoli and Evangelisti [Bendazzoli, G. L.; Evangelisti, S. *J Chem
Phys* **98**, 3141 (1993)].

In Forte’s string-based implementation, the FCI wave function is
expressed in terms of separate occupation strings for the alpha
(:math:`I_\alpha`) and beta (:math:`I_\beta`) electrons, so that a state
can be written as

.. math::


   |\Psi\rangle = \sum_{I_\mathrm{\alpha}}^{N_\alpha} \sum_{I_\mathrm{\beta}}^{N_\beta} C_{I_\alpha I_\beta} |\Phi_{I_\alpha I_\beta}\rangle
   = \sum_{I_\mathrm{\alpha}}^{N_\alpha} \sum_{I_\mathrm{\beta}}^{N_\beta} C_{I_\alpha I_\beta} |I_\alpha \rangle \otimes |I_\beta \rangle

This means that internally, the FCI vector :math:`C_I` is stored as a
matrix indexed by the strings :math:`I_\alpha` and :math:`I_\beta`:
:math:`C_{[I_\alpha][I_\beta]}`.

A Few Practical Notes
---------------------

-  FCI computations in Forte run only in the active orbitals (defined by
   the ``ACTIVE`` keyword). See more information in the Tutorials to
   learn how to define the target active space.
-  The default procedure in Forte solves the FCI eigenvalue equation in
   the Slater determinant basis. In this case Forte projects out
   contaminants of the wrong spin symmetry; however, this procedure can
   fail to yield a state with correct multiplicity. To guarantee that
   the final state is an eigenfunction of :math:`\hat{S}^2`, turn on
   spin adapation by setting the option ``CI_SPIN_ADAPT`` to ``True``.
-  Difficult cases of convergence can be resolved with a combination of
   using a spin-adapted FCI algorithm and by modifying the parameters of
   the Davidson–Liu solver used by the FCI code.

A First Example
---------------

The following is an example of a FCI computation on the Li2 molecule
using the cc-pVDZ basis.

.. code:: python

   # forte/tests/manual/fci-1/input.dat

   import forte

   molecule li2 {
   0 1
   Li
   Li 1 1.6
   }

   set {
       basis cc-pVDZ
       reference rhf
   }

   set forte {
       active_space_solver fci
   }

   # run a RHF computation
   E_scf, scf_wfn = energy('scf', return_wfn=True)

   # pass the RHF orbitals to Forte and run a FCI computation
   energy('forte', ref_wfn=scf_wfn)

The output of this computation contains useful information about the
number of determinants, symmetry, multiplicity of the root
(:math:`2S + 1`), number of roots required, etc.

::

     ==> FCI Solver <==

       Number of determinants                     1345608
       Symmetry                                         0
       Multiplicity                                     1
       Number of roots                                  1
       Target root                                      0
       Trial vectors per root                          10
       Spin adapt                                   false

     Allocating memory for the Hamiltonian algorithm. Size: 2 x 435 x 435.   Memory: 0.002820 GB

By default, to guess an intial solution, the FCI code identifies a small
list of low-energy determinant that are spin complete and diagonalizes
the Hamiltonian and :math:`\hat{S}^2` operators. This procedure yields a
list of root with their corresponding energy and expectation value of
:math:`\hat{S}^2`. The roots projected out are listed at the bottom.

::

     ==> FCI Initial Guess <==

     ---------------------------------------------
       Root            Energy     <S^2>   Spin
     ---------------------------------------------
         0      -14.821706304246  0.000  singlet
         1      -14.701890096488  0.000  singlet
         2      -14.697750811390  2.000  triplet
         3      -14.688167598595  0.000  singlet
         4      -14.626162130001  0.000  singlet
         5      -14.623675382053  2.000  triplet
       ...
        19      -14.374215745393  0.000  singlet
     ---------------------------------------------
     Timing for initial guess  =      0.002 s

     Projecting out guess roots: [2,5,7,9,11,13,15]

The next block shows the convergence of the Davidson–Liu procedure

::

   ==> Diagonalizing Hamiltonian <==

     Energy   convergence: 1.00e-06
     Residual convergence: 1.00e-06
     -----------------------------------------------------
       Iter.      Avg. Energy       Delta_E     Res. Norm
     -----------------------------------------------------
         1      -14.821706304246  -1.482e+01  +2.474e-01
         2      -14.834596416564  -1.289e-02  +3.363e-02
         3      -14.835056965855  -4.605e-04  +1.352e-02
         4      -14.835126439226  -6.947e-05  +4.432e-03
       ...
        11      -14.835137265477  -1.047e-11  +1.575e-06
        12      -14.835137265478  -9.344e-13  +4.629e-07
     -----------------------------------------------------
     The Davidson-Liu algorithm converged in 13 iterations.

For each target root(s), a list of the most important determinants is
shown next. In this printout, the orbitals are grouped by irrep and
their occupation is labeled by one character (``0`` = empty, ``a``/``b``
= one alpha/beta electron, ``2`` doubly occupied):

::

     ==> Root No. 0 <==

       2200000 0 000 000 0 2000000 000 000     -0.91351927
       2000000 0 000 000 0 2000000 000 200      0.19711995
       2000000 0 000 000 0 2000000 200 000      0.19711995
       2000000 0 000 000 0 2000000 ab0 000      0.11601362
       2000000 0 000 000 0 2000000 ba0 000      0.11601362
       2000000 0 000 000 0 2000000 000 ab0      0.11601362
       2000000 0 000 000 0 2000000 000 ba0      0.11601362

The energy of all the roots is summarized in a table:

::

     ==> Energy Summary <==

       Multi.(2ms)  Irrep.  No.               Energy      <S^2>
       --------------------------------------------------------
          1  (  0)    Ag     0      -14.835137265478   0.000000
       --------------------------------------------------------

The end of the output also shows the expectation value of the dipole and
quadrupole operators and the occupation numer of natural orbitals
(obtained from the FCI one-body reduced density matrix):

::

     ==> NATURAL ORBITALS <==

           1Ag     1.998924      1B1u    1.998773      2Ag     1.679206
           1B3u    0.143579      1B2u    0.143579      2B1u    0.024482
           ...

Spin-adapted FCI
----------------

In certain cases, convergence to a state with target multiplicity fails
due to either variational collapse to a root of lower energy and
different multiplicity or because no guess state can be found. Spin
adaptation can be turned on by setting the option ``CI_SPIN_ADAPT`` to
``True``.

Forte implements within the determinant-based FCI code a procedure to
perform the Davidson–Liu procedure in a basis of configuration state
funcions (CSFs). CSFs are spin-adapted linear combinations of Slater
determinants with a given orbital occupation pattern (electron
configuration).

When expressed in the CSF basis a FCI state is given by:

.. math::


   |\Psi\rangle = \sum_{i} C'_{i} | \mathrm{CSF}_i \rangle

where the coefficients :math:`C'_{i}` are **different** from the ones
that express :math:`\Psi` in the Slater determinant basis. Forte’s
spin-adapted code computes the sigma vector in the determinant basis
and, before feeding it to the Davidson–Liu solver, it converts it to the
CSF basis. Spin-adapted FCI computation are more expensive than
conventional ones, with the additional cost of the order of 10–15%.

To demonstrate the utility of spin adaptation, consider a computation of
the :math:`A_1` quintet state of Li2. In a determinant code, a
straightforward modification of the previous example (``fci-1``) fails
because the algorithm that guesses the initial state cannot find a
quintet state.

In the following input we set ``ci_spin_adapt`` to ``True`` and specify
the multiplicity of the state (``5``).

.. code:: python

   # forte/tests/manual/fci-2/input.dat

   import forte

   molecule li2 {
   0 1
   Li
   Li 1 2.0
   }

   set {
       basis cc-pVDZ
       reference rhf
       e_convergence 9
   }

   set forte {
       active_space_solver fci
       ci_spin_adapt true
       multiplicity 5
   }

   # run a RHF computation
   E_scf, scf_wfn = energy('scf', return_wfn=True)

   # pass the RHF orbitals to Forte and run a FCI computation
   energy('forte', ref_wfn=scf_wfn)

The output file contains some extra/different sections. At the beginning
of the computation we can read information about the number of CSF and
CSF construction timing:

::

     ==> Spin Adapter <==

       Number of CSFs:                            295572
       Number of couplings:                      4570632

       Timing for identifying configurations:     0.1099
       Timing for finding the CSFs:               0.3369

The initial guess contains only CSFs with the correct value of spin:

::

     ==> FCI Initial Guess <==

     Selected 2 CSF
     ---------------------------------------------
       CSF             Energy     <S^2>   Spin
     ---------------------------------------------
     227224     -12.339361395518  6.000  quintet
     110597     -12.339361395518  6.000  quintet
     ---------------------------------------------
     Timing for initial guess  =      0.002 s

The final state is a linear combination of many determinants

::

    ==> Root No. 0 <==

       2a00000 0 0b0 000 0 a000000 000 b00     -0.13907742
       2b00000 0 0a0 000 0 b000000 000 a00     -0.13907742
       2b00000 0 0a0 000 0 a000000 000 b00      0.13907742
       2a00000 0 0b0 000 0 b000000 000 a00      0.13907742
       2a00000 0 0a0 000 0 b000000 000 b00      0.13907742
       2b00000 0 0b0 000 0 a000000 000 a00      0.13907742
       ...
       2a00000 0 000 a00 0 b000000 b00 000     -0.13247287
       2b00000 0 000 b00 0 a000000 a00 000     -0.13247287

**Since in this example the orbitals come from a RHF computation** (same
number of alpha and beta electrons) Forte will assume that the target
state havs :math:`M_S = 0`. This can be seen from the determinant
composition and in the final energy summary that reports the value of
:math:`2 M_S` (``2ms``)

::

     ==> Energy Summary <==

       Multi.(2ms)  Irrep.  No.               Energy      <S^2>
       --------------------------------------------------------
          5  (  0)    Ag     0      -12.596862494551   6.000000
       --------------------------------------------------------
