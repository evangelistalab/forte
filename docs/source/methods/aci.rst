.. _`sec:aci`:

ACI: Adaptive Configuration Interaction
=======================================

.. codeauthor:: Francesco A. Evangelista and Jeffrey B. Schriber
.. sectionauthor:: Jeffrey B. Schriber

Theory
^^^^^^

The Adaptive Configuration Interaction Method (ACI)
is an iterative selected CI method that optimizes a space of determinants such that
the total error in the energy is controlled by a user-defined parameter, :math:`\sigma`,

.. math::  |E_{\text{CASCI}} - E_{\text{ACI}}| \approx \sigma .

The ACI algorithm grows a set of reference determinants (:math:`P`) and screens its first-order
interacting space using perturbative energy estimates. This screening is done in a cumulative
fasion to produce an approximation to the total correlation energy ignored.
The space of reference (:math:`P`) and selected determinants (:math:`Q`) define the ACI model space (:math:`M`).
The Hamiltonian is diagonalized in this space to produce the ACI energy and wave function,

.. math:: |\Psi_{M}\rangle = \sum_{\Phi_{\mu}}C_{\mu}|\Phi_{\mu}\rangle .

The algorithm proceeds with a pruning step to get a new (:math:`P`) space to start the next iteration. The iterations end when the
ACI energy is satisfactorily converged, which produces a total error that matches :math:`sigma` very closely.
Additionally, the perturbative estimates of the determinants excluded from the model space can be used as a perturbative correction,
which we denote as the ACI+PT2 energy.

A Few Practical Notes
^^^^^^^^^^^^^^^^^^^^^
- In Forte, ACI wave functions are defined only in the active orbitals.
- The ACI wave function is defined is a set of Slater Determinants, so it is not guaranteed to be a pure spin eigenfunction. As a result, we augment ACI wave functions throughout the procedure to ensure each intermediate space of determinants is spin complete.
- The initial :math:`P` space is defined from a small CAS wave function so that there are fewer than 1000 determinants. This can be enlarged to improve convergence if needed using the ACTIVE_GUESS_SIZE option.
- This portion of the manual will discuss ACI usage generally, but all content is transferrable to the case where ACI is used as a reference for DSRG computations. If that is the case, the option CAS_TYPE ACI needs to be set.

A First Example
^^^^^^^^^^^^^^^

The simplest input for an ACI calculation involves specifying values for :math:`\sigma`.

::

        import forte

        molecule h2o {
        0 1
          O
          H 1 0.96
          H 1 0.96 2 104.5
        }

        set {
            basis sto-3g
            reference rhf
        }

        set forte {
            job_type aci
            sigma 0.001
        }
        E_scf, scf_wfn = energy('scf', return_wfn=True)
        energy('forte', ref_wfn=scf_wfn)



Though not required, it is good practice to also specify the number of roots, multiplicity, symmetry, and charge.
The output contains information about the sizes and energies of the :math:`P` and :math:`M` spaces at each
step of the iteration, and provides a summary of the converged wave function:  ::

  ==> ACI Summary <==

  Iterations required:                         3
  Dimension of optimized determinant space:    24

  * Adaptive-CI Energy Root   0        = -75.012317069484 Eh =   0.0000 eV
  * Adaptive-CI Energy Root   0 + EPT2 = -75.013193884201 Eh =   0.0000 eV

  ==> Wavefunction Information <==

  Most important contributions to root   0:
    0  -0.987158 0.974480589          16 |2220220>
    1   0.076700 0.005882905          15 |2220202>
    2  -0.046105 0.002125685          13 |22-+2+->
    3  -0.046105 0.002125685          14 |22+-2-+>
    4   0.044825 0.002009273          12 |2202220>
    5   0.043438 0.001886853          11 |2222200>
    6   0.040971 0.001678638          10 |2200222>
    7   0.033851 0.001145896           9 |22--2++>
    8   0.033851 0.001145896           8 |22++2-->
    9   0.032457 0.001053488           7 |2+-2220>

  Spin state for root 0: S^2 = 0.000000, S = 0.000, singlet

  ==> Computing Coupling Lists <==
  --------------------------------
        α          0.000186 s
        β          0.000186 s
        αα         0.000333 s
        ββ         0.000307 s
        αβ         0.000866 s
  --------------------------------
  1-RDM  took 0.000107 s (determinant)

  ==> NATURAL ORBITALS <==

        1A1     2.000000      1B1     1.998476      2A1     1.998399
        3A1     1.977478      1B2     1.974442      2B2     0.025891
        4A1     0.025314


  RDMS took 0.002290

  Adaptive-CI ran in : 0.067389 s

For ground state computations, very few additional options are required unless very large determinants spaces are considered. In this case, memory efficient
screening and diagonalization algorithms can be chosen.

Computing Excited States with ACI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



ACI Options
^^^^^^^^^^^

Basic Options
~~~~~~~~~~~~~

**NROOT**

Number of CI roots to find. If energy('aci') is used, energy criteria will be computed for each
root with respect to a trial wavefunction. The maximum value among each root will then be used
for evaluation with :math:`\tau_{q}`.

* Type: int
* Default: 1

**SELECT_TYPE**

Specifies whether second order PT theory energy correction, or first order amplitude is used
in selecting the :math:`Q` space.

* Type: string
* Options: AMP, ENERGY, AIMED_AMP, AIMED_ENERGY
* Default: AMP

**TAUP**

Threshold used to prune the :math:`P+Q` space

* Type: double
* Default: 0.01

**TAUQ**

Threshold used to select the :math:`Q` space

* Type: double
* Default: 0.000001


Expert Options
~~~~~~~~~~~~~~

**DIAG_ALGORITHM**

The algorithm used in all diagonalizations. This option is only needed for calculations
with very large configuration spaces.

* Type: string
* Options: DAVIDSON, FULL, DAVIDSON_LIST
* Default: DAVIDSON

**SMOOTH**

This option implements a smoothing function for the Hamiltonian that makes the energy an
everywhere-differentiable function of a geometric coordinate by gradually gradually
decoupling the determinant of least importance. This function is useful for correcting
discontinuities in potential energy curves, but it can yeild non-physical curves if the
discontinuities are large.

* Type: bool
* Default: False

**SMOOTH_THRESHOLD**

The threshold for smoothing the Hamiltonian

* Type: double
* Default: 0.01



**EXCITED_ALGORITHM**

This option determines the algorithm to compute excited states. Currently the only options
implemented are "STATE_AVERAGE" which means that a function of the criteria among the excited
states of interest are used to build the configuraiton space, and "ROOT_SELECT" where the
determinant space is constructed with respect to a single root.

* Type: string
* Options: "STATE_AVERAGE", "ROOT_SELECT"
* Default: "STATE_AVERAGE"

**PERTURB_SELECT**

Option defines :math:`\tau_{q}` as either MP2 estimate or estimate derived from 2D diagonalization.
True uses the MP2 estimation.

* Type: bool
* Default: false

**POST_DIAGONALIZE**

Option to re-diagonalize Hamiltonian in final CI space. This can be is useful to compute more roots.

* Type: bool
* Default: False

**POST_ROOT**

Number of roots to compute on post-diagonalization. For this option to be used, post-diagonalize
must be true.

* Type: int
* Default: 1

**PQ_FUNCTION**

Option that selects the function of energy estimates per root and the expansion coefficients per root.
This option is only meaningful if more than one root is desired.

* Type: string
* Options: "MAX", "AVERAGE"
* Default: "MAX"

**Q_REFERENCE**

Reference state type to be used when computing estimates of the energy difference between two states. The
estimation of the change in energy gap a determinant introduces can be done for all excited states with
respect to the ground state (GS), or with respect to the nearest, lower state.

* Type: string
* Options: "GS", "ADJACENT"
* Default: "GS"

**Q_REL**

Rather than using the absolute energy to define the importance of a determinant, an energy gap between
 two states can be used. This allows the determinant space to be constructed such that the energy difference
 between to states is optimized.

* Type: bool
* Default: False

**REF_ROOT**

Option that selects the desired root that is used to build the determinant space. This option should
only be used when the EXCITED_ALGORITHM is set to "ROOT_SELECT".

* Type: int
* Default: 0

**SPIN_TOL**

For all of the algorithms in EX_ACI, roots are only used to build determinant spaces if their spin
multiplicity is within a given tolerance of the input spin multiplicity. This option defines that
spin tolerance. NOTE: the multiplicity must be defined within the EX_ACI scope. For poorly behaved
systems, it may be useful to increase this to an arbitrarily large value such that the lowest-energy
multiplicities can be confirmed

* Type: double
* Default: 1.0e-4
