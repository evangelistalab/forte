.. index::
   single: ACI
   pair: ACI; theory

.. _`sec:aci`:

ACI: Adaptive Configuration Interaction
=======================================

.. codeauthor:: Francesco A. Evangelista
.. sectionauthor:: Jeffrey B. Schriber

Theory
^^^^^^

The Adaptive Configuration Interaction Method (ACI) uses a dual-parameter, 
iterative procedure to build a multideterminental wavefunction used to solve the
Schrodinger equation.

.. math::  \hat{H} \sum_{I}c_{I}\Psi_{I}= E \sum_{I}c_{I}\Psi_{I} 
   :label: aci_1

In traditional CI methods, configuration spaces are built with respect to a predefined
excitation order from a single reference---CISD includes singly and doubly excited 
determinants, CISDT adds triples, etc. While these approaches can describe single-reference
systems with appreciable accuracy, they cannot describe multireference systems, they are not
size-extensive, and they often include many determinants that do not contribute 
significantly in either the wavefunction or the energy. 

Though not size-extensive, ACI forms a compact configuration space capable of describing
multireference systems. Starting with a single- or multi- determinental reference, called the 
initial :math:`P` space, all unique single and double excitations are screened using the first
parameter :math:`\tau_{q}`. For each determinant :math:`\Psi_{SD}` in the single-double 
space :math:`SD`, an energetic contribution is estimated with perturbation theory as

.. math:: E^{SD}_{PT2} = \sum_{I}^{\Psi_I \in P}\frac{|\langle \Psi_{I}|\hat{H}|\Psi_{SD}\rangle|^{2}}{\Delta_{SD}^{I}}
   :label: ept2

or with direct diagonalization as

.. comment this is only implemeted in ex-aci
.. math:: E^{SD}_{corr} = \sum_{I}^{\Psi_I \in P}2\Delta_{SD}^{I} - 2\sqrt{\Delta_{SD}^{I^{2}} + V_{I,SD}^{2}}  
   :label: direct_diag

where :math:`\Delta_{SD}^{I}` is the normal energy denominator. Additionally, this criteria
can also be defined as the first order perturbative correction to the wavefunction:

.. math:: C_{PT2}^{SD} = \sum_{I}^{\Psi_I \in P}\frac{\langle \Psi_{I}|\hat{H}|\Psi_{SD}\rangle}{\Delta_{SD}^{I}}

If the selected criteria is greater than :math:`\tau_{q}`, then that determinant :math:`\Psi_{SD}` 
is added to a new determinant space called the :math:`Q` space. Alternatively, the :math:`Q` space can
be created by summing the lowest energy contributions until this sum reaches 
:math:`\tau_{q}`, whereafter the determinants whose energy contribution was excluded
from this are then included in the :math:`Q` space. This is called the "aimed" algorithm. In both cases, 
the effect of the determinants not included in the :math:`P+Q` space can be included using an `a posteriori` correction for
now called :math:`E_{PT2}`.

Once the :math:`Q` is constructed, the Hamiltonian is built and diagonalized in the :math:`P+Q` space.
Only determinants whose expansion coefficient is greater than :math:`\tau_{p}` in magnitude are kept
and added to the updated :math:`P` space. From here, the procedure iterates until the energy is converged.

A First Example
^^^^^^^^^^^^^^^

The simplest input for an ACI calculation involves specifying values for :math:`\tau_{p}`, :math:`\tau_{q}`,
and the definition of the energy criteria. ::

        import libadaptive

        molecule li2{
           Li
           Li 1 2.0000
        }

        set {
          basis sto-3g
          e_convergence 10
        }
                 
        set libadaptive{
          select_type energy
          taup 0.01
          tauq 0.01
        }

        energy('aci')

Though not required, it is good practice to also specify the number of roots, multiplicity, symmetry, and charge. 
The output contains information about the sizes and energies of the :math:`P` and :math:`P+Q` spaces at each
step of the iteration. ::
        Cycle   1
         Dimension of the P space: 53 determinants
         
        ...

          P-space  CI Energy Root   1        = -14.645855651246 Eh =   0.0000 eV

        Dimension of the SD space: 993 determinants
        Time spent building the model space: 0.004031 s

        Dimension of the P + Q space: 55 determinants
        Time spent screening the model space: 0.000644 s

        ...

          PQ-space CI Energy Root   1        = -14.645928006457 Eh =   0.0000 eV
          PQ-space CI Energy + EPT2 Root   1 = -14.646163676471 Eh =   0.0000 eV


        Most important contributions to root   0:
        0  -0.933642 0.871686681           0 |1100010000|1100010000>    
        1   0.229711 0.052767022          50 |1000010001|1000010001>
        2   0.229711 0.052767022          44 |1000010010|1000010010>
        3   0.119695 0.014326909          28 |1000011000|1000011000>
        4   0.080508 0.006481570          10 |1010010000|1010010000>
        5   0.024565 0.000603443          36 |1000010100|1000011000>
        6   0.024565 0.000603443          29 |1000011000|1000010100>
        7   0.014633 0.000214115          53 |1100010000|1010010000>
        8   0.014633 0.000214115          54 |1010010000|1100010000>
        9   0.007793 0.000060729          37 |1000010100|1000010100>
        
        ...

        ==> Post-Iterations <==

          * Adaptive-CI Energy Root   1        = -14.645928006457 Eh =   0.0000 eV
          * Adaptive-CI Energy Root   1 + EPT2 = -14.646166917621 Eh =   0.0000 eV

        Adaptive-CI (bitset) ran in : 0.018030 s

        Saving information for root: 1
        Your calculation took 0.05159900 seconds

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

Computing Excited States with ACI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additional functionality has been added to the adaptive-ci class, and these changes are implemented
in the EX_ACI class. Upon optimization, these changes will be moved to the adaptive-ci class, and
here all functionality of the current EX_ACI code will be summarized. All options from the adaptive-ci
class are still useable in EX-ACI.

A First Example
~~~~~~~~~~~~~~~

Here is an example input file that computes the lowest two states of :math:`Li_{2}` by comparing 
parameters :math:`\tau_{p}` and :math:`\tau_q` to the respective averages of the MP2 energy 
estimate and CI expansion coefficient, where these averages run over all roots of interest. :: 

        import libadaptive

        molecule li2{
           Li
           Li 1 2.0000
        }

        set {
          basis sto-3g
          e_convergence 10
        }
                 
        set libadaptive{
          multiplicity 1
          select_type energy
          excited_algorithm state_average
          pq_function average
          taup 0.01
          tauq 0.01
          nroot 2
        }

        energy('ex-aci')

A Second Example
~~~~~~~~~~~~~~~~
Below is a similar example, but with two key differences. First, the :math:`\tau_{q}` parameter
is defined as the lowest eigenvalue obtained from diagonalizing a 2-dimensional matrix containing
the CI-wavefunction and a determinant outside of that space (see above). The second difference is
that the maximum values for each criteria among the excited states are chosen as the importance criteria
for a given determinant.:: 

        import libadaptive

        molecule li2{
           Li
           Li 1 2.0000
        }

        set {
          basis sto-3g
          e_convergence 10
        }
                 
        set libadaptive{
          multiplicity 1
          perturb_select false
          select_type energy
          excited_algorithm state_average
          pq_function max
          taup 0.01
          tauq 0.01
          nroot 2
        }

        energy('ex-aci')

EX-ACI Options
~~~~~~~~~~~~~~

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


