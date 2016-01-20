
.. include:: autodoc_abbr_options_c.rst

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

.. index::
   single: APIFCI
   pair: APIFCI; theory

.. _`sec:apifci`:

APIFCI: Adaptive Path-Integral Full Configuration Interaction
=============================================================

.. codeauthor:: Francesco A. Evangelista
.. sectionauthor:: Francesco A. Evangelista

This section describes the adaptive path-integral full configuration interaction
(APIFCI) algorithm implemented in the libadaptive plugin.   
APIFCI solves the imaginary-time Schrodinger equation to find the ground state of
a system of electrons.

.. math:: -\frac{d}{d \tau} \Psi = \hat{H} \Psi

APIFCI the wave function at time :math:`\beta` is expressed as a path integral:

.. math:: e^{-\beta \hat{H}}  |\Psi_0\rangle = \sum_{I_1\cdots I_M}|\Phi_{I_M}\rangle
   \langle\Phi_{I_M}| e^{-\tau \hat{H}}  |\Phi_{I_{M-1}}\rangle \cdots \langle\Phi_{I_2}|
   e^{-\tau \hat{H}}  |\Phi_{I_1}\rangle \langle \Phi_{I_1}|e^{-\tau \hat{H}}|\Psi^{(0)}\rangle
   :label: path_integral

A First Example
^^^^^^^^^^^^^^^

The following input peformes a APIFCI calculation on :math:`\mathrm{Li}_2`.  We request a time step
:math:`\tau` = 0.1 :math:`E_{h}` and a spawning threshold equal to 0.0001 :math:`E_{h}`. ::

        import libbtl
        import libadaptive

        molecule li2{
           Li
           Li 1 2.0000
        }

        set {
          basis sto-3g
          e_convergence 8
        }

        set libadaptive {
          tau 0.1
          spawning_threshold 0.0001
        }

        energy('apici')

The output produced by this input should look like this: ::

        Initial guess energy (variational) =     -14.645949230125 Eh

        Most important contributions to the wave function:

          0  -0.933701 0.871797367          10 |1100010000|1100010000>
          1   0.231506 0.053595199          81 |1000010001|1000010001>
          2   0.231506 0.053595199          71 |1000010010|1000010010>
          3   0.113961 0.012986997          41 |1000011000|1000011000>
        ...

        ------------------------------------------------------------------------------------------
          Cycle  Beta/Eh      Ndets     Proj. Energy/Eh  |dEp/dt|      Var. Energy/Eh   |dEv/dt|
        ------------------------------------------------------------------------------------------
              0   0.0000        418     -14.645953547690 5.858e+00     -14.645993145479 5.858e+00
             25   2.5000        418     -14.646018152091 2.584e-05     -14.646099916845 4.271e-05
             50   5.0000        418     -14.646065290580 1.886e-05     -14.646114917107 6.000e-06
        ...

            800  80.0000        422     -14.646235610328 4.679e-07     -14.646155450371 1.071e-08
            825  82.5000        422     -14.646234624473 3.943e-07     -14.646155336464 4.556e-08
            850  85.0000        422     -14.646233791918 3.330e-07     -14.646155355731 7.707e-09
        ------------------------------------------------------------------------------------------

          Calculation converged.

          ==> Post-Iterations <==

          * Adaptive-CI Var.  Energy Root   1  = -14.646160290078 Eh
          * Adaptive-CI Proj. Energy Root   1  = -14.646233791918 Eh

The APIFCI module starts by performing an initial wave function guess.
The default approach followed in the module is to form a small basis of
reference determinants (:math:`M_0`) determined using a threshold
:math:`\eta'`:

.. math:: M_0 = \{ \Phi_I : \langle \Phi_I |\hat{H}_{\eta'}|\Phi_{\rm ref}\rangle \neq 0\}

where by default :math:`\eta'` is ten times the spawning threshold used
in the APIFCI calculation.  The initial guess is obtained by diagonalization
of the Hamiltonian in the space :math:`M_0`.
Next, the APIFCI equations are propagated until convergence.
The convergence is monitored via the projective and variational estimates
of the energy.
These are defined as

.. math:: E_{\rm proj}(I) = H_II + \sum_{J \neq I} H_{JI} \frac{C_J}{C_I}

.. math:: E_{\rm var} = \sum_{IJ} C_I H_{IJ} C_J

.. index:: SAPT; SAPT0


APIFCI Options
^^^^^^^^^^^^^^

Basic APFCI Keywords
~~~~~~~~~~~~~~~~~~~~

.. include:: autodir_options_c/apifci__apifci_level.rst
.. include:: autodir_options_c/apifci__basis.rst
.. include:: autodir_options_c/apifci__df_basis_apifci.rst
.. include:: autodir_options_c/apifci__df_basis_elst.rst
.. include:: autodir_options_c/apifci__freeze_core.rst
.. include:: autodir_options_c/apifci__d_convergence.rst
.. include:: autodir_options_c/apifci__e_convergence.rst
.. include:: autodir_options_c/apifci__maxiter.rst
.. include:: autodir_options_c/apifci__print.rst

Advanced APFCI Keywords
~~~~~~~~~~~~~~~~~~~~~~~

.. include:: autodir_options_c/apifci__aio_cphf.rst
.. include:: autodir_options_c/apifci__aio_df_ints.rst
.. include:: autodir_options_c/apifci__no_response.rst
.. include:: autodir_options_c/apifci__ints_tolerance.rst
.. include:: autodir_options_c/apifci__denominator_delta.rst
.. include:: autodir_options_c/apifci__denominator_algorithm.rst
.. include:: autodir_options_c/apifci__apifci_os_scale.rst
.. include:: autodir_options_c/apifci__apifci_ss_scale.rst
.. include:: autodir_options_c/globals__debug.rst

.. index:: SAPT; higher-order

MP2 Natural Orbitals
^^^^^^^^^^^^^^^^^^^^

One of the unique features of the SAPT module is its ability to use
MP2 natural orbitals (NOs) to speed up the evaluation of the triples
contribution to disperison. By transforming to the MP2 NO basis, we can
throw away virtual orbitals that are expected to contribute little to the
dispersion energy. Speedups in excess of :math:`50 \times` are possible. In
practice, this approximation is very good and should always be applied.
Publications resulting from the use of MP2 NO-based approximations should 
cite the following: [Hohenstein:2010:104107]_.

Basic Keywords Controlling MP2 NO Approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: autodir_options_c/apifci__nat_orbs_t2.rst
.. include:: autodir_options_c/apifci__occ_tolerance.rst

Advanced Keywords Controlling MP2 NO Approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. comment .. include:: autodir_options_c/apifci__nat_orbs_t2.rst

.. index:: SAPT; charge-transfer

.. _`sec:apifcict`:

At the bottom of this output are the total SAPT energies (defined above),
they are composed of subsets of the individual terms printed above. The
individual terms are grouped according to the component of the interaction
to which they contribute. The total component energies (*i.e.,*
electrostatics, exchange, induction, and dispersion) represent what we
regard as the best estimate available at a given level of SAPT computed
from a subset of the terms of that grouping. The groupings shown above are
not unique and are certainly not rigorously defined. We regard the groupings 
used in |PSIfour| as a "chemist's grouping" as opposed to a more
mathematically based grouping, which would group all exchange terms 
(*i.e.* :math:`E_{exch-ind,resp}^{(20)}`, :math:`E_{exch-disp}^{(20)}`, *etc.* in
the exchange component. A final note is that both ``Disp22(T)``
and ``Est.Disp22(T)`` results appear if MP2 natural orbitals are 
used to evaluate the triples correction to dispersion. The ``Disp22(T)`` 
result is the triples correction as computed in the truncated NO basis;  
``Est.Disp22(T)`` is a scaled result that attempts to recover
the effect of the truncated virtual space. The ``Est.Disp22(T)``
value used in the SAPT energy and dispersion component (see [Hohenstein:2010:104107]_ 
for details).

.. include:: autodoc_abbr_options_c.rst

.. index::
   single: THREE-DSRG-MRPT2
   pair: THREE-DSRG-MRPT2; theory

.. _`sec:THREE-DSRG-MRPT2`:

THREE-DSRG-MRPT2: The Driven Similarity Renormalization Group: a  Multireference Pertubation Theory with CD and DF integrals
=============================================================

.. codeauthor:: Kevin P. Hannon and Chenyang Li
.. sectionauthor:: Kevin P. Hannon

This section describes the THREE-DSRG-MRPT2 method in libadaptive.  This method is an efficient implementation of the DSRG-MRPT2 method.  The memory requirements are much less and this allows applications up to 1000 basis functions.  As a reminder, Density-Fitting and Cholesky decomposition approaches seek to factorize the two integral integrals.

.. math:: \langle ij || ab \rangle = b_{ia}^{Q}b_{jb}^{Q} - b_{ib}^{Q}b_{ja}^{Q}

Note: The equations in implemented in this method use physicist notation for the two electron integrals, but the DF/CD literature use chemist notation.  The main difference between DF and CD is the formation of the B tensors. 
In DF, the b tensor is defined as 

.. math:: b_{pq}^{Q} = \sum_p (pq | P)[(P | Q)^{-1/2}]_{PQ}

and P and Q refer to the auxiliary basis set.  

In the CD approach, the b tensor is formed by performing a cholesky decomposition of the two electron integrals.  The accuracy of this decomposition is determined by a user defined tolerance.  The accuracy of the two electron integral is directly determined by the tolerance.    

A First Example
^^^^^^^^^^^^^^^

The following input performs a THREE_DSRG-MRPT2 calculation on N2. Since this is a multireference perturbation theory, the user typically has to select an active space, a restricted_docc and if the user wants to freeze any core orbitals, a frozen_docc.  If the system is small enough, the active space can be specified by itself.  When using DF, one needs to specify the df_basis_mp2 keyword in the globals bracket.  INT_TYPE has either the options cholesky or DF.  This input has taken from DF-DSRG-MRPT2-4 in the test folder.   ::

       import libadaptive
       molecule N2{
         0 1
         N
         N  1 R
         R = 1.1
       }
       
       set globals{
          basis                   cc-pvdz
          df_basis_mp2            cc-pvdz-ri
          reference               rhf
          scf_type                pk
          d_convergence           12
          e_convergence           15
       }
       
       set libadaptive{
          restricted_docc        [2,0,0,0,0,2,0,0]
          active                 [1,0,1,1,0,1,1,1]
          dsrg_s                  1.0
          int_type                df
       }
The output produced by this input: ::

       ==> DSRG-MRPT2 Energy Summary <==
         E0 (reference)                 =   -109.023271814349570
         <[F, T1]>                      =     -0.000030401784533
         <[F, T2]>                      =     -0.000158216679810
         <[V, T1]>                      =     -0.000192286736389
         <[V, T2]> (C_2)^4              =     -0.265177905968907
         <[V, T2]> C_4 (C_2)^2 HH       =      0.003654264803308
         <[V, T2]> C_4 (C_2)^2 PP       =      0.015965766856248
         <[V, T2]> C_4 (C_2)^2 PH       =      0.017506302404369
         <[V, T2]> C_6 C_2              =     -0.000193839261787
         <[V, T2]>                      =     -0.228245411166769
         DSRG-MRPT2 correlation energy  =     -0.228626316367502
         DSRG-MRPT2 total energy        =   -109.251898130717066
         max(T1)                        =      0.002246022946733
         max(T2)                        =      0.082589536601635
         ||T1||                         =      0.006990654164021
         ||T2||                         =      2.004482652834453
         Your calculation took 2.03655700 seconds

A Second Example
^^^^^^^^^^^^^^^^

This calculation runs a calculation on N2 with the cholesky integrals. ::

       import libadaptive
       molecule N2{
         0 1
         N
         N  1 R
         R = 1.1
       }
       set globals{
          basis                   cc-pvdz
          reference               rhf
          scf_type                pk
          d_convergence           12
          e_convergence           15
       }
       set libadaptive{
          restricted_docc        [2,0,0,0,0,2,0,0]
          active                 [1,0,1,1,0,1,1,1]
          dsrg_s                  1.0
          int_type                cholesky
          cholesky_tolerance      1e-5
       }
Running cholesky_tolerance with 1e-5 provides energies accurate to this tolerance.  The cholesky algorithm, as currently written, does not allow application to larger systems.  If one is interested in larger systems, they would be wise to use the int_type DF.  

In addition to this, timings are printed for each term.  For larger systems, one energy term of the <[V, T2]> becomes the rate limiting step and the memory bottleneck of the code.  This algorithm never stores the (me | nf) integrals in core.  This integrals correspond to the core, core, virtual, virtual block of the two electron integrals.  By avoiding the storage of the ccvv term for the integrals and all other intermediates, larger systems are within reach with the THREE-DSRG-MRPT2 method.  

The algorithm for the most expensive term is 

.. math::

    \tilde{v}_{kl}^{cd}t_{ab}^{ij}\gamma_i^k\gamma_j^l\eta_c^a\eta_d^b = \sum_{ijab}^{\text{one active}}\quad \sum_{klcd}^{\text{one active}}
    \tilde{v}_{kl}^{cd}t_{ab}^{ij}\gamma_{i}^{k}\gamma_{j}^{l}\eta_c^a\eta_b^d
    +  \sum_{mnef}\tilde{v}_{mn}^{ef}t_{ef}^{mn}

By specifing either ccvv_algorithm core, the program will just build the every block of v.  This algorithm will fail for large systems.  

For larger systems, the user should select ccvv_algorithm fly_ambit or ccvv_algorithm fly_loop if the system is large (600 basis functions).  Both the fly_ambit and fly_loop are openmp paralleized, so please use those.  

The last option worth mentioning is the dsrg_s.  Our method has a severe dependence on the s.  For the best results (one without intruders and satisfactory agreement), s should be between 0.01 and 1.  

Basic THREE-DSRG-MRPT2 Keywords
^^^^^^^^^^^^^^

These keywords are actually controlled by the integral class so they affect all areas of the code, not just THREE-DSRG-MRPT2.  

**INT_TYPE**

If one is going to use THREE-DSRG-MRPT2, this keyword needs to be set to either DF or CHOLESKY.  
The default value for integrals is actually conventional, but the code won't work if conventional is used.  

INT_TYPE tells what type of integals will be used in the calculation

* Type: string

* Possible Values: DF, CHOLESKY

* Default: CONVENTIONAL

**CHOLESKY_TOLERANCE**

The tolerance for the cholesky decomposition.  This keyword determines the accuracy of the computation.  
A smaller tolerance is a more accurate computation.  
The tolerance for the cholesky decomposition:

* Type: double in scientific notation (ie 1e-5 or 0.0001)
* Default:1.0e-6

**DF_BASIS_MP2**

The type of basis set used for DF.  This keyword needs to be placed in the globals section.  
A common advice for these is to use the basis set designed for the primary basis set, ie cc-pVDZ should use cc-pVDZ-RI.  

* Type: string specifing basis set

* Default: none

Advanced THREE-DSRG-MRPT2 keywords
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**MEMORY_SUMMARY**

A debug option for letting the user know how much each intermediate takes up in memory.  If the code is seg faulting in random places, this keyword
should be enabled so you can see which tensor is responsible for the seg fault.  

This keyword will tell how big every tensor is, how many blocks are used, and will provide a summary of the total amount of memory the tensor took up.  

* Type: bool
* Default: false

**CCVV_ALGORITHM**

The type of algorithm used for the ccvv term.  The fastest algorithm is the ccvv_ambit.  Both fly methods have the same memory requirements and are well
paralleized through openmp.  

* Type: string
* Possible values: CORE FLY_AMBIT FLY_LOOP
* Default: FLY_AMBIT

.. include:: autodoc_abbr_options_c.rst

.. index::
   single: CASSCF
   pair: CASSCF; theory

.. _`sec:CASSCF`:

CASSCF : Complete Active Space Self Consistent Field
=============================================================

.. codeauthor:: Kevin P. Hannon 
.. sectionauthor:: Kevin P. Hannon

Theory
^^^^^^

CASSCF seeks to find a solution where both the orbitals and the CI coefficients are optimum.  

.. math::   | \Psi_{\text{casscf}} \rangle = \sum_i \textbf{C}_i \exp{\textbf{x}} | \Psi_0 \rangle
   :label: casscf_1

We assume that the CI coefficients and the orbital rotations can be separated, so this means that the :math:`\textbf{C}_{i}` are solved via
a diagonalization of the CI Hamiltonian: 

.. math::   | \textbf{H} \textbf{C}_i = E_i \textbf{C}_i
    :label: casscf_ci_1

By separating the CI and the Orbital rotations, we are losing the quadratic convergence of the CASSCF method.  The equations for the orbital
optimization assume a second order expansion of the energy via a taylor series with respect to the orbital rotation parameter (:math:`\textbf{x}`)

.. math:: E^{(2)} (\textbf{x}) = E(0) + g^{T}\textbf{x} + textbf{x}^{T} H \textbf{x}

.. math:: g_{pq} = [E_{pq}^{-}, \hat{H}]

.. math:: H_{pqrs} = [[E_{pq}^{-}, \hat{H}], E_{rs}^{-}]

.. math:: E_{pq}^{-} = E_{pq} - E_{qp} 

.. math:: \hat{H} = h_{pq}E_{pq} + g_{pqrs}(E_{pqrs})

The convergence of the CASSCF algorithm implies that :math:`||g_{pq}||_2 \approx 0`.


The second term of the energy expansion is called the orbital gradient:

.. math:: g_{pq} = 2.0 (F_{pq} - F_{qp})

.. math:: F_{pq} = h_{pn} \gamma_{qn} + g_{pmnr}\Gamma_{qmnr}

.. math:: \gamma_{qn} = \langle 0 | E_{qn} | 0 \rangle

.. math:: \Gamma_{pqrs} = \langle 0 | E_{pqrs} | 0 \rangle

where :math:`\gamma` and :math:`\Gamma` are the reduced density matrices of the CI wavefunction.  By taking advantage of the sparsity of these density matrices, the
computation of the orbital gradient becomes quite inexpensive:

.. math:: F_{ip} = 2 ^{I}F_{qi} + 2 ^{A}F_{qi}

.. math:: F_{up} = ^{I}F_{pv}\gamma_{uv} + \Gamma_{uvxy} g_{pvxy}

.. math:: F_{ap} = 0

The :math:`^{I}F_{pq}` and :math:`^{A}F_{pq}` are the inactive and active Fock matrices,

.. math:: ^{I}F_{pq} = h_{mn} + 2.0 g_{pqii} - g_{piiq}

.. math:: ^{A}F_{pq} = \gamma_{ux}(g_{pqux} - \frac{1}{2} g_{pxuq})

These inactive and active fock matrices are evaluated very efficiently and do not assume storage of the integrals, so it is expected that these terms
can be evaluated for larger systems.  However, these two intermediates do become the computational bottleneck for small active spaces.  

It is
important to note that the algorithm used to form these terms directly depends on what the user specifies for the SCF_TYPE. When the user 
specifies
DF or CD as the int_type, :math: `g_{pvxy}` is the only intermediate that is evaluated with the forte integrals class.

The second term of the energy expansion is the orbital hessian.  In this program, only the diagonal values of the hessian are computed:

.. math:: H_{me, me} = 4^{I}F_ee + 4^{A}F_{ee} - 4^{I}F_{mm} - 4^{A}F_{mm}

.. math:: H_{te, te} = 2 \gamma_{tt}^{I}F_{ee} + 2 \gamma_{tt}^{A}F_{ee} - 2(^{I}F_{pu} \gamma_{tu})_{tt} + 2.0 (g_{puxy} \gamma_{tuxy})_{tt})

.. math:: H_{mt, mt} = 4.0(^{I}{F}_{tt} + ^{A}F_{tt}) + 2.0 \gamma_{tt}^{I}F_{mm} + 2.0 \gamma_{tt} ^{A}F_{mm} - (4.0 ^{I}F_{mm} + 4.0 ^{A}F_{mm})

.. math:: H_{mt, mt} = -2 (^{I}F_{pu} \gamma_{tu})_{tt} - 2.0 (g_{puxy} \Gamma_{tuxy})_{tt}

With both the orbital gradient and hessian built, 


