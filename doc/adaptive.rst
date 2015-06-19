
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

