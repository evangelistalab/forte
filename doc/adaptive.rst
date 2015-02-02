
.. include:: autodoc_abbr_options_c.rst

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

