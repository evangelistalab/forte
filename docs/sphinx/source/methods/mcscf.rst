.. _`sec:methods:mcscf`:

MCSCF: Multi-Configuration Self-Consistent Field
================================================

.. codeauthor:: Chenyang Li, Kevin P. Hannon, Meng Huang, Shuhe Wang

.. sectionauthor:: Chenyang Li

Theory
^^^^^^

The Multi-Configuration Self-Consistent Field (MCSCF) tries to optimize the orbitals
and the CI coefficients for a multi-configuration wave function:

.. math:: |\Psi \rangle = \sum_{I}^{N_{\rm det}} c_{I} | \Phi_I \rangle,

where :math:`c_I` is the coefficient for Slater determinant :math:`\Phi_I`.
In MCSCF, the molecular orbitals (MOs) are generally separated into three subsets:
core (:math:`\mathbf{C}`, doubly occupied),
active (:math:`\mathbf{A}`),
and virtual (:math:`\mathbf{V}`, unoccuied).
The set of determinants are formed by arranging the number of active electrons
(i.e., the total number of electrons minus twice the number of core orbitals)
in the active orbitals.
There are many ways to pick the determinant basis, including complete active space (CAS),
restricted active space (RAS), generalized active space (GAS),
and other selective configuration interaction schemes (such as ACI).

For convenience, we first introduce the following convention for orbital indices:
:math:`i, j` for core orbitals,
:math:`t, u, v, w` for active orbitals,
and :math:`a, b` for virtual orbitals.
General orbitals (core, active, or virtual) are denoted using indices :math:`p,q,r,s`.

The MCSCF energy can be expressed as

.. math:: E = \sum_{tu}^{\bf A} f^{\rm c}_{tu} \, D_{tu} + \frac{1}{2} \sum_{tuvw}^{\bf A} (tu|vw)\, \overline{D}_{tu,vw} + E_{\rm c} + E_{\rm nuc},

where :math:`f^{\rm c}_{pq} = h_{pq} + \sum_{i}^{\bf C} [2 (pq|ii) - (pi|iq)]` are the closed-shell Fock matrix elements
and :math:`(pq|rs)` are the MO two-electron integrals in chemists' notation.
The term :math:`E_{\rm c} = \sum_{j}^{\bf C} (h_{jj} + f^{\rm c}_{jj})` is the closed-shell energy and :math:`E_{\rm nuc}` is the nuclear repulsion energy.
We have also used the 1- and 2-body reduced density matrices (RDMs) defined respectively as
:math:`D_{tu} = \sum_{IJ} c_I c_J \langle \Phi_I | \hat{E}_{tu} | \Phi_J \rangle`
and :math:`D_{tu,vw} = \sum_{IJ} c_I c_J \langle \Phi_I | \hat{E}_{tu,vw} | \Phi_J \rangle`,
where the unitary group generators are defined as
:math:`\hat{E}_{tu} = \sum_{\sigma}^{\uparrow \downarrow} a^\dagger_{t_\sigma} a_{u_\sigma}` and
:math:`\hat{E}_{tu,vw} = \sum_{\sigma\tau}^{\uparrow \downarrow} a^\dagger_{t_\sigma} a_{u_\sigma} a^\dagger_{v_\tau} a_{w_\tau}`.
Moreover, we use the symmetrized 2-RDM in the MCSCF energy expression such that it has the same 8-fold symmetry as the two-electron integrals:
:math:`\overline{D}_{tu,vw} = \frac{1}{2} (D_{tu,vw} + D_{ut,vw})`.

There are then two sets of parameters in MCSCF:
1) CI coefficients :math:`\{c_I|I = 1,2,\dots,N_{\rm det}\}`, and
2) MO coefficients :math:`\{C_{\mu p}| p = 1,2,\dots,N_{\rm MO}\}` with :math:`| \phi_p \rangle = \sum_{\mu}^{\rm AO} C_{\mu p} | \chi_{\mu} \rangle`.
The goal of MCSCF is then to optimize both sets of parameters to minimize the energy,
subject to orthonormal molecular orbitals
:math:`| \phi_p^{\rm new} \rangle = \sum_{s} | \phi_s^{\rm old} \rangle U_{sp}`,
:math:`{\bf U} = \exp({\bf R})` with :math:`{\bf R}^\dagger = -{\bf R}`.
It is then straightforward to see the two steps in MCSCF:
CI optimization (for given orbitals) and orbital optimization (for given RDMs).

Implementation
^^^^^^^^^^^^^^

In Forte, we implement the atomic-orbital-driven two-step MCSCF algorithm based on JK build.
We largely follow the article by Hohenstein et al.
[`J. Chem. Phys. 142, 224103 (2015) <https://doi.org/10.1063/1.4921956>`_]
with exceptions on the orbital diagonal Hessian which can be found in
`Theor. Chem. Acc. 97, 88-95 (1997) <http://link.springer.com/10.1007/s002140050241>`_
(non-redundant rotaions) and
`J. Chem. Phys. 152, 074102 (2020) <https://doi.org/10.1063/1.5142241>`_
(active-active rotaions).
The difference is that we improve the orbital optimization step via L-BFGS iterations
to obtain a better approxiamtion to the orbital Hessian.
The optimization procedure is shown in the following figure:

.. image:: images/mcscf_2step.png
    :width: 600
    :align: center
    :alt: Optimization procedure used in AO-driven 2-step MCSCF code

All types of integrals available in Forte are supported for energy computations.

.. note::
  External integrals read from a FCIDUMP file (:code:`CUSTOM`) are supported,
  but their use in the current code is very inefficient,
  which requires further optimization.

Besides MCSCF energies, we have also implement analytic MCSCF energy gradients.
Frozen orbitals are allowed for computing both the energy and gradients,
although these frozen orbitals must come from canonical Hartree-Fock
in order to compute analytic gradients.

.. warning::
  The density-fitted (:code:`DF`, :code:`DISKDF`)
  and Cholesky-decomposed (:code:`CHOLESKY`) integrals are fully supported for energy computations.
  However, there is a small discrepancy for gradients between analytic results and finite difference.
  This is caused by the DF derivative integrals in Psi4.

  Meanwhile, analytic gradient calculations are not available for FCIDUMP (:code:`CUSTOM`) integrals.

Input Example
^^^^^^^^^^^^^

The following performs an MCSCF calculation on CO molecule.
Specifically, this is a CASSCF(6,6)/cc-pCVDZ calculation with 2 frozen-core orbitals.
::

    import forte

    molecule CO{
      0 1
      C
      O  1 1.128
    }

    set {
      basis                 cc-pcvdz
      reference             rhf
      scf_type              pk
      maxiter               300
      e_convergence         10
      d_convergence         8
      docc                  [5,0,1,1]
    }

    set forte {
      job_type              mcscf_two_step
      frozen_docc           [2,0,0,0]
      frozen_uocc           [0,0,0,0]
      restricted_docc       [2,0,0,0]
      active                [2,0,2,2]
      e_convergence         8  # energy convergence of the FCI iterations
      r_convergence         8  # residual convergence of the FCI iterations
      casscf_e_convergence  8  # energy convergence of the MCSCF iterations
      casscf_g_convergence  6  # gradient convergence of the MCSCF iterations
      casscf_micro_maxiter  4  # do at least 4 micro iterations per macro iteration
    }

    Eforte = energy('forte')

Near the end of the output, we can find a summary of the MCSCF iterations:
::

    ==> MCSCF Iteration Summary <==

                        Energy CI                    Energy Orbital
             ------------------------------  ------------------------------
      Iter.        Total Energy       Delta        Total Energy       Delta  Orb. Grad.  Micro
      ----------------------------------------------------------------------------------------
         1    -112.799334478817  0.0000e+00   -112.835855509518  0.0000e+00  1.9581e-03     4
         2    -112.843709831147 -4.4375e-02   -112.849267918030 -1.3412e-02  5.8096e-03     4
         3    -112.867656057839 -2.3946e-02   -112.871626476542 -2.2359e-02  5.4580e-03     4
         4    -112.871805690190 -4.1496e-03   -112.871829079776 -2.0260e-04  9.6326e-04     4
         5    -112.871833833468 -2.8143e-05   -112.871834596898 -5.5171e-06  1.0716e-04     4
         6    -112.871834848100 -1.0146e-06   -112.871834858812 -2.6191e-07  1.4395e-05     4
         7    -112.871834862835 -1.4735e-08   -112.871834862936 -4.1231e-09  1.1799e-06     3
         8    -112.871834862954 -1.1940e-10   -112.871834862958 -2.2439e-11  1.4635e-07     2
      ----------------------------------------------------------------------------------------

The last column shows the number of micro iterations used in a given macro iteration.

To obtain the analytic energy gradients, just replace the last line of the above input to ::

    gradient('forte')

The output prints out all the components that contribute to the energy first derivatives: ::

    -Nuclear Repulsion Energy 1st Derivatives:
       Atom            X                  Y                   Z
      ------   -----------------  -----------------  -----------------
         1        0.000000000000     0.000000000000    10.563924863908
         2        0.000000000000     0.000000000000   -10.563924863908

    -Core Hamiltonian Gradient:
       Atom            X                  Y                   Z
      ------   -----------------  -----------------  -----------------
         1        0.000000000000     0.000000000000   -25.266171481954
         2        0.000000000000     0.000000000000    25.266171481954

    -Lagrangian contribution to gradient:
       Atom            X                  Y                   Z
      ------   -----------------  -----------------  -----------------
         1        0.000000000000     0.000000000000     0.763603330124
         2        0.000000000000     0.000000000000    -0.763603330124

    -Two-electron contribution to gradient:
       Atom            X                  Y                   Z
      ------   -----------------  -----------------  -----------------
         1        0.000000000000     0.000000000000    13.964810830002
         2        0.000000000000     0.000000000000   -13.964810830002

    -Total gradient:
       Atom            X                  Y                   Z
      ------   -----------------  -----------------  -----------------
         1        0.000000000000     0.000000000000     0.026167542081
         2        0.000000000000     0.000000000000    -0.026167542081

The :code:`Total gradient` can be compared with that from finite-difference calculations: ::

        1     0.00000000000000     0.00000000000000     0.02616749349810
        2     0.00000000000000     0.00000000000000    -0.02616749349810

obtained from input ::

    set findif{
      points 5
    }
    gradient('forte', dertype=0)

Here the difference between finite difference and analytic formalism is 4.8E-8,
which is reasonable as our energy only converges to 1.0E-8.
Note that only the `total` gradient is available for finite-difference calculations.

The geometry optimization is invoked by ::

    optimize('forte')                                     # Psi4 optimization procedure

    mol = psi4.core.get_active_molecule()                 # grab the optimized geoemtry
    print(mol.to_string(dtype='psi4', units='angstrom'))  # print geometry to screen

Assuming the initial geometry is close to the equilibrium, we can also pass the MCSCF
converged orbitals of the initial geometry as an initial orbital guess for subsequent
geometries along the optimization steps ::

    Ecas, ref_wfn = energy('forte', return_wfn=True)      # energy at initial geometry
    Eopt = optimize('forte', ref_wfn=ref_wfn)             # Psi4 optimization procedure

    mol = psi4.core.get_active_molecule()                 # grab optimized geometry
    print(mol.to_string(dtype='psi4', units='angstrom'))  # print geometry to screen

Similarly, we can also optimize geometries using finite difference technique: ::

    Ecas, ref_wfn = energy('forte', return_wfn=True)      # energy at initial geometry
    Eopt = optimize('forte', ref_wfn=ref_wfn, dertype=0)  # Psi4 optimization procedure

.. warning::
    After optimization, the input :code:`ref_wfn` no longer holds the data of the
    initial geometry!

.. tip::
    We could use this code to perform FCI analytic energy gradients
    (and thus geometry optimizations).
    The trick is to set all correalted orbitals as active.
    In test case :code:`casscf-opt-3`, we optimize the geometry of HF molecule at the
    FCI/3-21G level of theory with frozen 1s orbital of F.
    Note that frozen orbitals will be kept as they are in the original geometry and
    therefore the final optimized geometry will be slightly different
    if a different starting geometry is used.


Options
^^^^^^^

Basic Options
~~~~~~~~~~~~~

**CASSCF_MAXITER**

The maximum number of macro iterations.

* Type: int
* Default: 100

**CASSCF_MICRO_MAXITER**

The maximum number of micro iterations.

* Type: int
* Default: 50

**CASSCF_MICRO_MINITER**

The minimum number of micro iterations.

* Type: int
* Default: 15

**CASSCF_E_CONVERGENCE**

The convergence criterion for the energy (two consecutive energies).

* Type: double
* Default: 1.0e-8

**CASSCF_G_CONVERGENCE**

The convergence criterion for the orbital gradient (RMS of gradient vector).
This value should be roughly in the same order of magnitude as CASSCF_E_CONVERGENCE.
For example, given the default energy convergence (1.0e-8),
set CASSCF_G_CONVERGENCE to 1.0e-7 -- 1.0e-8 for a better convergence behavior.

* Type: double
* Default: 1.0e-7

**CASSCF_MAX_ROTATION**

The max value allowed in orbital update vector.
If a value in the orbital update vector is greater than this number,
the update vector will be scaled by this number / max value.

* Type: double
* Default: 0.5

**CASSCF_DIIS_START**

The iteration number to start DIIS on orbital rotation matrix R.
DIIS will not be used if this number is smaller than 1.

* Type: int
* Default: 2

**CASSCF_DIIS_MIN_VEC**

The minimum number of DIIS vectors allowed for DIIS extrapolation.

* Type: int
* Default: 2

**CASSCF_DIIS_MAX_VEC**

The maximum number of DIIS vectors, exceeding which the oldest vector will be discarded.

* Type: int
* Default: 8

**CASSCF_DIIS_FREQ**

How often to do a DIIS extrapolation.
For example, 1 means do DIIS every iteration and 2 is for every other iteration, etc.

* Type: int
* Default: 1

**CASSCF_CI_SOLVER**

Which active space solver to be used.

* Type: string
* Options: CAS, FCI, ACI, PCI
* Default: CAS

**CASSCF_DEBUG_PRINTING**

Whether to enable debug printing.

* Type: Boolean
* Default: False

**CASSCF_FINAL_ORBITAL**

What type of orbitals to be used for redundant orbital pairs for a converged calculation.

* Type: string
* Options: CANONICAL, NATURAL, UNSPECIFIED
* Default: CANONICAL

**CASSCF_NO_ORBOPT**

Turn off orbital optimization procedure if true.

* Type: Boolean
* Default: False

Expert Options
~~~~~~~~~~~~~~~

**CASSCF_INTERNAL_ROT**

Whether to enable pure internal (GASn-GASn) orbital rotations.

* Type: Boolean
* Default: False

**CASSCF_ZERO_ROT**

Zero the optimization between orbital pairs.
Format: [[irrep1, mo1, mo2], [irrep1, mo3, mo4], ...] where
irreps are 0-based, while MO indices are 1-based and relative within the irrep.
For example, zeroing the mixing of 3A1 and 2A1 translates to [[0, 3, 2]].

* Type: array
* Default: No Default

**CASSCF_ACTIVE_FROZEN_ORBITAL**

A list of active orbitals to be frozen in the casscf optimization.
Active orbitals contain all GAS1, GAS2, ..., GAS6 orbitals.
Orbital indices are zero-based and in Pitzer ordering.
For example, GAS1 [1,0,0,1]; GAS2 [1,2,2,1];
CASSCF_ACTIVE_FROZEN_ORBITAL [2,6]
means we freeze the first A2 orbital in GAS2 and the B2 orbital in GAS1.
This option is useful when doing core-excited state computations.

* Type: array
* Default: No Default

CPSCF Options
~~~~~~~~~~~~~

**CPSCF_MAXITER**

Max number of iterations for solving coupled perturbed SCF equation

* Type: int
* Default: 50

**CPSCF_CONVERGENCE**

Convergence criterion for the CP-SCF equation

* Type: double
* Default: 1.0e-8

