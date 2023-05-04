.. _`sec:options`:

List of Forte options
=====================

.. sectionauthor:: Francesco A. Evangelista


General options
===============

**ACTIVE_REF_TYPE**

Initial guess for active space wave functions

Type: str

Default value: CAS

Allowed values: ['HF', 'CAS', 'GAS', 'GAS_SINGLE', 'CIS', 'CID', 'CISD', 'DOCI']

**ACTIVE_SPACE_SOLVER**

Active space solver type

Type: str

Default value: 

Allowed values: ['FCI', 'ACI', 'ASCI', 'PCI', 'DETCI', 'CAS', 'DMRG']

**CALC_TYPE**

The type of computation

Type: str

Default value: SS

Allowed values: ['SS', 'SA', 'MS', 'DWMS']

**CHARGE**

The charge of the molecule. If a value is provided it overrides the charge of Psi4.

Type: int

Default value: None

**CORRELATION_SOLVER**

Dynamical correlation solver type

Type: str

Default value: NONE

Allowed values: ['DSRG-MRPT2', 'THREE-DSRG-MRPT2', 'DSRG-MRPT3', 'MRDSRG', 'SA-MRDSRG', 'DSRG_MRPT', 'MRDSRG_SO', 'SOMRDSRG']

**DERTYPE**

Derivative order

Type: str

Default value: NONE

Allowed values: ['NONE', 'FIRST']

**DUMP_ORBITALS**

Save orbitals to file if true

Type: bool

Default value: False

**D_CONVERGENCE**

The density convergence criterion

Type: float

Default value: 1e-06

**E_CONVERGENCE**

The energy convergence criterion

Type: float

Default value: 1e-09

**JOB_TYPE**

Specify the job type

Type: str

Default value: NEWDRIVER

Allowed values: ['NONE', 'NEWDRIVER', 'MR-DSRG-PT2', 'CASSCF', 'MCSCF_TWO_STEP']

**MINAO_BASIS**

The basis used to define an orbital subspace

Type: str

Default value: STO-3G

**MS**

Projection of spin onto the z axis

Type: float

Default value: None

**MULTIPLICITY**

The multiplicity = (2S + 1) of the electronic state. For example, 1 = singlet, 2 = doublet, 3 = triplet, ... If a value is provided it overrides the multiplicity of Psi4.

Type: int

Default value: None

**NEL**

The number of electrons. Used when reading from FCIDUMP files.

Type: int

Default value: None

**ORBITAL_TYPE**

Type of orbitals to use

Type: str

Default value: CANONICAL

Allowed values: ['CANONICAL', 'LOCAL', 'MP2NO', 'MRPT2NO']

**PRINT**

Set the print level.

Type: int

Default value: 1

**READ_ORBITALS**

Read orbitals from file if true

Type: bool

Default value: False

**REF_TYPE**

The type of reference used by forte if a psi4 wave function is missing

Type: str

Default value: SCF

Allowed values: ['SCF', 'CASSCF']

**ROOT_SYM**

The symmetry of the electronic state. (zero based)

Type: int

Default value: None

**SCF_TYPE**

The integrals used in the SCF calculation

Type: str

Default value: None

**SUBSPACE**

A list of orbital subspaces

Type: gen_list

Default value: []

**SUBSPACE_PI_PLANES**

A list of arrays of atoms composing the plane

Type: gen_list

Default value: []

ACI options
===========

**ACI_ADD_AIMED_DEGENERATE**

Add degenerate determinants not included in the aimed selection

Type: bool

Default value: True

**ACI_APPROXIMATE_RDM**

Approximate the RDMs?

Type: bool

Default value: False

**ACI_AVERAGE_OFFSET**

Offset for state averaging

Type: int

Default value: 0

**ACI_CONVERGENCE**

ACI Convergence threshold

Type: float

Default value: 1e-09

**ACI_LOW_MEM_SCREENING**

Use low-memory screening algorithm?

Type: bool

Default value: False

**ACI_MAX_MEM**

Sets max memory for batching algorithm (MB)

Type: int

Default value: 1000

**ACI_NBATCH**

Number of batches in screening

Type: int

Default value: 0

**ACI_NFROZEN_CORE**

Number of orbitals to freeze for core excitations

Type: int

Default value: 0

**ACI_NO**

Computes ACI natural orbitals?

Type: bool

Default value: False

**ACI_NO_THRESHOLD**

Threshold for active space prediction

Type: float

Default value: 0.02

**ACI_N_AVERAGE**

Number of roots to average. When set to zero (default) it averages over all roots

Type: int

Default value: 0

**ACI_PQ_FUNCTION**

Function of q-space criteria, per root for SA-ACI

Type: str

Default value: AVERAGE

Allowed values: ['AVERAGE', 'MAX']

**ACI_PRESCREEN_THRESHOLD**

The SD space prescreening threshold

Type: float

Default value: 1e-12

**ACI_PRINT_NO**

Print the natural orbitals?

Type: bool

Default value: True

**ACI_PRINT_REFS**

Print the P space?

Type: bool

Default value: False

**ACI_PRINT_WEIGHTS**

Print weights for active space prediction?

Type: bool

Default value: False

**ACI_REF_RELAX**

Do reference relaxation in ACI?

Type: bool

Default value: False

**ACI_RELAXED_SPIN**

Do spin correlation analysis for relaxed wave function?

Type: bool

Default value: False

**ACI_RELAX_SIGMA**

Sigma for reference relaxation

Type: float

Default value: 0.01

**ACI_ROOTS_PER_CORE**

Number of roots to compute per frozen orbital

Type: int

Default value: 1

**ACI_SCALE_SIGMA**

Scales sigma in batched algorithm

Type: float

Default value: 0.5

**ACI_SCREEN_ALG**

The screening algorithm to use

Type: str

Default value: AVERAGE

Allowed values: ['AVERAGE', 'SR', 'RESTRICTED', 'CORE', 'BATCH_HASH', 'BATCH_VEC', 'MULTI_GAS']

**ACI_SPIN_PROJECTION**

Type of spin projection
     0 - None
     1 - Project initial P spaces at each iteration
     2 - Project only after converged PQ space
     3 - Do 1 and 2

Type: int

Default value: 0

**ACI_SPIN_TOL**

Tolerance for S^2 value

Type: float

Default value: 0.02

**ACTIVE_GUESS_SIZE**

Number of determinants for CI guess

Type: int

Default value: 1000

**CORR_LIMIT**

Correlation limit for considering if two orbitals are correlated in the post calculation analysis.

Type: float

Default value: -0.01

**DIAG_ALGORITHM**

The diagonalization method

Type: str

Default value: SPARSE

Allowed values: ['DYNAMIC', 'FULL', 'SPARSE']

**FORCE_DIAG_METHOD**

Force the diagonalization procedure?

Type: bool

Default value: False

**FULL_MRPT2**

Compute full PT2 energy?

Type: bool

Default value: False

**GAMMA**

The threshold for the selection of the Q space

Type: float

Default value: 1.0

**N_GUESS_VEC**

Number of guess vectors for Sparse CI solver

Type: int

Default value: 10

**OCC_ANALYSIS**

Doing post calcualtion occupation analysis?

Type: bool

Default value: False

**OCC_LIMIT**

Occupation limit for considering if an orbital is occupied/unoccupied in the post calculation analysis.

Type: float

Default value: 0.0001

**ONE_CYCLE**

Doing only one cycle of ACI (FCI) ACI iteration?

Type: bool

Default value: False

**PI_ACTIVE_SPACE**

Active space type?

Type: bool

Default value: False

**PRINT_IAOS**

Print IAOs?

Type: bool

Default value: True

**SIGMA**

The energy selection threshold for the P space

Type: float

Default value: 0.01

**SPIN_ANALYSIS**

Do spin correlation analysis?

Type: bool

Default value: False

**SPIN_BASIS**

Basis for spin analysis

Type: str

Default value: LOCAL

Allowed values: ['LOCAL', 'IAO', 'NO', 'CANONICAL']

**SPIN_MAT_TO_FILE**

Save spin correlation matrix to file?

Type: bool

Default value: False

**SPIN_PROJECT_FULL**

Project solution in full diagonalization algorithm?

Type: bool

Default value: False

**SPIN_TEST**

Do test validity of correlation analysis

Type: bool

Default value: False

**UNPAIRED_DENSITY**

Compute unpaired electron density?

Type: bool

Default value: False

ASCI options
============

**ASCI_CDET**

ASCI Max reference det

Type: int

Default value: 200

**ASCI_E_CONVERGENCE**

ASCI energy convergence threshold

Type: float

Default value: 1e-05

**ASCI_PRESCREEN_THRESHOLD**

ASCI prescreening threshold

Type: float

Default value: 1e-12

**ASCI_TDET**

ASCI Max det

Type: int

Default value: 2000

AVAS options
============

**AVAS**

Form AVAS orbitals?

Type: bool

Default value: False

**AVAS_CUTOFF**

The eigenvalues of the overlap greater than this cutoff will be considered as active. If not equal to 1.0, it takes priority over cumulative cutoff selection.

Type: float

Default value: 1.0

**AVAS_DIAGONALIZE**

Diagonalize Socc and Svir?

Type: bool

Default value: True

**AVAS_EVALS_THRESHOLD**

Threshold smaller than which is considered as zero for an eigenvalue of the projected overlap.

Type: float

Default value: 1e-06

**AVAS_NUM_ACTIVE**

The total number of active orbitals. If not equal to 0, it takes priority over threshold-based selections.

Type: int

Default value: 0

**AVAS_NUM_ACTIVE_OCC**

The number of active occupied orbitals. If not equal to 0, it takes priority over cutoff-based selections and that based on the total number of active orbitals.

Type: int

Default value: 0

**AVAS_NUM_ACTIVE_VIR**

The number of active virtual orbitals. If not equal to 0, it takes priority over cutoff-based selections and that based on the total number of active orbitals.

Type: int

Default value: 0

**AVAS_SIGMA**

Cumulative cutoff to the eigenvalues of the overlap, which controls the size of the active space. This value is tested against (sum of active e.values) / (sum of total e.values)

Type: float

Default value: 0.98

Active Space Solver options
===========================

**AVG_STATE**

A list of integer triplets that specify the irrep, multiplicity, and the number of states requested.Uses the format [[irrep1, multi1, nstates1], [irrep2, multi2, nstates2], ...]

Type: gen_list

Default value: []

**AVG_WEIGHT**

A list of lists that specify the weights assigned to all the states requested with AVG_STATE [[w1_1, w1_2, ..., w1_n], [w2_1, w2_2, ..., w2_n], ...]

Type: gen_list

Default value: []

**DUMP_ACTIVE_WFN**

Save CI wave function of ActiveSpaceSolver to disk

Type: bool

Default value: False

**DUMP_TRANSITION_RDM**

Dump transition reduced matrix into disk?

Type: bool

Default value: False

**NROOT**

The number of roots computed

Type: int

Default value: 1

**PRINT_DIFFERENT_GAS_ONLY**

Only calculate the transition dipole between states with different GAS occupations?

Type: bool

Default value: False

**READ_ACTIVE_WFN_GUESS**

Read CI wave function of ActiveSpaceSolver from disk

Type: bool

Default value: False

**ROOT**

The root selected for state-specific computations

Type: int

Default value: 0

**S_TOLERANCE**

The maximum deviation from the spin quantum number S tolerated.

Type: float

Default value: 0.25

**TRANSITION_DIPOLES**

Compute the transition dipole moments and oscillator strengths

Type: bool

Default value: False

CASSCF options
==============

**CASSCF_ACTIVE_FROZEN_ORBITAL**

A list of active orbitals to be frozen in the MCSCF optimization (in Pitzer order, zero based). Useful when doing core-excited state computations.

Type: int_list

Default value: []

**CASSCF_CI_FREQ**

How often to solve CI?
< 1: do CI in the first macro iteration ONLY
= n: do CI every n macro iteration

Type: int

Default value: 1

**CASSCF_CI_SOLVER**

The active space solver to use in CASSCF

Type: str

Default value: FCI

**CASSCF_CI_STEP**

Do a CAS step for every CASSCF_CI_FREQ

Type: bool

Default value: False

**CASSCF_CI_STEP_START**

When to start skipping CI steps

Type: int

Default value: -1

**CASSCF_DEBUG_PRINTING**

Enable debug printing if True

Type: bool

Default value: False

**CASSCF_DIE_IF_NOT_CONVERGED**

Stop Forte if MCSCF is not converged

Type: bool

Default value: True

**CASSCF_DIIS_FREQ**

How often to do DIIS extrapolation

Type: int

Default value: 1

**CASSCF_DIIS_MAX_VEC**

Maximum size of DIIS vectors for orbital rotations

Type: int

Default value: 8

**CASSCF_DIIS_MIN_VEC**

Minimum size of DIIS vectors for orbital rotations

Type: int

Default value: 3

**CASSCF_DIIS_NORM**

Do DIIS when the orbital gradient norm is below this value

Type: float

Default value: 0.001

**CASSCF_DIIS_START**

Iteration number to start adding error vectors (< 1 will not do DIIS)

Type: int

Default value: 15

**CASSCF_DO_DIIS**

Use DIIS in CASSCF orbital optimization

Type: bool

Default value: True

**CASSCF_E_CONVERGENCE**

The energy convergence criterion (two consecutive energies)

Type: float

Default value: 1e-08

**CASSCF_FINAL_ORBITAL**

Constraints for redundant orbital pairs at the end of macro iteration

Type: str

Default value: CANONICAL

Allowed values: ['CANONICAL', 'NATURAL', 'UNSPECIFIED']

**CASSCF_G_CONVERGENCE**

The orbital gradient convergence criterion (RMS of gradient vector)

Type: float

Default value: 1e-07

**CASSCF_INTERNAL_ROT**

Keep GASn-GASn orbital rotations if true

Type: bool

Default value: False

**CASSCF_MAXITER**

The maximum number of CASSCF macro iterations

Type: int

Default value: 100

**CASSCF_MAX_ROTATION**

Max value in orbital update vector

Type: float

Default value: 0.2

**CASSCF_MICRO_MAXITER**

The maximum number of CASSCF micro iterations

Type: int

Default value: 40

**CASSCF_MICRO_MINITER**

The minimum number of CASSCF micro iterations

Type: int

Default value: 6

**CASSCF_MULTIPLICITY**

Multiplicity for the CASSCF solution (if different from multiplicity)
    You should not use this if you are interested in having a CASSCF
    solution with the same multiplicitity as the DSRG-MRPT2

Type: int

Default value: 0

**CASSCF_NO_ORBOPT**

No orbital optimization if true

Type: bool

Default value: False

**CASSCF_ORB_ORTHO_TRANS**

Ways to compute the orthogonal transformation U from orbital rotation R

Type: str

Default value: CAYLEY

Allowed values: ['CAYLEY', 'POWER', 'PADE']

**CASSCF_REFERENCE**

Run a FCI followed by CASSCF computation?

Type: bool

Default value: False

**CASSCF_SOSCF**

Run a complete SOSCF (form full Hessian)?

Type: bool

Default value: False

**CASSCF_ZERO_ROT**

An array of MOs [[irrep1, mo1, mo2], [irrep2, mo3, mo4], ...]

Type: gen_list

Default value: []

**CPSCF_CONVERGENCE**

Convergence criterion for CP-SCF equation

Type: float

Default value: 1e-08

**CPSCF_MAXITER**

Max iteration of solving coupled perturbed SCF equation

Type: int

Default value: 50

**MONITOR_SA_SOLUTION**

Monitor the CAS-CI solutions through iterations

Type: bool

Default value: False

**OPTIMIZE_FROZEN_CORE**

Ignore frozen core option and optimize orbitals?

Type: bool

Default value: False

**ORB_ROTATION_ALGORITHM**

Orbital rotation algorithm

Type: str

Default value: DIAGONAL

Allowed values: ['DIAGONAL', 'AUGMENTED_HESSIAN']

**RESTRICTED_DOCC_JK**

Use JK builder for restricted docc (EXPERT)?

Type: bool

Default value: True

CINO options
============

**CINO**

Do a CINO computation?

Type: bool

Default value: False

**CINO_AUTO**

{ass frozen_docc, actice_docc, and restricted_docc?

Type: bool

Default value: False

**CINO_NROOT**

The number of roots computed

Type: int

Default value: 1

**CINO_ROOTS_PER_IRREP**

The number of excited states per irreducible representation

Type: int_list

Default value: []

**CINO_THRESHOLD**

The fraction of NOs to include in the active space

Type: float

Default value: 0.99

**CINO_TYPE**

The type of wave function.

Type: str

Default value: CIS

Allowed values: ['CIS', 'CISD']

DETCI options
=============

**DETCI_CISD_NO_HF**

Exclude HF determinant in active CID/CISD space

Type: bool

Default value: False

**DETCI_PRINT_CIVEC**

The printing threshold for CI vectors

Type: float

Default value: 0.05

DMRG options
============

**DMRG_PRINT_CORR**

Whether or not to print the correlation functions after the DMRG calculation

Type: bool

Default value: False

**DMRG_SWEEP_DVDSON_RTOL**

The residual tolerances for the Davidson diagonalization during DMRG instructions

Type: float_list

Default value: []

**DMRG_SWEEP_ENERGY_CONV**

Energy convergence to stop an instruction during successive DMRG instructions

Type: float_list

Default value: []

**DMRG_SWEEP_MAX_SWEEPS**

Max number of sweeps to stop an instruction during successive DMRG instructions

Type: int_list

Default value: []

**DMRG_SWEEP_NOISE_PREFAC**

The noise prefactors for successive DMRG instructions

Type: float_list

Default value: []

**DMRG_SWEEP_STATES**

Number of reduced renormalized basis states kept during successive DMRG instructions

Type: int_list

Default value: []

DSRG options
============

**AO_DSRG_MRPT2**

Do AO-DSRG-MRPT2 if true (not available)

Type: bool

Default value: False

**CCVV_ALGORITHM**

Algorithm to compute the CCVV term in DSRG-MRPT2 (only in three-dsrg-mrpt2 code)

Type: str

Default value: FLY_AMBIT

Allowed values: ['CORE', 'FLY_AMBIT', 'FLY_LOOP', 'BATCH_CORE', 'BATCH_VIRTUAL', 'BATCH_CORE_GA', 'BATCH_VIRTUAL_GA', 'BATCH_VIRTUAL_MPI', 'BATCH_CORE_MPI', 'BATCH_CORE_REP', 'BATCH_VIRTUAL_REP']

**CCVV_BATCH_NUMBER**

Batches for CCVV_ALGORITHM

Type: int

Default value: -1

**CCVV_SOURCE**

Definition of source operator: special treatment for the CCVV term

Type: str

Default value: NORMAL

Allowed values: ['ZERO', 'NORMAL']

**CORR_LEVEL**

Correlation level of MR-DSRG (used in mrdsrg code, LDSRG2_P3 and QDSRG2_P3 not implemented)

Type: str

Default value: PT2

Allowed values: ['PT2', 'PT3', 'LDSRG2', 'LDSRG2_QC', 'LSRG2', 'SRG_PT2', 'QDSRG2', 'LDSRG2_P3', 'QDSRG2_P3']

**DSRGPT**

Renormalize (if true) the integrals for purturbitive calculation (only in toy code mcsrgpt2)

Type: bool

Default value: True

**DSRG_3RDM_ALGORITHM**

Algorithm to compute 3-RDM contributions in fully contracted [H2, T2]

Type: str

Default value: EXPLICIT

Allowed values: ['EXPLICIT', 'DIRECT']

**DSRG_DIIS_FREQ**

Frequency of extrapolating error vectors for DSRG DIIS

Type: int

Default value: 1

**DSRG_DIIS_MAX_VEC**

Maximum size of DIIS vectors

Type: int

Default value: 8

**DSRG_DIIS_MIN_VEC**

Minimum size of DIIS vectors

Type: int

Default value: 3

**DSRG_DIIS_START**

Iteration cycle to start adding error vectors for DSRG DIIS (< 1 for not doing DIIS)

Type: int

Default value: 2

**DSRG_DIPOLE**

Compute (if true) DSRG dipole moments

Type: bool

Default value: False

**DSRG_DUMP_AMPS**

Dump converged amplitudes to the current directory

Type: bool

Default value: False

**DSRG_DUMP_RELAXED_ENERGIES**

Dump the energies after each reference relaxation step to JSON.

Type: bool

Default value: False

**DSRG_HBAR_SEQ**

Evaluate H_bar sequentially if true

Type: bool

Default value: False

**DSRG_MAXITER**

Max iterations for nonperturbative MR-DSRG amplitudes update

Type: int

Default value: 50

**DSRG_MRPT2_DEBUG**

Excssive printing for three-dsrg-mrpt2

Type: bool

Default value: False

**DSRG_MRPT3_BATCHED**

Force running the DSRG-MRPT3 code using the batched algorithm

Type: bool

Default value: False

**DSRG_MULTI_STATE**

Multi-state DSRG options (MS and XMS recouple states after single-state computations)
  - State-average approach
    - SA_SUB:  form H_MN = <M|Hbar|N>; M, N are CAS states of interest
    - SA_FULL: redo a CASCI
  - Multi-state approach (currently only for MRPT2)
    - MS:  form 2nd-order Heff_MN = <M|H|N> + 0.5 * [<M|(T_M)^+ H|N> + <M|H T_N|N>]
    - XMS: rotate references such that <M|F|N> is diagonal before MS procedure

Type: str

Default value: SA_FULL

Allowed values: ['SA_FULL', 'SA_SUB', 'MS', 'XMS']

**DSRG_NIVO**

NIVO approximation: Omit tensor blocks with >= 3 virtual indices if true

Type: bool

Default value: False

**DSRG_POWER**

The power of the parameter s in the regularizer

Type: float

Default value: 2.0

**DSRG_PT2_H0TH**

Different Zeroth-order Hamiltonian of DSRG-MRPT (used in mrdsrg code)

Type: str

Default value: FDIAG

Allowed values: ['FDIAG', 'FFULL', 'FDIAG_VACTV', 'FDIAG_VDIAG']

**DSRG_RDM_MS_AVG**

Form Ms-averaged density if true

Type: bool

Default value: False

**DSRG_READ_AMPS**

Read initial amplitudes from the current directory

Type: bool

Default value: False

**DSRG_RESTART_AMPS**

Restart DSRG amplitudes from a previous step

Type: bool

Default value: True

**DSRG_RSC_NCOMM**

The maximum number of commutators in the recursive single commutator approximation

Type: int

Default value: 20

**DSRG_RSC_THRESHOLD**

The treshold for terminating the recursive single commutator approximation

Type: float

Default value: 1e-12

**DSRG_S**

The value of the DSRG flow parameter s

Type: float

Default value: 0.5

**DSRG_T1_AMPS_GUESS**

The initial guess of T1 amplitudes for nonperturbative DSRG methods

Type: str

Default value: PT2

Allowed values: ['PT2', 'ZERO']

**DSRG_TRANS_TYPE**

DSRG transformation type

Type: str

Default value: UNITARY

Allowed values: ['UNITARY', 'CC']

**FORM_HBAR3**

Form 3-body Hbar (only used in dsrg-mrpt2 with SA_SUB for testing)

Type: bool

Default value: False

**FORM_MBAR3**

Form 3-body mbar (only used in dsrg-mrpt2 for testing)

Type: bool

Default value: False

**IGNORE_MEMORY_ERRORS**

Continue running DSRG-MRPT3 even if memory exceeds

Type: bool

Default value: False

**INTERNAL_AMP**

Include internal amplitudes for VCIS/VCISD-DSRG acording to excitation level

Type: str

Default value: NONE

Allowed values: ['NONE', 'SINGLES_DOUBLES', 'SINGLES', 'DOUBLES']

**INTERNAL_AMP_SELECT**

Excitation types considered when internal amplitudes are included
Select only part of the asked internal amplitudes (IAs) in V-CIS/CISD
- AUTO: all IAs that changes excitations (O->V; OO->VV, OO->OV, OV->VV)
- ALL:  all IAs (O->O, V->V, O->V; OO->OO, OV->OV, VV->VV, OO->VV, OO->OV, OV->VV)
- OOVV: pure external (O->V; OO->VV)

Type: str

Default value: AUTO

Allowed values: ['AUTO', 'ALL', 'OOVV']

**INTRUDER_TAMP**

Threshold for amplitudes considered as intruders for printing

Type: float

Default value: 0.1

**ISA_B**

Intruder state avoidance parameter when use ISA to form amplitudes (only in toy code mcsrgpt2)

Type: float

Default value: 0.02

**MAXITER_RELAX_REF**

Max macro iterations for DSRG reference relaxation

Type: int

Default value: 15

**NTAMP**

Number of largest amplitudes printed in the summary

Type: int

Default value: 15

**PRINT_1BODY_EVALS**

Print eigenvalues of 1-body effective H

Type: bool

Default value: False

**PRINT_DENOM2**

Print (if true) (1 - exp(-2*s*D)) / D, renormalized denominators in DSRG-MRPT2

Type: bool

Default value: False

**PRINT_TIME_PROFILE**

Print detailed timings in dsrg-mrpt3

Type: bool

Default value: False

**RELAX_E_CONVERGENCE**

The energy relaxation convergence criterion

Type: float

Default value: 1e-08

**RELAX_REF**

Relax the reference for MR-DSRG

Type: str

Default value: NONE

Allowed values: ['NONE', 'ONCE', 'TWICE', 'ITERATE']

**R_CONVERGENCE**

Residue convergence criteria for amplitudes

Type: float

Default value: 1e-06

**SMART_DSRG_S**

Automatically adjust the flow parameter according to denominators

Type: str

Default value: DSRG_S

Allowed values: ['DSRG_S', 'MIN_DELTA1', 'MAX_DELTA1', 'DAVG_MIN_DELTA1', 'DAVG_MAX_DELTA1']

**SOURCE**

Source operator used in DSRG (AMP, EMP2, LAMP, LEMP2 only available in toy code mcsrgpt2)

Type: str

Default value: STANDARD

Allowed values: ['STANDARD', 'LABS', 'DYSON', 'AMP', 'EMP2', 'LAMP', 'LEMP2']

**T1_AMP**

The way of forming T1 amplitudes (only in toy code mcsrgpt2)

Type: str

Default value: DSRG

Allowed values: ['DSRG', 'SRG', 'ZERO']

**TAYLOR_THRESHOLD**

DSRG Taylor expansion threshold for small denominator

Type: int

Default value: 3

**THREEPDC_ALGORITHM**

Algorithm for evaluating 3-body cumulants in three-dsrg-mrpt2

Type: str

Default value: CORE

Allowed values: ['CORE', 'BATCH']

**THREE_MRPT2_TIMINGS**

Detailed printing (if true) in three-dsrg-mrpt2

Type: bool

Default value: False

**T_ALGORITHM**

The way of forming T amplitudes (DSRG_NOSEMI, SELEC, ISA only available in toy code mcsrgpt2)

Type: str

Default value: DSRG

Allowed values: ['DSRG', 'DSRG_NOSEMI', 'SELEC', 'ISA']

DWMS options
============

**DWMS_ALGORITHM**

DWMS algorithms:
  - SA: state average Hαβ = 0.5 * ( <α|Hbar(β)|β> + <β|Hbar(α)|α> )
  - XSA: extended state average (rotate Fαβ to a diagonal form)
  - MS: multi-state (single-state single-reference)
  - XMS: extended multi-state (single-state single-reference)
  - To Be Deprecated:
    - SH-0: separated diagonalizations, non-orthogonal final solutions
    - SH-1: separated diagonalizations, orthogonal final solutions

Type: str

Default value: SA

Allowed values: ['MS', 'XMS', 'SA', 'XSA', 'SH-0', 'SH-1']

**DWMS_CORRLV**

DWMS-DSRG-PT level

Type: str

Default value: PT2

Allowed values: ['PT2', 'PT3']

**DWMS_DELTA_AMP**

Consider (if true) amplitudes difference between states X(αβ) = A(β) - A(α) in SA algorithm, testing in non-DF DSRG-MRPT2

Type: bool

Default value: False

**DWMS_E_CONVERGENCE**

Energy convergence criteria for DWMS iteration

Type: float

Default value: 1e-07

**DWMS_ITERATE**

Iterative update the reference CI coefficients in SA algorithm, testing in non-DF DSRG-MRPT2

Type: bool

Default value: False

**DWMS_MAXITER**

Max number of iteration in the update of the reference CI coefficients in SA algorithm, testing in non-DF DSRG-MRPT2

Type: int

Default value: 10

**DWMS_REFERENCE**

Energies to compute dynamic weights and CI vectors to do multi-state
  CAS: CASCI energies and CI vectors
  PT2: SA-DSRG-PT2 energies and SA-DSRG-PT2/CASCI vectors
  PT3: SA-DSRG-PT3 energies and SA-DSRG-PT3/CASCI vectors
  PT2D: Diagonal SA-DSRG-PT2c effective Hamiltonian elements and original CASCI vectors

Type: str

Default value: CASCI

Allowed values: ['CASCI', 'PT2', 'PT3', 'PT2D']

**DWMS_ZETA**

Automatic Gaussian width cutoff for the density weights
Weights of state α:
Wi = exp(-ζ * (Eα - Ei)^2) / sum_j exp(-ζ * (Eα - Ej)^2)Energies (Eα, Ei, Ej) can be CASCI or SA-DSRG-PT2/3 energies.

Type: float

Default value: 0.0

Davidson-Liu options
====================

**DL_COLLAPSE_PER_ROOT**

The number of trial vector to retain after collapsing

Type: int

Default value: 2

**DL_GUESS_SIZE**

Set the number of determinants in the initial guess space for the DL solver

Type: int

Default value: 50

**DL_MAXITER**

The maximum number of Davidson-Liu iterations

Type: int

Default value: 100

**DL_SUBSPACE_PER_ROOT**

The maxim number of trial vectors

Type: int

Default value: 10

**SIGMA_VECTOR_MAX_MEMORY**

The maximum number of doubles stored in memory in the sigma vector algorithm

Type: int

Default value: 67108864

Embedding options
=================

**EMBEDDING**

Whether to perform embedding partition and projection

Type: bool

Default value: False

**EMBEDDING_ADJUST_B_DOCC**

Adjust number of occupied orbitals between A and B, +: move to B, -: move to A

Type: int

Default value: 0

**EMBEDDING_ADJUST_B_UOCC**

Adjust number of virtual orbitals between A and B, +: move to B, -: move to A

Type: int

Default value: 0

**EMBEDDING_CUTOFF_METHOD**

Cut off by: threshold ,cum_threshold or num_of_orbitals.

Type: str

Default value: THRESHOLD

**EMBEDDING_REFERENCE**

HF for any reference without active, CASSCF for any reference with an active space.

Type: str

Default value: CASSCF

**EMBEDDING_SEMICANONICALIZE_ACTIVE**

Perform semi-canonicalization on active space or not

Type: bool

Default value: True

**EMBEDDING_SEMICANONICALIZE_FROZEN**

Perform semi-canonicalization on frozen core/virtual space or not

Type: bool

Default value: True

**EMBEDDING_THRESHOLD**

Projector eigenvalue threshold for both simple and cumulative threshold

Type: float

Default value: 0.5

**EMBEDDING_VIRTUAL_SPACE**

Vitual space scheme

Type: str

Default value: ASET

Allowed values: ['ASET', 'PAO', 'IAO']

**NUM_A_DOCC**

Number of occupied orbitals in A fixed to this value when embedding method is num_of_orbitals

Type: int

Default value: 0

**NUM_A_UOCC**

Number of virtual orbitals in A fixed to this value when embedding method is num_of_orbitals

Type: int

Default value: 0

**PAO_FIX_VIRTUAL_NUMBER**

Enable this option will generate PAOs equivlent to ASET virtuals, instead of using threshold

Type: bool

Default value: False

**PAO_THRESHOLD**

Virtual space truncation threshold for PAO.

Type: float

Default value: 1e-08

FCI options
===========

**FCI_MAXITER**

Maximum number of iterations for FCI code

Type: int

Default value: 30

**FCI_TEST_RDMS**

Test the FCI reduced density matrices?

Type: bool

Default value: False

**NTRIAL_PER_ROOT**

The number of trial guess vectors to generate per root

Type: int

Default value: 10

**PRINT_NO**

Print the NO from the rdm of FCI

Type: bool

Default value: False

FCIMO options
=============

**FCIMO_ACTV_TYPE**

The active space type

Type: str

Default value: COMPLETE

Allowed values: ['COMPLETE', 'CIS', 'CISD', 'DOCI']

**FCIMO_CISD_NOHF**

Ground state: HF; Excited states: no HF determinant in CISD space

Type: bool

Default value: True

**FCIMO_IPEA**

Generate IP/EA CIS/CISD space

Type: str

Default value: NONE

Allowed values: ['NONE', 'IP', 'EA']

**FCIMO_PRINT_CIVEC**

The printing threshold for CI vectors

Type: float

Default value: 0.05

GAS options
===========

**GAS1MAX**

The maximum number of electrons in GAS1 for different states

Type: int_list

Default value: []

**GAS1MIN**

The minimum number of electrons in GAS1 for different states

Type: int_list

Default value: []

**GAS2MAX**

The maximum number of electrons in GAS2 for different states

Type: int_list

Default value: []

**GAS2MIN**

The minimum number of electrons in GAS2 for different states

Type: int_list

Default value: []

**GAS3MAX**

The maximum number of electrons in GAS3 for different states

Type: int_list

Default value: []

**GAS3MIN**

The minimum number of electrons in GAS3 for different states

Type: int_list

Default value: []

**GAS4MAX**

The maximum number of electrons in GAS4 for different states

Type: int_list

Default value: []

**GAS4MIN**

The minimum number of electrons in GAS4 for different states

Type: int_list

Default value: []

**GAS5MAX**

The maximum number of electrons in GAS5 for different states

Type: int_list

Default value: []

**GAS5MIN**

The minimum number of electrons in GAS5 for different states

Type: int_list

Default value: []

**GAS6MAX**

The maximum number of electrons in GAS6 for different states

Type: int_list

Default value: []

**GAS6MIN**

The minimum number of electrons in GAS6 for different states

Type: int_list

Default value: []

Integrals options
=================

**FCIDUMP_DOCC**

The number of doubly occupied orbitals assumed for a FCIDUMP file. This information is used to build orbital energies.

Type: int_list

Default value: []

**FCIDUMP_FILE**

The file that stores the FCIDUMP integrals

Type: str

Default value: INTDUMP

**FCIDUMP_SOCC**

The number of singly occupied orbitals assumed for a FCIDUMP file. This information is used to build orbital energies.

Type: int_list

Default value: []

**INT_TYPE**

The type of molecular integrals used in a computation- CONVENTIONAL Conventional four-index two-electron integrals- DF Density fitted two-electron integrals- CHOLESKY Cholesky decomposed two-electron integrals- FCIDUMP Read integrals from a file in the FCIDUMP format

Type: str

Default value: CONVENTIONAL

Allowed values: ['CONVENTIONAL', 'CHOLESKY', 'DF', 'DISKDF', 'FCIDUMP']

**PRINT_INTS**

Print the one- and two-electron integrals?

Type: bool

Default value: False

Localize options
================

**LOCALIZE**

The method used to localize the orbitals

Type: str

Default value: PIPEK_MEZEY

Allowed values: ['PIPEK_MEZEY', 'BOYS']

**LOCALIZE_SPACE**

Sets the orbital space for localization

Type: int_list

Default value: []

MO Space Info options
=====================

**ACTIVE**

 Number of active orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**FROZEN_DOCC**

Number of frozen occupied orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**FROZEN_UOCC**

Number of frozen unoccupied orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**GAS1**

Number of GAS1 orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**GAS2**

Number of GAS2 orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**GAS3**

Number of GAS3 orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**GAS4**

Number of GAS4 orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**GAS5**

Number of GAS5 orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**GAS6**

Number of GAS6 orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**RESTRICTED_DOCC**

Number of restricted doubly occupied orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**RESTRICTED_UOCC**

Number of restricted unoccupied orbitals per irrep (in Cotton order)

Type: int_list

Default value: []

**ROTATE_MOS**

An array of MOs to swap in the format [irrep, mo_1, mo_2, irrep, mo_3, mo_4]. Irrep and MOs are all 1-based (NOT 0-based)!

Type: int_list

Default value: []

MRCINO options
==============

**MRCINO**

Do a MRCINO computation?

Type: bool

Default value: False

**MRCINO_AUTO**

Allow the users to choosewhether pass frozen_doccactice_docc and restricted_doccor not

Type: bool

Default value: False

**MRCINO_NROOT**

The number of roots computed

Type: int

Default value: 1

**MRCINO_ROOTS_PER_IRREP**

The number of excited states per irreducible representation

Type: int_list

Default value: []

**MRCINO_THRESHOLD**

The fraction of NOs to include in the active space

Type: float

Default value: 0.99

**MRCINO_TYPE**

The type of wave function.

Type: str

Default value: CIS

Allowed values: ['CIS', 'CISD']

Old options
===========

**BASIS**

The primary basis set

Type: str

Default value: 

**BASIS_RELATIVISTIC**

The basis set used to run relativistic computations

Type: str

Default value: 

**CHOLESKY_TOLERANCE**

Tolerance for Cholesky integrals

Type: float

Default value: 1e-06

**DF_BASIS_MP2**

Auxiliary basis set for density fitting computations

Type: str

Default value: 

**DF_FITTING_CONDITION**

Eigenvalue threshold for RI basis

Type: float

Default value: 1e-10

**DF_INTS_IO**

IO caching for CP corrections

Type: str

Default value: NONE

Allowed values: ['NONE', 'SAVE', 'LOAD']

**DIIS_MAX_VECS**

The maximum number of DIIS vectors

Type: int

Default value: 5

**DIIS_MIN_VECS**

The minimum number of DIIS vectors

Type: int

Default value: 2

**INTS_TOLERANCE**

Schwarz screening threshold

Type: float

Default value: 1e-12

**MAXITER**

The maximum number of iterations

Type: int

Default value: 100

**MEMORY_SUMMARY**

Print summary of memory

Type: bool

Default value: False

**PT2NO_OCC_THRESHOLD**

Occupancy smaller than which is considered as active

Type: float

Default value: 0.98

**PT2NO_VIR_THRESHOLD**

Occupancy greater than which is considered as active

Type: float

Default value: 0.02

**NAT_ACT**

Use Natural Orbitals to suggest active space?

Type: bool

Default value: False

**NAT_ORBS_PRINT**

View the natural orbitals with their symmetry information

Type: bool

Default value: False

**REFERENCE**

The SCF refernce type

Type: str

Default value: 

**SEMI_CANONICAL**

Semicanonicalize orbitals for each elementary orbital space

Type: bool

Default value: True

**SEMI_CANONICAL_MIX_ACTIVE**

Treat all GAS orbitals together for semi-canonicalization

Type: bool

Default value: False

**SEMI_CANONICAL_MIX_INACTIVE**

Treat frozen and restricted orbitals together for semi-canonicalization

Type: bool

Default value: False

**SRG_COMM**

Select a modified commutator

Type: str

Default value: STANDARD

Allowed values: ['STANDARD', 'FO', 'FO2']

**SRG_DT**

The initial time step used by the ode solver

Type: float

Default value: 0.001

**SRG_ODEINT**

The integrator used to propagate the SRG equations

Type: str

Default value: FEHLBERG78

Allowed values: ['DOPRI5', 'CASHKARP', 'FEHLBERG78']

**SRG_ODEINT_ABSERR**

The absolute error tollerance for the ode solver

Type: float

Default value: 1e-12

**SRG_ODEINT_RELERR**

The absolute error tollerance for the ode solver

Type: float

Default value: 1e-12

**SRG_SMAX**

The end value of the integration parameter s

Type: float

Default value: 10.0

**THREEPDC**

The form of the three-particle density cumulant

Type: str

Default value: MK

Allowed values: ['MK', 'MK_DECOMP', 'ZERO']

**TWOPDC**

The form of the two-particle density cumulant

Type: str

Default value: MK

Allowed values: ['MK', 'ZERO']

**USE_DMRGSCF**

Use the older DMRGSCF algorithm?

Type: bool

Default value: False

PCI options
===========

**PCI_ADAPTIVE_BETA**

Use an adaptive time step?

Type: bool

Default value: False

**PCI_CHEBYSHEV_ORDER**

The order of Chebyshev truncation

Type: int

Default value: 5

**PCI_COLINEAR_THRESHOLD**

The minimum norm of orthogonal vector

Type: float

Default value: 1e-06

**PCI_DL_COLLAPSE_PER_ROOT**

The number of trial vector to retain after Davidson-Liu collapsing

Type: int

Default value: 2

**PCI_DL_SUBSPACE_PER_ROOT**

The maxim number of trial Davidson-Liu vectors

Type: int

Default value: 8

**PCI_DYNAMIC_PRESCREENING**

Use dynamic prescreening?

Type: bool

Default value: False

**PCI_ENERGY_ESTIMATE_FREQ**

Iterations in between variational estimation of the energy

Type: int

Default value: 1

**PCI_ENERGY_ESTIMATE_THRESHOLD**

The threshold with which we estimate the variational energy. Note that the final energy is always estimated exactly.

Type: float

Default value: 1e-06

**PCI_EVAR_MAX_ERROR**

The max allowed error for variational energy

Type: float

Default value: 0.0

**PCI_E_CONVERGENCE**

The energy convergence criterion

Type: float

Default value: 1e-08

**PCI_FAST_EVAR**

Use a fast (sparse) estimate of the energy?

Type: bool

Default value: False

**PCI_FUNCTIONAL**

The functional for determinant coupling importance evaluation

Type: str

Default value: MAX

Allowed values: ['MAX', 'SUM', 'SQUARE', 'SQRT', 'SPECIFY-ORDER']

**PCI_FUNCTIONAL_ORDER**

The functional order of PCI_FUNCTIONAL is SPECIFY-ORDER

Type: float

Default value: 1.0

**PCI_GENERATOR**

The propagation algorithm

Type: str

Default value: WALL-CHEBYSHEV

Allowed values: ['LINEAR', 'QUADRATIC', 'CUBIC', 'QUARTIC', 'POWER', 'TROTTER', 'OLSEN', 'DAVIDSON', 'MITRUSHENKOV', 'EXP-CHEBYSHEV', 'WALL-CHEBYSHEV', 'CHEBYSHEV', 'LANCZOS', 'DL']

**PCI_GUESS_SPAWNING_THRESHOLD**

The determinant importance threshold

Type: float

Default value: -1.0

**PCI_INITIATOR_APPROX**

Use initiator approximation?

Type: bool

Default value: False

**PCI_INITIATOR_APPROX_FACTOR**

The initiator approximation factor

Type: float

Default value: 1.0

**PCI_KRYLOV_ORDER**

The order of Krylov truncation

Type: int

Default value: 5

**PCI_MAXBETA**

The maximum value of beta

Type: float

Default value: 1000.0

**PCI_MAX_DAVIDSON_ITER**

The maximum value of Davidson generator iteration

Type: int

Default value: 12

**PCI_MAX_GUESS_SIZE**

The maximum number of determinants used to form the guess wave function

Type: int

Default value: 10000

**PCI_NROOT**

The number of roots computed

Type: int

Default value: 1

**PCI_PERTURB_ANALYSIS**

Do result perturbation analysis?

Type: bool

Default value: False

**PCI_POST_DIAGONALIZE**

Do a final diagonalization after convergence?

Type: bool

Default value: False

**PCI_PRINT_FULL_WAVEFUNCTION**

Print full wavefunction when finished?

Type: bool

Default value: False

**PCI_REFERENCE_SPAWNING**

Do spawning according to reference?

Type: bool

Default value: False

**PCI_R_CONVERGENCE**

The residual 2-norm convergence criterion

Type: float

Default value: 1.0

**PCI_SCHWARZ_PRESCREENING**

Use schwarz prescreening?

Type: bool

Default value: False

**PCI_SIMPLE_PRESCREENING**

Prescreen the spawning of excitations?

Type: bool

Default value: False

**PCI_SPAWNING_THRESHOLD**

The determinant importance threshold

Type: float

Default value: 0.001

**PCI_STOP_HIGHER_NEW_LOW**

Stop iteration when higher new low detected?

Type: bool

Default value: False

**PCI_SYMM_APPROX_H**

Use Symmetric Approximate Hamiltonian?

Type: bool

Default value: False

**PCI_TAU**

The time step in imaginary time (a.u.)

Type: float

Default value: 1.0

**PCI_USE_INTER_NORM**

Use intermediate normalization?

Type: bool

Default value: False

**PCI_USE_SHIFT**

Use a shift in the exponential?

Type: bool

Default value: False

**PCI_VAR_ESTIMATE**

Estimate variational energy during calculation?

Type: bool

Default value: False

PT2 options
===========

**PT2_MAX_MEM**

Maximum size of the determinant hash (GB)

Type: float

Default value: 1.0

SCI options
===========

**SCI_CORE_EX**

Use core excitation algorithm

Type: bool

Default value: False

**SCI_DIRECT_RDMS**

Computes RDMs without coupling lists?

Type: bool

Default value: False

**SCI_ENFORCE_SPIN_COMPLETE**

Enforce determinant spaces (P and Q) to be spin-complete?

Type: bool

Default value: True

**SCI_ENFORCE_SPIN_COMPLETE_P**

Enforce determinant space P to be spin-complete?

Type: bool

Default value: False

**SCI_EXCITED_ALGORITHM**

The selected CI excited state algorithm

Type: str

Default value: NONE

Allowed values: ['AVERAGE', 'ROOT_ORTHOGONALIZE', 'ROOT_COMBINE', 'MULTISTATE']

**SCI_FIRST_ITER_ROOTS**

Compute all roots on first iteration?

Type: bool

Default value: False

**SCI_MAX_CYCLE**

Maximum number of cycles

Type: int

Default value: 20

**SCI_PREITERATIONS**

Number of iterations to run SA-ACI before SS-ACI

Type: int

Default value: 0

**SCI_PROJECT_OUT_SPIN_CONTAMINANTS**

Project out spin contaminants in Davidson-Liu's algorithm?

Type: bool

Default value: True

**SCI_QUIET_MODE**

Print during ACI procedure?

Type: bool

Default value: False

**SCI_SAVE_FINAL_WFN**

Save final wavefunction to file?

Type: bool

Default value: False

**SCI_TEST_RDMS**

Run test for the RDMs?

Type: bool

Default value: False
