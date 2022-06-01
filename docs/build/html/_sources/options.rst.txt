.. _`sec:options`:

List of Forte options
=====================

.. sectionauthor:: Francesco A. Evangelista

**ACI_ADD_AIMED_DEGENERATE**

Add degenerate determinants not included in the aimed selection

* Type: Boolean

* Default value: True


**ACI_ADD_EXTERNAL_EXCITATIONS**

Adds external single excitations to the final wave function

* Type: Boolean

* Default value: False


**ACI_ADD_SINGLES**

Adds all active single excitations to the final wave function

* Type: Boolean

* Default value: False


**ACI_APPROXIMATE_RDM**

Approximate the RDMs

* Type: Boolean

* Default value: False


**ACI_AVERAGE_OFFSET**

Offset for state averaging

* Type: Integer

* Default value: 0


**ACI_BATCHED_SCREENING**

Control batched screeing

* Type: Boolean

* Default value: False


**ACI_CONVERGENCE**

ACI Convergence threshold

* Type: Double

* Default value: 0.000000


**ACI_DIRECT_RDMS**

Computes RDMs without coupling lists

* Type: Boolean

* Default value: False


**ACI_ENFORCE_SPIN_COMPLETE**

Enforce determinant spaces to be spin-complete

* Type: Boolean

* Default value: True


**ACI_EXCITED_ALGORITHM**

The excited state algorithm

* Type: String

* Default value: ROOT_ORTHOGONALIZE


**ACI_EXTERNAL_EXCITATION_ORDER**

Order of external excitations to add

* Type: String

* Default value: SINGLES


**ACI_EXTERNAL_EXCITATION_TYPE**

Type of external excitations to add

* Type: String

* Default value: ALL


**ACI_EX_TYPE**

Type of excited state to compute

* Type: String

* Default value: CONV


**ACI_FIRST_ITER_ROOTS**

Compute all roots on first iteration?

* Type: Boolean

* Default value: False


**ACI_INITIAL_SPACE**

The initial reference space

* Type: String

* Default value: CAS


**ACI_LOW_MEM_SCREENING**

Use low-memory screening algorithm

* Type: Boolean

* Default value: False


**ACI_MAX_CYCLE**

Maximum number of cycles

* Type: Integer

* Default value: 20


**ACI_MAX_MEM**

Sets max memory for batching algorithm (MB)

* Type: Integer

* Default value: 1000


**ACI_MAX_RDM**

Order of RDM to compute

* Type: Integer

* Default value: 1


**ACI_MAX_RDM**

Order of RDM to compute

* Type: Integer

* Default value: 1


**ACI_MAX_RDM**

Order of RDM to compute

* Type: Integer

* Default value: 1


**ACI_NBATCH**

Number of batches in screening

* Type: Integer

* Default value: 1


**ACI_NFROZEN_CORE**

Number of orbitals to freeze for core excitations

* Type: Integer

* Default value: 0


**ACI_NO**

Computes ACI natural orbitals

* Type: Boolean

* Default value: False


**ACI_NO_THRESHOLD**

Threshold for active space prediction

* Type: Double

* Default value: 0.020000


**ACI_NROOT**

Number of roots for ACI computation

* Type: Integer

* Default value: 1


**ACI_N_AVERAGE**

Number of roots to averag

* Type: Integer

* Default value: 1


**ACI_PERTURB_SELECT**

Type of energy selection

* Type: Boolean

* Default value: False


**ACI_PQ_FUNCTION**

Function for SA-ACI

* Type: String

* Default value: AVERAGE


**ACI_PREITERATIONS**

Number of iterations to run SA-ACI before SS-ACI

* Type: Integer

* Default value: 0


**ACI_PRESCREEN_THRESHOLD**

The SD space prescreening threshold

* Type: Double

* Default value: 0.000000


**ACI_PRINT_NO**

Print the natural orbitals

* Type: Boolean

* Default value: True


**ACI_PRINT_REFS**

Print the P space

* Type: Boolean

* Default value: False


**ACI_PRINT_WEIGHTS**

Print weights for active space prediction

* Type: Boolean

* Default value: False


**ACI_PROJECT_OUT_SPIN_CONTAMINANTS**

Project out spin contaminants in Davidson-Liu's algorithm

* Type: Boolean

* Default value: True


**ACI_QUIET_MODE**

Print during ACI procedure

* Type: Boolean

* Default value: False


**ACI_REF_RELAX**

Do reference relaxation in ACI

* Type: Boolean

* Default value: False


**ACI_RELAX_SIGMA**

Sigma for reference relaxation

* Type: Double

* Default value: 0.010000


**ACI_ROOT**

Root for single-state computations

* Type: Integer

* Default value: 0


**ACI_ROOTS_PER_CORE**

Number of roots to compute per frozen occupation

* Type: Integer

* Default value: 1


**ACI_SAVE_FINAL_WFN**

Print final wavefunction to file

* Type: Boolean

* Default value: False


**ACI_SCALE_SIGMA**

Scales sigma in batched algorithm

* Type: Double

* Default value: 0.500000


**ACI_SELECT_TYPE**

The energy selection criteria

* Type: String

* Default value: AIMED_ENERGY


**ACI_SIZE_CORRECTION**

Perform size extensivity correction

* Type: String

* Default value: 


**ACI_SPIN_ANALYSIS**

Do spin correlation analysis

* Type: Boolean

* Default value: False


**ACI_SPIN_PROJECTION**

Type of spin projection

* Type: Integer

* Default value: 0


**ACI_SPIN_TOL**

Tolerance for S^2 value

* Type: Double

* Default value: 0.020000


**ACI_STREAMLINE_Q**

Do streamlined algorithm

* Type: Boolean

* Default value: False


**ACI_TEST_RDMS**

Run test for the RDMs

* Type: Boolean

* Default value: False


**ACTIVE_REF_TYPE**

Initial guess for active space wave functions

* Type: String

* Default value: CAS


**AO_DSRG_MRPT2**

Do AO-DSRG-MRPT2 if true (not available)

* Type: Boolean

* Default value: False


**AVAS_DIAGONALIZE**

Allow the users to specifydiagonalization of Socc and SvirIt takes priority over thethreshold based selection.

* Type: Boolean

* Default value: True


**AVAS_NUM_ACTIVE**

Allows the user to specify the total number of active orbitals. It takes priority over the threshold based selection.

* Type: Integer

* Default value: 0


**AVAS_NUM_ACTIVE_OCC**

Allows the user to specify the number of active occupied orbitals. It takes priority over the threshold based selection.

* Type: Integer

* Default value: 0


**AVAS_NUM_ACTIVE_VIR**

Allows the user to specify the number of active occupied orbitals. It takes priority over the threshold based selection.

* Type: Integer

* Default value: 0


**AVAS_SIGMA**

Threshold that controls the size of the active space

* Type: Double

* Default value: 0.980000


**CCVV_ALGORITHM**

Algorithm to compute the CCVV term in DSRG-MRPT2 (only used in three-dsrg-mrpt2 code)

* Type: String

* Default value: FLY_AMBIT

* Allowed values: CORE, FLY_AMBIT, FLY_LOOP, BATCH_CORE, BATCH_VIRTUAL, BATCH_CORE_GA, BATCH_VIRTUAL_GA, BATCH_VIRTUAL_MPI, BATCH_CORE_MPI, BATCH_CORE_REP, BATCH_VIRTUAL_REP
**CCVV_BATCH_NUMBER**

Batches for CCVV_ALGORITHM

* Type: Integer

* Default value: -1


**CCVV_SOURCE**

Special treatment for the CCVV term in DSRG-MRPT2 (used in three-dsrg-mrpt2 code)

* Type: String

* Default value: NORMAL

* Allowed values: ZERO, NORMAL
**CHOLESKY_TOLERANCE**

The tolerance for cholesky integrals

* Type: Double

* Default value: 0.000001


**CINO**

Do a CINO computation?

* Type: Boolean

* Default value: False


**CINO_AUTO**

Allow the users to choosewhether pass frozen_doccactice_docc and restricted_doccor not

* Type: Boolean

* Default value: False


**CINO_NROOT**

The number of roots computed

* Type: Integer

* Default value: 1


**CINO_ROOTS_PER_IRREP**

The number of excited states per irreducible representation

* Type: Array

* Default value: []


**CINO_THRESHOLD**

The fraction of NOs to include in the active space

* Type: Double

* Default value: 0.990000


**CINO_TYPE**

The type of wave function.

* Type: String

* Default value: CIS

* Allowed values: CIS, CISD
**CORR_LEVEL**

Correlation level of MR-DSRG (used in mrdsrg code, LDSRG2_P3 and QDSRG2_P3 not implemented)

* Type: String

* Default value: PT2

* Allowed values: PT2, PT3, LDSRG2, LDSRG2_QC, LSRG2, SRG_PT2, QDSRG2, LDSRG2_P3, QDSRG2_P3
**DL_GUESS_SIZE**

Set the initial guess space size for DL solver

* Type: Integer

* Default value: 100


**DSRGPT**

Renormalize (if true) the integrals (only used in toy code mcsrgpt2)

* Type: Boolean

* Default value: True


**DSRG_DIPOLE**

Compute (if true) DSRG dipole moments

* Type: Boolean

* Default value: False


**DSRG_HBAR_SEQ**

Evaluate H_bar sequentially if true

* Type: Boolean

* Default value: False


**DSRG_MAXITER**

Max iterations for MR-DSRG amplitudes update

* Type: Integer

* Default value: 50


**DSRG_MRPT2_DEBUG**

Excssive printing for three-dsrg-mrpt2

* Type: Boolean

* Default value: False


**DSRG_MULTI_STATE**

Multi-state DSRG options (MS and XMS recouple states after single-state computations)

* Type: String

* Default value: SA_FULL

* Allowed values: SA_FULL, SA_SUB, MS, XMS
**DSRG_OMIT_V3**

Omit blocks with >= 3 virtual indices if true

* Type: Boolean

* Default value: False


**DSRG_TRANS_TYPE**

DSRG transformation type

* Type: String

* Default value: UNITARY

* Allowed values: UNITARY, CC
**DWMS_ALGORITHM**

DWMS algorithms

* Type: String

* Default value: DWMS-0

* Allowed values: DWMS-0, DWMS-1, DWMS-AVG0, DWMS-AVG1
**DWMS_ZETA**

Gaussian width cutoff for the density weights

* Type: Double

* Default value: 0.000000


**ESNOS**

Compute external single natural orbitals

* Type: Boolean

* Default value: False


**ESNO_MAX_SIZE**

Number of external orbitals to correlate

* Type: Integer

* Default value: 0


**FCIMO_ACTV_TYPE**

The active space type

* Type: String

* Default value: COMPLETE

* Allowed values: COMPLETE, CIS, CISD, DOCI
**FCIMO_CISD_NOHF**

Ground state: HF; Excited states: no HF determinant in CISD space

* Type: Boolean

* Default value: True


**FCIMO_IAO_ANALYSIS**

Intrinsic atomic orbital analysis

* Type: Boolean

* Default value: False


**FCIMO_IPEA**

Generate IP/EA CIS/CISD space

* Type: String

* Default value: NONE

* Allowed values: NONE, IP, EA
**FCIMO_LOCALIZE_ACTV**

Localize active orbitals before computation

* Type: Boolean

* Default value: False


**FCIMO_PRINT_CIVEC**

The printing threshold for CI vectors

* Type: Double

* Default value: 0.050000


**FCI_MAXITER**

Maximum number of iterations for FCI code

* Type: Integer

* Default value: 30


**FCI_MAX_RDM**

The number of trial guess vectors to generate per root

* Type: Integer

* Default value: 1


**FCI_NROOT**

The number of roots computed

* Type: Integer

* Default value: 1


**FCI_NTRIAL_PER_ROOT**

The number of trial guess vectors to generate per root

* Type: Integer

* Default value: 10


**FCI_PRINT_NO**

Print the NO from the rdm of FCI

* Type: Boolean

* Default value: False


**FCI_ROOT**

The root selected for state-specific computations

* Type: Integer

* Default value: 0


**FCI_TEST_RDMS**

Test the FCI reduced density matrices?

* Type: Boolean

* Default value: False


**FORM_HBAR3**

Form 3-body Hbar (only used in dsrg-mrpt2 with SA_SUB for testing)

* Type: Boolean

* Default value: False


**FORM_MBAR3**

Form 3-body mbar (only used in dsrg-mrpt2 for testing)

* Type: Boolean

* Default value: False


**GAMMA**

The reference space selection threshold

* Type: Double

* Default value: 1.000000


**H0TH**

Zeroth-order Hamiltonian of DSRG-MRPT (used in mrdsrg code)

* Type: String

* Default value: FDIAG

* Allowed values: FDIAG, FFULL, FDIAG_VACTV, FDIAG_VDIAG
**INTEGRAL_SCREENING**

The screening for JK builds and DF libraries

* Type: Double

* Default value: 0.000000


**INTERNAL_AMP**

Include internal amplitudes for VCIS/VCISD-DSRG

* Type: String

* Default value: NONE

* Allowed values: NONE, SINGLES_DOUBLES, SINGLES, DOUBLES
**INTERNAL_AMP_SELECT**

Excitation types considered when internal amplitudes are included

* Type: String

* Default value: AUTO

* Allowed values: AUTO, ALL, OOVV
**INTRUDER_TAMP**

Threshold for amplitudes considered as intruders for warning

* Type: Double

* Default value: 0.100000


**INT_TYPE**

The integral type

* Type: String

* Default value: CONVENTIONAL

* Allowed values: CONVENTIONAL, DF, CHOLESKY, DISKDF, DISTDF, ALL, OWNINTEGRALS
**ISA_B**

Intruder state avoidance parameter when use ISA to form amplitudes (only used in toy code mcsrgpt2)

* Type: Double

* Default value: 0.020000


**JOB_TYPE**

Specify the job type

* Type: String

* Default value: NONE

* Allowed values: NONE, ACI, PCI, CAS, DMRG, SR-DSRG, SR-DSRG-ACI, SR-DSRG-PCI, TENSORSRG, TENSORSRG-CI, DSRG-MRPT2, DSRG-MRPT3, MR-DSRG-PT2, THREE-DSRG-MRPT2, SOMRDSRG, MRDSRG, MRDSRG_SO, CASSCF, ACTIVE-DSRGPT2, DWMS-DSRGPT2, DSRG_MRPT, TASKS, CC, NOJOB, DOCUMENTATION
**MAXITER_RELAX_REF**

Max macro iterations for DSRG reference relaxation

* Type: Integer

* Default value: 15


**MINAO_BASIS**

The basis used to define an orbital subspace

* Type: String

* Default value: STO-3G


**MRCINO**

Do a MRCINO computation?

* Type: Boolean

* Default value: False


**MRCINO_AUTO**

Allow the users to choosewhether pass frozen_doccactice_docc and restricted_doccor not

* Type: Boolean

* Default value: False


**MRCINO_NROOT**

The number of roots computed

* Type: Integer

* Default value: 1


**MRCINO_ROOTS_PER_IRREP**

The number of excited states per irreducible representation

* Type: Array

* Default value: []


**MRCINO_THRESHOLD**

The fraction of NOs to include in the active space

* Type: Double

* Default value: 0.990000


**MRCINO_TYPE**

The type of wave function.

* Type: String

* Default value: CIS

* Allowed values: CIS, CISD
**MRPT2**

Compute full PT2 energy

* Type: Boolean

* Default value: False


**MS**

Projection of spin onto the z axis

* Type: Double

* Default value: 0.000000


**NTAMP**

Number of amplitudes printed in the summary

* Type: Integer

* Default value: 15


**N_GUESS_VEC**

Number of guess vectors for Sparse CI solver

* Type: Integer

* Default value: 10


**PCI_ADAPTIVE_BETA**

Use an adaptive time step?

* Type: Boolean

* Default value: False


**PCI_CHEBYSHEV_ORDER**

The order of Chebyshev truncation

* Type: Integer

* Default value: 5


**PCI_COLINEAR_THRESHOLD**

The minimum norm of orthogonal vector

* Type: Double

* Default value: 0.000001


**PCI_DL_COLLAPSE_PER_ROOT**

The number of trial vector to retain after Davidson-Liu collapsing

* Type: Integer

* Default value: 2


**PCI_DL_SUBSPACE_PER_ROOT**

The maxim number of trial Davidson-Liu vectors

* Type: Integer

* Default value: 8


**PCI_DYNAMIC_PRESCREENING**

Use dynamic prescreening

* Type: Boolean

* Default value: False


**PCI_ENERGY_ESTIMATE_FREQ**

Iterations in between variational estimation of the energy

* Type: Integer

* Default value: 1


**PCI_ENERGY_ESTIMATE_THRESHOLD**

The threshold with which we estimate the variational energy. Note that the final energy is always estimated exactly.

* Type: Double

* Default value: 0.000001


**PCI_EVAR_MAX_ERROR**

The max allowed error for variational energy

* Type: Double

* Default value: 0.000000


**PCI_E_CONVERGENCE**

The energy convergence criterion

* Type: Double

* Default value: 0.000000


**PCI_FAST_EVAR**

Use a fast (sparse) estimate of the energy

* Type: Boolean

* Default value: False


**PCI_FUNCTIONAL**

The functional for determinant coupling importance evaluation

* Type: String

* Default value: MAX

* Allowed values: MAX, SUM, SQUARE, SQRT, SPECIFY-ORDER
**PCI_FUNCTIONAL_ORDER**

The functional order of PCI_FUNCTIONAL is SPECIFY-ORDER

* Type: Double

* Default value: 1.000000


**PCI_GENERATOR**

The propagation algorithm

* Type: String

* Default value: WALL-CHEBYSHEV

* Allowed values: LINEAR, QUADRATIC, CUBIC, QUARTIC, POWER, TROTTER, OLSEN, DAVIDSON, MITRUSHENKOV, EXP-CHEBYSHEV, WALL-CHEBYSHEV, CHEBYSHEV, LANCZOS, DL
**PCI_GUESS_SPAWNING_THRESHOLD**

The determinant importance threshold

* Type: Double

* Default value: -1.000000


**PCI_INITIATOR_APPROX**

Use initiator approximation

* Type: Boolean

* Default value: False


**PCI_INITIATOR_APPROX_FACTOR**

The initiator approximation factor

* Type: Double

* Default value: 1.000000


**PCI_KRYLOV_ORDER**

The order of Krylov truncation

* Type: Integer

* Default value: 5


**PCI_MAXBETA**

The maximum value of beta

* Type: Double

* Default value: 1000.000000


**PCI_MAX_DAVIDSON_ITER**

The maximum value of Davidson generator iteration

* Type: Integer

* Default value: 12


**PCI_MAX_GUESS_SIZE**

The maximum number of determinants used to form the guess wave function

* Type: Double

* Default value: 10000.000000


**PCI_NROOT**

The number of roots computed

* Type: Integer

* Default value: 1


**PCI_PERTURB_ANALYSIS**

Do result perturbation analysis

* Type: Boolean

* Default value: False


**PCI_POST_DIAGONALIZE**

Do a post diagonalization?

* Type: Boolean

* Default value: False


**PCI_PRINT_FULL_WAVEFUNCTION**

Print full wavefunction when finish

* Type: Boolean

* Default value: False


**PCI_REFERENCE_SPAWNING**

Do spawning according to reference

* Type: Boolean

* Default value: False


**PCI_SCHWARZ_PRESCREENING**

Use schwarz prescreening

* Type: Boolean

* Default value: False


**PCI_SIMPLE_PRESCREENING**

Prescreen the spawning of excitations

* Type: Boolean

* Default value: False


**PCI_SPAWNING_THRESHOLD**

The determinant importance threshold

* Type: Double

* Default value: 0.001000


**PCI_STOP_HIGHER_NEW_LOW**

Stop iteration when higher new low detected

* Type: Boolean

* Default value: False


**PCI_SYMM_APPROX_H**

Use Symmetric Approximate Hamiltonian

* Type: Boolean

* Default value: False


**PCI_TAU**

The time step in imaginary time (a.u.)

* Type: Double

* Default value: 1.000000


**PCI_USE_INTER_NORM**

Use intermediate normalization

* Type: Boolean

* Default value: False


**PCI_USE_SHIFT**

Use a shift in the exponential

* Type: Boolean

* Default value: False


**PCI_VAR_ESTIMATE**

Estimate variational energy during calculation

* Type: Boolean

* Default value: False


**PI_ACTIVE_SPACE**

Active space type

* Type: Boolean

* Default value: False


**PRINT_1BODY_EVALS**

Print eigenvalues of 1-body effective H

* Type: Boolean

* Default value: False


**PRINT_DENOM2**

Print (if true) renormalized denominators in DSRG-MRPT2

* Type: Boolean

* Default value: False


**PRINT_IAOS**

Print IAOs

* Type: Boolean

* Default value: True


**PRINT_INTS**

Print the one- and two-electron integrals?

* Type: Boolean

* Default value: False


**PRINT_TIME_PROFILE**

Print detailed timings in dsrg-mrpt3

* Type: Boolean

* Default value: False


**PT2_MAX_MEM**

 Maximum size of the determinant hash (GB)

* Type: Double

* Default value: 1.000000


**RELAX_REF**

Relax the reference for MR-DSRG (used in dsrg-mrpt2/3, mrdsrg)

* Type: String

* Default value: NONE

* Allowed values: NONE, ONCE, TWICE, ITERATE
**R_CONVERGENCE**

Convergence criteria for amplitudes

* Type: Double

* Default value: 0.000001


**SAVE_FINAL_WFN**

Save the final wavefunction to a file

* Type: Boolean

* Default value: False


**SIGMA**

The energy selection threshold

* Type: Double

* Default value: 0.010000


**SMART_DSRG_S**

Automatic adjust the flow parameter according to denominators

* Type: String

* Default value: DSRG_S

* Allowed values: DSRG_S, MIN_DELTA1, MAX_DELTA1, DAVG_MIN_DELTA1, DAVG_MAX_DELTA1
**SOURCE**

Source operator used in DSRG (AMP, EMP2, LAMP, LEMP2 only available in toy code mcsrgpt2)

* Type: String

* Default value: STANDARD

* Allowed values: STANDARD, LABS, DYSON, AMP, EMP2, LAMP, LEMP2
**SPIN_BASIS**

Basis for spin analysis

* Type: String

* Default value: LOCAL


**SPIN_MAT_TO_FILE**

Save spin correlation matrix to file

* Type: Boolean

* Default value: False


**SPIN_PROJECT_FULL**

Project solution in full diagonalization algorithm

* Type: Boolean

* Default value: False


**SUBSPACE**

A list of orbital subspaces

* Type: Array

* Default value: []


**T1_AMP**

The way of forming T1 amplitudes (used in toy code mcsrgpt2)

* Type: String

* Default value: DSRG

* Allowed values: DSRG, SRG, ZERO
**TAYLOR_THRESHOLD**

Taylor expansion threshold for small denominator

* Type: Integer

* Default value: 3


**THREEPDC_ALGORITHM**

Algorithm for evaluating 3-body cumulants in three-dsrg-mrpt2

* Type: String

* Default value: CORE

* Allowed values: CORE, BATCH
**THREE_MRPT2_TIMINGS**

Detailed printing (if true) in three-dsrg-mrpt2

* Type: Boolean

* Default value: False


**T_ALGORITHM**

The way of forming amplitudes (DSRG_NOSEMI, SELEC, ISA only available in toy code mcsrgpt2)

* Type: String

* Default value: DSRG

* Allowed values: DSRG, DSRG_NOSEMI, SELEC, ISA
**UNPAIRED_DENSITY**

Compute unpaired electron density

* Type: Boolean

* Default value: False

