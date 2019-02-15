# -*- coding: utf-8 -*-

def register_forte_options(forte_options):
    register_driver_options(forte_options)
    register_mo_space_info_options(forte_options)
    register_avas_options(forte_options)
    register_cino_options(forte_options)
    register_mrcino_options(forte_options)
    register_integral_options(forte_options)
    register_pt2_options(forte_options)
    register_pci_options(forte_options)
    register_fci_options(forte_options)
    register_aci_options(forte_options)
    register_asci_options(forte_options)
    register_fci_mo_options(forte_options)
    register_active_space_solver_options(forte_options)
    register_dsrg_options(forte_options)
    register_dwms_options(forte_options)

def register_driver_options(forte_options):
    forte_options.add_str('JOB_TYPE', 'NEWDRIVER', [
        'NONE', 'ACI', 'PCI', 'CAS', 'DMRG', 'SR-DSRG', 'SR-DSRG-ACI',
        'SR-DSRG-PCI', 'DSRG-MRPT2', 'DSRG-MRPT3', 'MR-DSRG-PT2',
        'THREE-DSRG-MRPT2', 'SOMRDSRG', 'MRDSRG', 'MRDSRG_SO', 'CASSCF',
        'ACTIVE-DSRGPT2', 'DWMS-DSRGPT2', 'DSRG_MRPT', 'TASKS', 'CC'
    ], 'Specify the job type')

    forte_options.add_str('ACTIVE_SPACE_SOLVER', '', ['FCI', 'ACI'],
                          'Active space solver type') # TODO: why is PCI running even if it is not in this list (Francesco)
    forte_options.add_str('CORRELATION_SOLVER', 'NONE', ['DSRG-MRPT2', 'THREE-DSRG-MRPT2', 'DSRG-MRPT3', 'MRDSRG'],
                          'Dynamical correlation solver type')
    forte_options.add_str('CALC_TYPE', 'SS', ['SS', 'SA', 'MS', 'DWMS'],
                          'The type of computation')

def register_avas_options(forte_options):
    forte_options.add_double("AVAS_SIGMA", 0.98, "Threshold that controls the size of the active space")
    forte_options.add_int("AVAS_NUM_ACTIVE", 0,
                     "The total number of active orbitals. "
                     "If not equal to 0, it takes priority over "
                     "threshold based selection.")
    forte_options.add_int("AVAS_NUM_ACTIVE_OCC", 0,
                     "The number of active occupied orbitals. "
                     "If not equal to 0, it takes priority over "
                     "threshold based selection.")
    forte_options.add_int("AVAS_NUM_ACTIVE_VIR", 0,
                     "The number of active occupied orbitals. "
                     "If not equal to 0, it takes priority over "
                     "threshold based selection.")
    forte_options.add_bool("AVAS_DIAGONALIZE", True,
                      "Diagonalize Socc and Svir?"
                      "This option takes priority over "
                      "threshold based selection.")

def register_cino_options(forte_options):
    forte_options.add_bool("CINO", False, "Do a CINO computation?")

    forte_options.add_str("CINO_TYPE", "CIS", ["CIS", "CISD"], "The type of wave function.")

    forte_options.add_int("CINO_NROOT", 1, "The number of roots computed")

    forte_options.add_array("CINO_ROOTS_PER_IRREP",
                       "The number of excited states per irreducible representation")
    forte_options.add_double("CINO_THRESHOLD", 0.99,
                        "The fraction of NOs to include in the active space")
    forte_options.add_bool("CINO_AUTO", False,
                      "{ass frozen_docc, actice_docc, and restricted_docc?")

def register_mrcino_options(forte_options):
    forte_options.add_bool("MRCINO", False, "Do a MRCINO computation?")

    forte_options.add_str("MRCINO_TYPE", "CIS", ["CIS", "CISD"], "The type of wave function.")

    forte_options.add_int("MRCINO_NROOT", 1, "The number of roots computed")

    forte_options.add_array("MRCINO_ROOTS_PER_IRREP",
                       "The number of excited states per irreducible representation")
    forte_options.add_double("MRCINO_THRESHOLD", 0.99,
                        "The fraction of NOs to include in the active space")
    forte_options.add_bool("MRCINO_AUTO", False,
                      "Allow the users to choose"
                      "whether pass frozen_docc"
                      "actice_docc and restricted_docc"
                      "or not")

def register_mo_space_info_options(forte_options):
    forte_options.add_array(
        "FROZEN_DOCC",
        "Number of frozen occupied orbitals per irrep (in Cotton order)")
    forte_options.add_array(
        "RESTRICTED_DOCC",
        "Number of restricted doubly occupied orbitals per irrep (in Cotton order)"
    )
    forte_options.add_array(
        "ACTIVE", " Number of active orbitals per irrep (in Cotton order)")
    forte_options.add_array(
        "RESTRICTED_UOCC",
        "Number of restricted unoccupied orbitals per irrep (in Cotton order)")
    forte_options.add_array(
        "FROZEN_UOCC",
        "Number of frozen unoccupied orbitals per irrep (in Cotton order)")

    #    /*- Molecular orbitals to swap -
    #     *  Swap mo_1 with mo_2 in irrep symmetry
    #     *  Swap mo_3 with mo_4 in irrep symmetry
    #     *  Format: [irrep, mo_1, mo_2, irrep, mo_3, mo_4]
    #     *          Irrep and MO indices are 1-based (NOT 0-based)!
    #    -*/
    forte_options.add_array(
        "ROTATE_MOS",
        "An array of MOs to swap in the format [irrep, mo_1, mo_2, irrep, mo_3, mo_4]. Irrep and MO indices are 1-based (NOT 0-based)!"
    )

    # Options for state-averaged CASSCF
    forte_options.add_array(
        "AVG_STATE",
        "An array of states [[irrep1, multi1, nstates1], [irrep2, multi2, nstates2], ...]"
    )
    forte_options.add_array(
        "AVG_WEIGHT",
        "An array of weights [[w1_1, w1_2, ..., w1_n], [w2_1, w2_2, ..., w2_n], ...]"
    )
    forte_options.add_array("NROOTPI",
                            "Number of roots per irrep (in Cotton order)")


def register_active_space_solver_options(forte_options):
    forte_options.add_int('NROOT', 1, 'The number of roots computed')
    forte_options.add_int('ROOT', 0,
                          'The root selected for state-specific computations')

def register_pt2_options(forte_options):
    forte_options.add_double("PT2_MAX_MEM", 1.0, " Maximum size of the determinant hash (GB)")

def register_pci_options(forte_options):
    forte_options.add_str("PCI_GENERATOR", "WALL-CHEBYSHEV",
                     ["LINEAR", "QUADRATIC", "CUBIC", "QUARTIC", "POWER", "TROTTER", "OLSEN",
                      "DAVIDSON", "MITRUSHENKOV", "EXP-CHEBYSHEV", "WALL-CHEBYSHEV", "CHEBYSHEV",
                      "LANCZOS", "DL"],
                     "The propagation algorithm")

    forte_options.add_int("PCI_NROOT", 1, "The number of roots computed")

    forte_options.add_double("PCI_SPAWNING_THRESHOLD", 0.001, "The determinant importance threshold")

    forte_options.add_double("PCI_MAX_GUESS_SIZE", 10000,
                        "The maximum number of determinants used to form the "
                        "guess wave function")

    forte_options.add_double("PCI_GUESS_SPAWNING_THRESHOLD", -1, "The determinant importance threshold")

    forte_options.add_double("PCI_ENERGY_ESTIMATE_THRESHOLD", 1.0e-6,
                        "The threshold with which we estimate the variational "
                        "energy. Note that the final energy is always "
                        "estimated exactly.")

    forte_options.add_double("PCI_TAU", 1.0, "The time step in imaginary time (a.u.)")

    forte_options.add_double("PCI_E_CONVERGENCE", 1.0e-8, "The energy convergence criterion")

    forte_options.add_bool("PCI_FAST_EVAR", False, "Use a fast (sparse) estimate of the energy?")

    forte_options.add_double("PCI_EVAR_MAX_ERROR", 0.0, "The max allowed error for variational energy")

    forte_options.add_int("PCI_ENERGY_ESTIMATE_FREQ", 1,
                     "Iterations in between variational estimation of the energy")

    forte_options.add_bool("PCI_ADAPTIVE_BETA", False, "Use an adaptive time step?")

    forte_options.add_bool("PCI_USE_INTER_NORM", False, "Use intermediate normalization?")

    forte_options.add_bool("PCI_USE_SHIFT", False, "Use a shift in the exponential?")

    forte_options.add_bool("PCI_VAR_ESTIMATE", False, "Estimate variational energy during calculation?")

    forte_options.add_bool("PCI_PRINT_FULL_WAVEFUNCTION", False, "Print full wavefunction when finished?")

    forte_options.add_bool("PCI_SIMPLE_PRESCREENING", False, "Prescreen the spawning of excitations?")

    forte_options.add_bool("PCI_DYNAMIC_PRESCREENING", False, "Use dynamic prescreening?")

    forte_options.add_bool("PCI_SCHWARZ_PRESCREENING", False, "Use schwarz prescreening?")

    forte_options.add_bool("PCI_INITIATOR_APPROX", False, "Use initiator approximation?")

    forte_options.add_double("PCI_INITIATOR_APPROX_FACTOR", 1.0, "The initiator approximation factor")

    forte_options.add_bool("PCI_PERTURB_ANALYSIS", False, "Do result perturbation analysis?")

    forte_options.add_bool("PCI_SYMM_APPROX_H", False, "Use Symmetric Approximate Hamiltonian?")

    forte_options.add_bool("PCI_STOP_HIGHER_NEW_LOW", False,
                      "Stop iteration when higher new low detected?")

    forte_options.add_double("PCI_MAXBETA", 1000.0, "The maximum value of beta")

    forte_options.add_int("PCI_MAX_DAVIDSON_ITER", 12,
                     "The maximum value of Davidson generator iteration")

    forte_options.add_int("PCI_DL_COLLAPSE_PER_ROOT", 2,
                     "The number of trial vector to retain after Davidson-Liu collapsing")

    forte_options.add_int("PCI_DL_SUBSPACE_PER_ROOT", 8,
                     "The maxim number of trial Davidson-Liu vectors")

    forte_options.add_int("PCI_CHEBYSHEV_ORDER", 5, "The order of Chebyshev truncation")

    forte_options.add_int("PCI_KRYLOV_ORDER", 5, "The order of Krylov truncation")

    forte_options.add_double("PCI_COLINEAR_THRESHOLD", 1.0e-6, "The minimum norm of orthogonal vector")

    forte_options.add_bool("PCI_REFERENCE_SPAWNING", False, "Do spawning according to reference?")

    forte_options.add_bool("PCI_POST_DIAGONALIZE", False, "Do a final diagonalization after convergence?")

    forte_options.add_str("PCI_FUNCTIONAL", "MAX", ["MAX", "SUM", "SQUARE", "SQRT", "SPECIFY-ORDER"],
                     "The functional for determinant coupling importance evaluation")

    forte_options.add_double("PCI_FUNCTIONAL_ORDER", 1.0,
                        "The functional order of PCI_FUNCTIONAL is SPECIFY-ORDER")



def register_fci_options(forte_options):
    forte_options.add_int('FCI_MAXITER', 30,
                          'Maximum number of iterations for FCI code')
    forte_options.add_int(
        'FCI_MAX_RDM', 1,
        'The number of trial guess vectors to generate per root')
    forte_options.add_bool('FCI_TEST_RDMS', False,
                           'Test the FCI reduced density matrices?')
    forte_options.add_bool('FCI_PRINT_NO', False,
                           'Print the NO from the rdm of FCI')
    forte_options.add_int(
        'FCI_NTRIAL_PER_ROOT', 10,
        'The number of trial guess vectors to generate per root')

def register_aci_options(forte_options):
    forte_options.add_double("ACI_CONVERGENCE", 1e-9, "ACI Convergence threshold")

    forte_options.add_str("ACI_SELECT_TYPE", "AIMED_ENERGY", ['AIMED_AMP', 'AIMED_ENERGY', 'ENERGY', 'AMP'],
                          "The selection type for the Q-space")

    forte_options.add_double("SIGMA", 0.01, "The energy selection threshold for the P space")

    forte_options.add_double("GAMMA", 1.0, "The threshold for the selection of the Q space")

    forte_options.add_double("ACI_PRESCREEN_THRESHOLD", 1e-12, "The SD space prescreening threshold")

    forte_options.add_bool("ACI_PERTURB_SELECT", False, "Type of energy selection")

    forte_options.add_str("ACI_PQ_FUNCTION", "AVERAGE",['AVERAGE','MAX'], "Function of q-space criteria, per root for SA-ACI")

    forte_options.add_str("ACI_EXCITED_ALGORITHM", "ROOT_ORTHOGONALIZE",
            ['AVERAGE','ROOT_ORTHOGONALIZE','ROOT_COMBINE','MULTISTATE'], "The excited state algorithm")

    forte_options.add_int("ACI_SPIN_PROJECTION", 0, """Type of spin projection
     0 - None
     1 - Project initial P spaces at each iteration
     2 - Project only after converged PQ space
     3 - Do 1 and 2""")

    forte_options.add_bool("ACI_ENFORCE_SPIN_COMPLETE", True,
                      "Enforce determinant spaces to be spin-complete?")

    forte_options.add_bool("ACI_PROJECT_OUT_SPIN_CONTAMINANTS", True,
                      "Project out spin contaminants in Davidson-Liu's algorithm?")

    forte_options.add_bool("SPIN_PROJECT_FULL", False,
                      "Project solution in full diagonalization algorithm?")

    forte_options.add_bool("ACI_ADD_AIMED_DEGENERATE", True,
                      "Add degenerate determinants not included in the aimed selection")

    forte_options.add_int("ACI_MAX_CYCLE", 20, "Maximum number of cycles")

    forte_options.add_bool("ACI_QUIET_MODE", False, "Print during ACI procedure?")

    forte_options.add_bool("ACI_STREAMLINE_Q", False, "Do streamlined algorithm?")

    forte_options.add_int("ACI_PREITERATIONS", 0, "Number of iterations to run SA-ACI before SS-ACI")

    forte_options.add_int("ACI_N_AVERAGE", 1, "Number of roots to averag")

    forte_options.add_int("ACI_AVERAGE_OFFSET", 0, "Offset for state averaging")

    forte_options.add_bool("ACI_SAVE_FINAL_WFN", False, "Print final wavefunction to file?")

    forte_options.add_bool("ACI_PRINT_REFS", False, "Print the P space?")

    forte_options.add_int("DL_GUESS_SIZE", 100, "Set the initial guess space size for DL solver")

    forte_options.add_int("N_GUESS_VEC", 10, "Number of guess vectors for Sparse CI solver")

    forte_options.add_double("ACI_NO_THRESHOLD", 0.02, "Threshold for active space prediction")

    forte_options.add_double("ACI_SPIN_TOL", 0.02, "Tolerance for S^2 value")

    # /*- Approximate 1RDM? -*/
    forte_options.add_bool("ACI_APPROXIMATE_RDM", False, "Approximate the RDMs?")

    forte_options.add_bool("ACI_TEST_RDMS", False, "Run test for the RDMs?")

    forte_options.add_bool("ACI_FIRST_ITER_ROOTS", False, "Compute all roots on first iteration?")

    forte_options.add_bool("ACI_PRINT_WEIGHTS", False, "Print weights for active space prediction?")

    forte_options.add_bool("ACI_PRINT_NO", True, "Print the natural orbitals?")

    forte_options.add_bool("ACI_NO", False, "Computes ACI natural orbitals?")

    forte_options.add_bool("MRPT2", False, "Compute full PT2 energy?")

    forte_options.add_bool("UNPAIRED_DENSITY", False, "Compute unpaired electron density?")

    forte_options.add_bool("ACI_ADD_SINGLES", False,
                      "Adds all active single excitations to the final wave function")

    forte_options.add_bool("ESNOS", False, "Compute external single natural orbitals (ESNO)")

    forte_options.add_int("ESNO_MAX_SIZE", 0, "Number of external orbitals to correlate")

    forte_options.add_bool("ACI_LOW_MEM_SCREENING", False, "Use low-memory screening algorithm?")

    forte_options.add_bool("ACI_REF_RELAX", False, "Do reference relaxation in ACI?")

    forte_options.add_bool("ACI_CORE_EX", False, "Use core excitation algorithm")

    forte_options.add_int("ACI_NFROZEN_CORE", 0, "Number of orbitals to freeze for core excitations")

    forte_options.add_int("ACI_ROOTS_PER_CORE", 1, "Number of roots to compute per frozen orbital")

    forte_options.add_bool("ACI_SPIN_ANALYSIS", False, "Do spin correlation analysis?")

    forte_options.add_bool("ACI_RELAXED_SPIN", False,
                      "Do spin correlation analysis for relaxed wave function?")

    forte_options.add_bool("PRINT_IAOS", True, "Print IAOs?")

    forte_options.add_bool("PI_ACTIVE_SPACE", False, "Active space type?")

    forte_options.add_bool("SPIN_MAT_TO_FILE", False, "Save spin correlation matrix to file?")

    forte_options.add_bool("ACI_RELAXED_SPIN", False,
                      "Do spin correlation analysis for relaxed wave function?")

    forte_options.add_bool("PRINT_IAOS", True, "Print IAOs?")

    forte_options.add_bool("PI_ACTIVE_SPACE", False, "Active space type?")

    forte_options.add_bool("SPIN_MAT_TO_FILE", False, "Save spin correlation matrix to file?")

    forte_options.add_str("SPIN_BASIS", "LOCAL", ['LOCAL','IAO','NO','CANONICAL'], "Basis for spin analysis")

    forte_options.add_double("ACI_RELAX_SIGMA", 0.01, "Sigma for reference relaxation")

    forte_options.add_bool("ACI_BATCHED_SCREENING", False, "Control batched screeing?")

    forte_options.add_int("ACI_NBATCH", 1, "Number of batches in screening")

    forte_options.add_int("ACI_MAX_MEM", 1000, "Sets max memory for batching algorithm (MB)")

    forte_options.add_double("ACI_SCALE_SIGMA", 0.5, "Scales sigma in batched algorithm")

    forte_options.add_bool("ACI_DIRECT_RDMS", False, "Computes RDMs without coupling lists?")

    forte_options.add_str("ACI_BATCH_ALG", "HASH",['HASH','VECSORT'], "Algorithm to use for batching")

def register_asci_options(forte_options):
    forte_options.add_double("ASCI_E_CONVERGENCE", 1e-5, "ASCI energy convergence threshold")

    forte_options.add_int("ASCI_MAX_CYCLE", 20, "ASCI MAX Cycle")

    forte_options.add_int("ASCI_TDET", 2000, "ASCI Max det")

    forte_options.add_int("ASCI_CDET", 200, "ASCI Max reference det")

    forte_options.add_double("ASCI_PRESCREEN_THRESHOLD", 1e-12, "ASCI prescreening threshold")

def register_fci_mo_options(forte_options):
    forte_options.add_str("FCIMO_ACTV_TYPE", "COMPLETE", ["COMPLETE", "CIS", "CISD", "DOCI"],
                     "The active space type")

    forte_options.add_bool("FCIMO_CISD_NOHF", True,
                      "Ground state: HF; Excited states: no HF determinant in CISD space")

    forte_options.add_str("FCIMO_IPEA", "NONE", ["NONE", "IP", "EA"], "Generate IP/EA CIS/CISD space")

    forte_options.add_double("FCIMO_PRINT_CIVEC", 0.05, "The printing threshold for CI vectors")

    # forte_options.add_bool("FCIMO_IAO_ANALYSIS", False, "Intrinsic atomic orbital analysis")

def register_integral_options(forte_options):
    forte_options.add_str(
        "INT_TYPE", "CONVENTIONAL",
        ["CONVENTIONAL", "DF", "CHOLESKY", "DISKDF", "DISTDF", "OWNINTEGRALS"],
        "The algorithm used to screen the determinant"
        "- CONVENTIONAL Conventional two-electron integrals"
        "- DF Density fitted two-electron integrals"
        "- CHOLESKY Cholesky decomposed two-electron integrals")
    forte_options.add_double(
        "INTEGRAL_SCREENING", 1.0e-12,
        "The screening threshold for JK builds and DF libraries")
    forte_options.add_double("CHOLESKY_TOLERANCE", 1.0e-6,
                             "The tolerance for cholesky integrals")

    forte_options.add_bool("PRINT_INTS", False,
                           "Print the one- and two-electron integrals?")

def register_dsrg_options(forte_options):

    forte_options.add_str("CORR_LEVEL", "PT2",
                     ["PT2", "PT3", "LDSRG2", "LDSRG2_QC", "LSRG2", "SRG_PT2", "QDSRG2",
                      "LDSRG2_P3", "QDSRG2_P3"],
                     "Correlation level of MR-DSRG (used in mrdsrg code, "
                     "LDSRG2_P3 and QDSRG2_P3 not implemented)")

    forte_options.add_str("SOURCE", "STANDARD",
                     ["STANDARD", "LABS", "DYSON", "AMP", "EMP2", "LAMP", "LEMP2"],
                     "Source operator used in DSRG (AMP, EMP2, LAMP, LEMP2 "
                     "only available in toy code mcsrgpt2)")

    forte_options.add_str("T_ALGORITHM", "DSRG", ["DSRG", "DSRG_NOSEMI", "SELEC", "ISA"],
                     "The way of forming T amplitudes (DSRG_NOSEMI, SELEC, ISA "
                     "only available in toy code mcsrgpt2)")

    forte_options.add_str("H0TH", "FDIAG", ["FDIAG", "FFULL", "FDIAG_VACTV", "FDIAG_VDIAG"],
                     "Different Zeroth-order Hamiltonian of DSRG-MRPT (used in mrdsrg code)")

    forte_options.add_bool("DSRG_DIPOLE", False, "Compute (if true) DSRG dipole moments")

    forte_options.add_int("DSRG_MAXITER", 50, "Max iterations for nonperturbative MR-DSRG amplitudes update")

    forte_options.add_double("R_CONVERGENCE", 1.0e-6, "Residue convergence criteria for amplitudes")

    forte_options.add_str("RELAX_REF", "NONE", ["NONE", "ONCE", "TWICE", "ITERATE"],
                     "Relax the reference for MR-DSRG (used in dsrg-mrpt2/3, mrdsrg)")

    forte_options.add_int("MAXITER_RELAX_REF", 15, "Max macro iterations for DSRG reference relaxation")

    forte_options.add_int("TAYLOR_THRESHOLD", 3, "DSRG Taylor expansion threshold for small denominator")

    forte_options.add_int("NTAMP", 15, "Number of largest amplitudes printed in the summary")

    forte_options.add_double("INTRUDER_TAMP", 0.10,
                        "T threshold for amplitudes considered as intruders for warning")

    forte_options.add_str("DSRG_TRANS_TYPE", "UNITARY", ["UNITARY", "CC"], "DSRG transformation type")

    forte_options.add_str("SMART_DSRG_S", "DSRG_S",
                     ["DSRG_S", "MIN_DELTA1", "MAX_DELTA1", "DAVG_MIN_DELTA1", "DAVG_MAX_DELTA1"],
                     "Automatically adjust the flow parameter according to denominators")

    forte_options.add_bool("PRINT_TIME_PROFILE", False, "Print detailed timings in dsrg-mrpt3")

    forte_options.add_str("DSRG_MULTI_STATE", "SA_FULL", ["SA_FULL", "SA_SUB", "MS", "XMS"],
                     """Multi-state DSRG options (MS and XMS recouple states after single-state computations)
                     - Multi-State DSRG options
                        - State-average approach
                          - SA_SUB:  form H_MN = <M|Hbar|N>; M, N are CAS states of interest
                          - SA_FULL: redo a CASCI
                        - Multi-state approach (currently only for MRPT2)
                          - MS:  form 2nd-order Heff_MN = <M|H|N> + 0.5 * [<M|(T_M)^+ H|N> + <M|H T_N|N>]
                          - XMS: rotate references such that <M|F|N> is diagonal before MS procedure  """
                     )

    forte_options.add_bool("FORM_HBAR3", False,
                      "Form 3-body Hbar (only used in dsrg-mrpt2 with SA_SUB for testing)")

    forte_options.add_bool("FORM_MBAR3", False,
                      "Form 3-body mbar (only used in dsrg-mrpt2 for testing)")

    forte_options.add_bool("DSRGPT", True,
                      "Renormalize (if true) the integrals for purturbitive calculation (only used in toy code mcsrgpt2)")

    forte_options.add_str("INTERNAL_AMP", "NONE", ["NONE", "SINGLES_DOUBLES", "SINGLES", "DOUBLES"],
                     "Include internal amplitudes for VCIS/VCISD-DSRG acording to excitation level")


    forte_options.add_str("INTERNAL_AMP_SELECT", "AUTO", ["AUTO", "ALL", "OOVV"],
                     """Excitation types considered when internal amplitudes are included
                     - Select only part of the asked internal amplitudes (IAs) in
                       V-CIS/CISD
                        - AUTO: all IAs that changes excitations (O->V; OO->VV, OO->OV,
                       OV->VV)
                        - ALL:  all IAs (O->O, V->V, O->V; OO->OO, OV->OV, VV->VV, OO->VV,
                       OO->OV, OV->VV)
                        - OOVV: pure external (O->V; OO->VV) -*/
                     """)

    forte_options.add_str("T1_AMP", "DSRG", ["DSRG", "SRG", "ZERO"],
                     "The way of forming T1 amplitudes (used in toy code mcsrgpt2)")

    forte_options.add_double("ISA_B", 0.02,
                        "Intruder state avoidance parameter "
                        "when use ISA to form amplitudes (only "
                        "used in toy code mcsrgpt2)")

    forte_options.add_str("CCVV_SOURCE", "NORMAL", ["ZERO", "NORMAL"],
                     "Definition of source oporator: special treatment for the CCVV term in DSRG-MRPT2 (used "
                     "in three-dsrg-mrpt2 code)")

    forte_options.add_str("CCVV_ALGORITHM", "FLY_AMBIT",
                     ["CORE", "FLY_AMBIT", "FLY_LOOP", "BATCH_CORE", "BATCH_VIRTUAL",
                      "BATCH_CORE_GA", "BATCH_VIRTUAL_GA", "BATCH_VIRTUAL_MPI", "BATCH_CORE_MPI",
                      "BATCH_CORE_REP", "BATCH_VIRTUAL_REP"],
                     "Algorithm to compute the CCVV term in DSRG-MRPT2 (only "
                     "used in three-dsrg-mrpt2 code)")

    forte_options.add_bool("AO_DSRG_MRPT2", False, "Do AO-DSRG-MRPT2 if true (not available)")

    forte_options.add_int("CCVV_BATCH_NUMBER", -1, "Batches for CCVV_ALGORITHM")

    forte_options.add_bool("DSRG_MRPT2_DEBUG", False, "Excssive printing for three-dsrg-mrpt2")

    forte_options.add_str("THREEPDC_ALGORITHM", "CORE", ["CORE", "BATCH"],
                     "Algorithm for evaluating 3-body cumulants in three-dsrg-mrpt2")

    forte_options.add_bool("THREE_MRPT2_TIMINGS", False,
                      "Detailed printing (if true) in three-dsrg-mrpt2")

    forte_options.add_bool("PRINT_DENOM2", False,
                      "Print (if true) (1 - exp(-2*s*D)) / D, renormalized denominators in DSRG-MRPT2")

    forte_options.add_bool("DSRG_HBAR_SEQ", False, "Evaluate H_bar sequentially if true")

    forte_options.add_bool("DSRG_NIVO", False,
                      "NIVO approximation: Omit tensor blocks with >= 3 virtual indices if true")

    forte_options.add_bool("PRINT_1BODY_EVALS", False, "Print eigenvalues of 1-body effective H")

def register_dwms_options(forte_options):
    forte_options.add_double("DWMS_ZETA", 0.0, """Automatic Gaussian width cutoff for the density weights
          Weights of state α:
             Wi = exp(-ζ * (Eα - Ei)^2) / sum_j exp(-ζ * (Eα - Ej)^2)
          Energies (Eα, Ei, Ej) can be CASCI or SA-DSRG-PT2/3 energies.
        """)

    forte_options.add_str("DWMS_CORRLV", "PT2", ["PT2", "PT3"], "DWMS-DSRG-PT level")


    forte_options.add_str("DWMS_REFERENCE", "CASCI", ["CASCI", "PT2", "PT3", "PT2D"],
                     """Energies to compute dynamic weights and CI vectors to do multi-state
                     - Using what energies to compute the weight and what CI vectors to do multi state
                      CAS: CASCI energies and CI vectors
                      PT2: SA-DSRG-PT2 energies and SA-DSRG-PT2/CASCI vectors
                      PT3: SA-DSRG-PT3 energies and SA-DSRG-PT3/CASCI vectors
                      PT2D: Diagonal SA-DSRG-PT2c effective Hamiltonian elements and original CASCI vectors -*/
                     """)

    forte_options.add_str("DWMS_ALGORITHM", "SA", ["MS", "XMS", "SA", "XSA", "SH-0", "SH-1"],
                     """DWMS algorithms
                        - SA: state average Hαβ = 0.5 * ( <α|Hbar(β)|β> + <β|Hbar(α)|α> )
                        - XSA: extended state average (rotate Fαβ to a diagonal form)
                        - MS: multi-state (single-state single-reference)
                        - XMS: extended multi-state (single-state single-reference)

                       To Be Deprecated:
                        - SH-0: separated diagonalizations, non-orthogonal final solutions
                        - SH-1: separated diagonalizations, orthogonal final solutions -*/
                     """)

    forte_options.add_bool("DWMS_DELTA_AMP", False,
                      "Consider (if true) amplitudes difference between states X(αβ) = A(β) - A(α) in SA "
                      "algorithm, testing in non-DF DSRG-MRPT2")

    forte_options.add_bool("DWMS_ITERATE", False,
                      "Iterative update the reference CI coefficients in SA "
                      "algorithm, testing in non-DF DSRG-MRPT2")

    forte_options.add_int("DWMS_MAXITER", 10,
                     "Max number of iteration in the update of the reference CI coefficients in SA "
                     "algorithm, testing in non-DF DSRG-MRPT2")

    forte_options.add_double("DWMS_E_CONVERGENCE", 1.0e-7,
                        "Energy convergence criteria for DWMS iteration")
