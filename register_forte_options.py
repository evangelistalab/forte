# -*- coding: utf-8 -*-

def register_forte_options(forte_options):
    register_driver_options(forte_options)
    register_mo_space_info_options(forte_options)
    register_integral_options(forte_options)
    register_fci_options(forte_options)
    register_active_space_solver_options(forte_options)

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
    # The selection type for the Q-space
    foptions.add_str("ACI_SELECT_TYPE", "AIMED_ENERGY", "The selection type for the Q-space")
    /*-Threshold for the selection of the P space -*/
    foptions.add_double("SIGMA", 0.01, "The energy selection threshold for the P space");
    /*- The threshold for the selection of the Q space -*/
    foptions.add_double("GAMMA", 1.0, "The reference space selection threshold");
    foptions.add_double("ACI_PRESCREEN_THRESHOLD", 1e-12, "The SD space prescreening threshold")
    /*- The type of selection parameters to use*/
    foptions.add_bool("ACI_PERTURB_SELECT", false, "Type of energy selection");
    /*Function of q-space criteria, per root*/
    foptions.add_str("ACI_PQ_FUNCTION", "AVERAGE", "Function for SA-ACI");
    /* Method to calculate excited state */
    foptions.add_str("ACI_EXCITED_ALGORITHM", "ROOT_ORTHOGONALIZE", "The excited state algorithm");
    foptions.add_int("ACI_SPIN_PROJECTION", 0, """Type of spin projection
     0 - None
     1 - Project initial P spaces at each iteration
     2 - Project only after converged PQ space
     3 - Do 1 and 2""")
    /*- Add determinants to enforce spin-complete set? -*/
    foptions.add_bool("ACI_ENFORCE_SPIN_COMPLETE", true,
                      "Enforce determinant spaces to be spin-complete");
    /*- Project out spin contaminants in Davidson-Liu's algorithm? -*/
    foptions.add_bool("ACI_PROJECT_OUT_SPIN_CONTAMINANTS", true,
                      "Project out spin contaminants in Davidson-Liu's algorithm");
    /*- Project solution in full diagonalization algorithm -*/
    foptions.add_bool("SPIN_PROJECT_FULL", false,
                      "Project solution in full diagonalization algorithm");
    /*- Add "degenerate" determinants not included in the aimed selection?
     * -*/
    foptions.add_bool("ACI_ADD_AIMED_DEGENERATE", true,
                      "Add degenerate determinants not included in the aimed selection");
    /*- Perform size extensivity correction -*/
    foptions.add_str("ACI_SIZE_CORRECTION", "", "Perform size extensivity correction");
    /*- Sets the maximum cycle -*/
    foptions.add_int("ACI_MAX_CYCLE", 20, "Maximum number of cycles");
    /*- Control print level -*/
    foptions.add_bool("ACI_QUIET_MODE", false, "Print during ACI procedure");
    /*- Control streamlining -*/
    foptions.add_bool("ACI_STREAMLINE_Q", false, "Do streamlined algorithm");
    /*- Initial reference wavefunction -*/
    foptions.add_str("ACI_INITIAL_SPACE", "CAS", "The initial reference space");
    /*- Number of iterations to run SA-ACI before SS-ACI -*/
    foptions.add_int("ACI_PREITERATIONS", 0, "Number of iterations to run SA-ACI before SS-ACI");
    /*- Number of roots to average -*/
    foptions.add_int("ACI_N_AVERAGE", 1, "Number of roots to averag");
    /*- Offset for state averaging -*/
    foptions.add_int("ACI_AVERAGE_OFFSET", 0, "Offset for state averaging");
    
    forte_options.add_bool("ACI_SAVE_FINAL_WFN", false, "Print final wavefunction to file")
    forte_options.add_bool("ACI_PRINT_REFS", false, "Print the P space")
    forte_options.add_int("DL_GUESS_SIZE", 100, "Set the initial guess space size for DL solver")
    forte_options.add_int("N_GUESS_VEC", 10, "Number of guess vectors for Sparse CI solver")
    forte_options.add_double("ACI_NO_THRESHOLD", 0.02, "Threshold for active space prediction")
    forte_options.add_double("ACI_SPIN_TOL", 0.02, "Tolerance for S^2 value")
    # /*- Approximate 1RDM? -*/
    forte_options.add_bool("ACI_APPROXIMATE_RDM", false, "Approximate the RDMs")
    forte_options.add_bool("ACI_TEST_RDMS", false, "Run test for the RDMs")
    forte_options.add_bool("ACI_FIRST_ITER_ROOTS", false, "Compute all roots on first iteration?")
    forte_options.add_bool("ACI_PRINT_WEIGHTS", false, "Print weights for active space prediction")

    forte_options.add_bool("ACI_PRINT_NO", true, "Print the natural orbitals")

    forte_options.add_bool("ACI_NO", false, "Computes ACI natural orbitals")

    forte_options.add_bool("MRPT2", false, "Compute full PT2 energy")

    forte_options.add_bool("UNPAIRED_DENSITY", false, "Compute unpaired electron density")

    forte_options.add_bool("ACI_ADD_SINGLES", false,
                      "Adds all active single excitations to the final wave function")
    forte_options.add_bool("ACI_ADD_EXTERNAL_EXCITATIONS", false,
                      "Adds external single excitations to the final wave function")
    forte_options.add_str("ACI_EXTERNAL_EXCITATION_ORDER", "SINGLES",
                     "Order of external excitations to add")
    forte_options.add_str("ACI_EXTERNAL_EXCITATION_TYPE", "ALL", "Type of external excitations to add")
    forte_options.add_bool("ESNOS", false, "Compute external single natural orbitals (ESNO)")
    forte_options.add_int("ESNO_MAX_SIZE", 0, "Number of external orbitals to correlate")

    forte_options.add_bool("ACI_LOW_MEM_SCREENING", false, "Use low-memory screening algorithm")

    forte_options.add_bool("ACI_REF_RELAX", false, "Do reference relaxation in ACI")

    forte_options.add_str("ACI_EX_TYPE", "CONV", "Type of excited state to compute")

    forte_options.add_int("ACI_NFROZEN_CORE", 0, "Number of orbitals to freeze for core excitations")

    forte_options.add_int("ACI_ROOTS_PER_CORE", 1, "Number of roots to compute per frozen orbital")

    forte_options.add_bool("ACI_SPIN_ANALYSIS", false, "Do spin correlation analysis")
    forte_options.add_bool("ACI_RELAXED_SPIN", false,
                      "Do spin correlation analysis for relaxed wave function")

    forte_options.add_bool("PRINT_IAOS", true, "Print IAOs")

    forte_options.add_bool("PI_ACTIVE_SPACE", false, "Active space type")

    forte_options.add_bool("SPIN_MAT_TO_FILE", false, "Save spin correlation matrix to file")

    forte_options.add_str("SPIN_BASIS", "LOCAL", "Basis for spin analysis")

    forte_options.add_double("ACI_RELAX_SIGMA", 0.01, "Sigma for reference relaxation")

    forte_options.add_bool("ACI_BATCHED_SCREENING", false, "Control batched screeing")

    forte_options.add_int("ACI_NBATCH", 1, "Number of batches in screening")

    forte_options.add_int("ACI_MAX_MEM", 1000, "Sets max memory for batching algorithm (MB)")

    forte_options.add_double("ACI_SCALE_SIGMA", 0.5, "Scales sigma in batched algorithm")

    forte_options.add_bool("ACI_DIRECT_RDMS", false, "Computes RDMs without coupling lists")

    forte_options.add_str("ACI_BATCH_ALG", "HASH", "Algorithm to use for batching")


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
