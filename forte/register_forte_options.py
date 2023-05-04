# -*- coding: utf-8 -*-


def register_forte_options(options):
    register_driver_options(options)
    register_mo_space_info_options(options)
    register_avas_options(options)
    register_cino_options(options)
    register_mrcino_options(options)
    register_embedding_options(options)
    register_integral_options(options)
    register_pt2_options(options)
    register_pci_options(options)
    register_fci_options(options)
    register_sci_options(options)
    register_aci_options(options)
    register_asci_options(options)
    register_tdci_options(options)
    register_detci_options(options)
    register_fci_mo_options(options)
    register_active_space_solver_options(options)
    register_dsrg_options(options)
    register_dwms_options(options)
    register_davidson_liu_options(options)
    register_localize_options(options)
    register_casscf_options(options)
    register_old_options(options)
    register_psi_options(options)
    register_gas_options(options)
    register_dmrg_options(options)


def register_driver_options(options):
    options.set_group("")
    options.add_str(
        'JOB_TYPE', 'NEWDRIVER', ['NONE', 'NEWDRIVER', 'MR-DSRG-PT2', 'CASSCF', 'MCSCF_TWO_STEP', 'TDCI'],
        'Specify the job type'
    )

    options.add_str("SCF_TYPE", None, "The integrals used in the SCF calculation")
    options.add_str(
        "REF_TYPE", 'SCF', ['SCF', 'CASSCF'], "The type of reference used by forte if a psi4 wave function is missing"
    )
    options.add_str('DERTYPE', 'NONE', ['NONE', 'FIRST'], 'Derivative order')

    options.add_double("E_CONVERGENCE", 1.0e-9, "The energy convergence criterion")
    options.add_double("D_CONVERGENCE", 1.0e-6, "The density convergence criterion")

    options.add_str(
        'ACTIVE_SPACE_SOLVER', '', ['FCI', 'ACI', 'ASCI', 'PCI', 'DETCI', 'CAS', 'DMRG', 'EXTERNAL'], 'Active space solver type'
    )
    options.add_str(
        'CORRELATION_SOLVER', 'NONE',
        ['DSRG-MRPT2', 'THREE-DSRG-MRPT2', 'DSRG-MRPT3', 'MRDSRG', 'SA-MRDSRG', 'DSRG_MRPT', 'MRDSRG_SO', 'SOMRDSRG'],
        'Dynamical correlation solver type'
    )
    options.add_str('CALC_TYPE', 'SS', ['SS', 'SA', 'MS', 'DWMS'], 'The type of computation')

    options.add_int("NEL", None, "The number of electrons. Used when reading from FCIDUMP files.")
    options.add_int(
        "CHARGE", None, "The charge of the molecule. "
        "If a value is provided it overrides the charge of Psi4."
    )
    options.add_int(
        "MULTIPLICITY", None, "The multiplicity = (2S + 1) of the electronic state. "
        "For example, 1 = singlet, 2 = doublet, 3 = triplet, ... "
        "If a value is provided it overrides the multiplicity of Psi4."
    )
    options.add_int("ROOT_SYM", None, 'The symmetry of the electronic state. (zero based)')
    options.add_str("ORBITAL_TYPE", "CANONICAL", ['CANONICAL', 'LOCAL', 'MP2NO', 'MRPT2NO'], 'Type of orbitals to use')

    options.add_str('MINAO_BASIS', 'STO-3G', "The basis used to define an orbital subspace")

    options.add_list("SUBSPACE", "A list of orbital subspaces")

    options.add_list("SUBSPACE_PI_PLANES", "A list of arrays of atoms composing the plane")

    options.add_double("MS", None, "Projection of spin onto the z axis")

    options.add_str(
        "ACTIVE_REF_TYPE", "CAS", ["HF", "CAS", "GAS", "GAS_SINGLE", "CIS", "CID", "CISD", "DOCI"],
        "Initial guess for active space wave functions"
    )

    options.add_bool("WRITE_RDM", False, "Save RDMs to ref_rdms.json for external computations")

    # TODO: Remove these in the future since they are redundant with READ/DUMP_ORBITALS (although they use different formats json vs. numpy)
    options.add_bool("WRITE_WFN", False, "Save ref_wfn.Ca() to coeff.json for external computations")

    options.add_bool("READ_WFN", False, "Read ref_wfn.Ca()/ref_wfn.Cb() from coeff.json for `external` active space solver")

    options.add_bool(
        "EXTERNAL_PARTIAL_RELAX", False,
        "Perform one relaxation step after building the DSRG effective Hamiltonian when using `external` active space solver")

    options.add_str(
        'EXT_RELAX_SOLVER', 'FCI', ['FCI', 'DETCI', 'CAS'], 'Active space solver used in the relaxation when using `external` active space solver'
    )

    options.add_int("PRINT", 1, "Set the print level.")

    options.add_bool("READ_ORBITALS", False, "Read orbitals from file if true")

    options.add_bool("DUMP_ORBITALS", False, "Save orbitals to file if true")


def register_avas_options(options):
    options.set_group("AVAS")

    options.add_bool("AVAS", False, "Form AVAS orbitals?")

    options.add_bool("AVAS_DIAGONALIZE", True, "Diagonalize Socc and Svir?")

    options.add_double(
        "AVAS_SIGMA", 0.98, "Cumulative cutoff to the eigenvalues of the overlap,"
        " which controls the size of the active space."
        " This value is tested against"
        " (sum of active e.values) / (sum of total e.values)"
    )

    options.add_double(
        "AVAS_CUTOFF", 1.0, "The eigenvalues of the overlap greater than this cutoff"
        " will be considered as active. If not equal to 1.0,"
        " it takes priority over cumulative cutoff selection."
    )

    options.add_double(
        "AVAS_EVALS_THRESHOLD", 1.0e-6, "Threshold smaller than which is considered as zero"
        " for an eigenvalue of the projected overlap."
    )

    options.add_int(
        "AVAS_NUM_ACTIVE", 0, "The total number of active orbitals. If not equal to 0,"
        " it takes priority over threshold-based selections."
    )

    options.add_int(
        "AVAS_NUM_ACTIVE_OCC", 0, "The number of active occupied orbitals. If not equal to 0,"
        " it takes priority over cutoff-based selections and"
        " that based on the total number of active orbitals."
    )

    options.add_int(
        "AVAS_NUM_ACTIVE_VIR", 0, "The number of active virtual orbitals. If not equal to 0,"
        " it takes priority over cutoff-based selections and"
        " that based on the total number of active orbitals."
    )


def register_cino_options(options):
    options.set_group("CINO")
    options.add_bool("CINO", False, "Do a CINO computation?")

    options.add_str("CINO_TYPE", "CIS", ["CIS", "CISD"], "The type of wave function.")

    options.add_int("CINO_NROOT", 1, "The number of roots computed")

    options.add_int_list("CINO_ROOTS_PER_IRREP", "The number of excited states per irreducible representation")
    options.add_double("CINO_THRESHOLD", 0.99, "The fraction of NOs to include in the active space")
    options.add_bool("CINO_AUTO", False, "{ass frozen_docc, actice_docc, and restricted_docc?")


def register_mrcino_options(options):
    options.set_group("MRCINO")

    options.add_bool("MRCINO", False, "Do a MRCINO computation?")

    options.add_str("MRCINO_TYPE", "CIS", ["CIS", "CISD"], "The type of wave function.")

    options.add_int("MRCINO_NROOT", 1, "The number of roots computed")

    options.add_int_list("MRCINO_ROOTS_PER_IRREP", "The number of excited states per irreducible representation")
    options.add_double("MRCINO_THRESHOLD", 0.99, "The fraction of NOs to include in the active space")
    options.add_bool(
        "MRCINO_AUTO", False, "Allow the users to choose"
        "whether pass frozen_docc"
        "actice_docc and restricted_docc"
        "or not"
    )


def register_embedding_options(options):
    options.set_group("Embedding")
    options.add_bool("EMBEDDING", False, "Whether to perform embedding partition and projection")
    options.add_str("EMBEDDING_CUTOFF_METHOD", "THRESHOLD", "Cut off by: threshold ,cum_threshold or num_of_orbitals.")
    options.add_double(
        "EMBEDDING_THRESHOLD", 0.5, "Projector eigenvalue threshold for both simple and cumulative threshold"
    )
    options.add_int(
        "NUM_A_DOCC", 0, "Number of occupied orbitals in A fixed to this value when embedding method is num_of_orbitals"
    )
    options.add_int(
        "Num_A_UOCC", 0, "Number of virtual orbitals in A fixed to this value when embedding method is num_of_orbitals"
    )
    options.add_str(
        "EMBEDDING_REFERENCE", "CASSCF",
        "HF for any reference without active, CASSCF for any reference with an active space."
    )
    options.add_bool("EMBEDDING_SEMICANONICALIZE_ACTIVE", True, "Perform semi-canonicalization on active space or not")
    options.add_bool(
        "EMBEDDING_SEMICANONICALIZE_FROZEN", True, "Perform semi-canonicalization on frozen core/virtual space or not"
    )
    options.add_int(
        "EMBEDDING_ADJUST_B_DOCC", 0, "Adjust number of occupied orbitals between A and B, +: move to B, -: move to A"
    )
    options.add_int(
        "EMBEDDING_ADJUST_B_UOCC", 0, "Adjust number of virtual orbitals between A and B, +: move to B, -: move to A"
    )
    options.add_str("EMBEDDING_VIRTUAL_SPACE", "ASET", ["ASET", "PAO", "IAO"], "Vitual space scheme")
    options.add_double("PAO_THRESHOLD", 1e-8, "Virtual space truncation threshold for PAO.")
    options.add_bool(
        "PAO_FIX_VIRTUAL_NUMBER", False,
        "Enable this option will generate PAOs equivlent to ASET virtuals, instead of using threshold"
    )


def register_mo_space_info_options(options):
    options.set_group("MO Space Info")

    options.add_int_list("FROZEN_DOCC", "Number of frozen occupied orbitals per irrep (in Cotton order)")
    options.add_int_list(
        "RESTRICTED_DOCC", "Number of restricted doubly"
        " occupied orbitals per irrep (in Cotton order)"
    )
    options.add_int_list("ACTIVE", " Number of active orbitals per irrep (in Cotton order)")
    options.add_int_list("RESTRICTED_UOCC", "Number of restricted unoccupied orbitals per irrep (in Cotton order)")
    options.add_int_list("FROZEN_UOCC", "Number of frozen unoccupied orbitals per irrep (in Cotton order)")

    options.add_int_list("GAS1", "Number of GAS1 orbitals per irrep (in Cotton order)")
    options.add_int_list("GAS2", "Number of GAS2 orbitals per irrep (in Cotton order)")
    options.add_int_list("GAS3", "Number of GAS3 orbitals per irrep (in Cotton order)")
    options.add_int_list("GAS4", "Number of GAS4 orbitals per irrep (in Cotton order)")
    options.add_int_list("GAS5", "Number of GAS5 orbitals per irrep (in Cotton order)")
    options.add_int_list("GAS6", "Number of GAS6 orbitals per irrep (in Cotton order)")

    #    /*- Molecular orbitals to swap -
    #     *  Swap mo_1 with mo_2 in irrep symmetry
    #     *  Swap mo_3 with mo_4 in irrep symmetry
    #     *  Format: [irrep, mo_1, mo_2, irrep, mo_3, mo_4]
    #     *          Irrep and MO indices are 1-based (NOT 0-based)!
    #    -*/
    options.add_int_list(
        "ROTATE_MOS", "An array of MOs to swap in the format"
        " [irrep, mo_1, mo_2, irrep, mo_3, mo_4]."
        " Irrep and MOs are all 1-based (NOT 0-based)!"
    )


def register_active_space_solver_options(options):
    options.set_group("Active Space Solver")
    options.add_int('NROOT', 1, 'The number of roots computed')
    options.add_int('ROOT', 0, 'The root selected for state-specific computations')

    options.add_list(
        "AVG_STATE",
        "A list of integer triplets that specify the irrep, multiplicity, and the number of states requested."
        "Uses the format [[irrep1, multi1, nstates1], [irrep2, multi2, nstates2], ...]"
    )

    options.add_list(
        "AVG_WEIGHT", "A list of lists that specify the weights assigned to all the states requested with AVG_STATE "
        "[[w1_1, w1_2, ..., w1_n], [w2_1, w2_2, ..., w2_n], ...]"
    )

    options.add_double("S_TOLERANCE", 0.25, "The maximum deviation from the spin quantum number S tolerated.")

    options.add_bool("DUMP_ACTIVE_WFN", False, "Save CI wave function of ActiveSpaceSolver to disk")

    options.add_bool("READ_ACTIVE_WFN_GUESS", False, "Read CI wave function of ActiveSpaceSolver from disk")

    options.add_bool("TRANSITION_DIPOLES", False, "Compute the transition dipole moments and oscillator strengths")

    options.add_bool(
        "PRINT_DIFFERENT_GAS_ONLY", False,
        "Only calculate the transition dipole between states with different GAS occupations?"
    )

    options.add_bool("DUMP_TRANSITION_RDM", False, "Dump transition reduced matrix into disk?")


def register_pt2_options(options):
    options.set_group("PT2")
    options.add_double("PT2_MAX_MEM", 1.0, "Maximum size of the determinant hash (GB)")


def register_pci_options(options):
    options.set_group("PCI")
    options.add_str(
        "PCI_GENERATOR", "WALL-CHEBYSHEV", [
            "LINEAR", "QUADRATIC", "CUBIC", "QUARTIC", "POWER", "TROTTER", "OLSEN", "DAVIDSON", "MITRUSHENKOV",
            "EXP-CHEBYSHEV", "WALL-CHEBYSHEV", "CHEBYSHEV", "LANCZOS", "DL"
        ], "The propagation algorithm"
    )

    options.add_int("PCI_NROOT", 1, "The number of roots computed")

    options.add_double("PCI_SPAWNING_THRESHOLD", 0.001, "The determinant importance threshold")

    options.add_int(
        "PCI_MAX_GUESS_SIZE", 10000, "The maximum number of determinants used to form the guess wave function"
    )

    options.add_double("PCI_GUESS_SPAWNING_THRESHOLD", -1, "The determinant importance threshold")

    options.add_double(
        "PCI_ENERGY_ESTIMATE_THRESHOLD", 1.0e-6, "The threshold with which we estimate the variational "
        "energy. Note that the final energy is always "
        "estimated exactly."
    )

    options.add_double("PCI_TAU", 1.0, "The time step in imaginary time (a.u.)")

    options.add_double("PCI_E_CONVERGENCE", 1.0e-8, "The energy convergence criterion")
    options.add_double("PCI_R_CONVERGENCE", 1.0, "The residual 2-norm convergence criterion")

    options.add_bool("PCI_FAST_EVAR", False, "Use a fast (sparse) estimate of the energy?")

    options.add_double("PCI_EVAR_MAX_ERROR", 0.0, "The max allowed error for variational energy")

    options.add_int("PCI_ENERGY_ESTIMATE_FREQ", 1, "Iterations in between variational estimation of the energy")

    options.add_bool("PCI_ADAPTIVE_BETA", False, "Use an adaptive time step?")

    options.add_bool("PCI_USE_INTER_NORM", False, "Use intermediate normalization?")

    options.add_bool("PCI_USE_SHIFT", False, "Use a shift in the exponential?")

    options.add_bool("PCI_VAR_ESTIMATE", False, "Estimate variational energy during calculation?")

    options.add_bool("PCI_PRINT_FULL_WAVEFUNCTION", False, "Print full wavefunction when finished?")

    options.add_bool("PCI_SIMPLE_PRESCREENING", False, "Prescreen the spawning of excitations?")

    options.add_bool("PCI_DYNAMIC_PRESCREENING", False, "Use dynamic prescreening?")

    options.add_bool("PCI_SCHWARZ_PRESCREENING", False, "Use schwarz prescreening?")

    options.add_bool("PCI_INITIATOR_APPROX", False, "Use initiator approximation?")

    options.add_double("PCI_INITIATOR_APPROX_FACTOR", 1.0, "The initiator approximation factor")

    options.add_bool("PCI_PERTURB_ANALYSIS", False, "Do result perturbation analysis?")

    options.add_bool("PCI_SYMM_APPROX_H", False, "Use Symmetric Approximate Hamiltonian?")

    options.add_bool("PCI_STOP_HIGHER_NEW_LOW", False, "Stop iteration when higher new low detected?")

    options.add_double("PCI_MAXBETA", 1000.0, "The maximum value of beta")

    options.add_int("PCI_MAX_DAVIDSON_ITER", 12, "The maximum value of Davidson generator iteration")

    options.add_int("PCI_DL_COLLAPSE_PER_ROOT", 2, "The number of trial vector to retain after Davidson-Liu collapsing")

    options.add_int("PCI_DL_SUBSPACE_PER_ROOT", 8, "The maxim number of trial Davidson-Liu vectors")

    options.add_int("PCI_CHEBYSHEV_ORDER", 5, "The order of Chebyshev truncation")

    options.add_int("PCI_KRYLOV_ORDER", 5, "The order of Krylov truncation")

    options.add_double("PCI_COLINEAR_THRESHOLD", 1.0e-6, "The minimum norm of orthogonal vector")

    options.add_bool("PCI_REFERENCE_SPAWNING", False, "Do spawning according to reference?")

    options.add_bool("PCI_POST_DIAGONALIZE", False, "Do a final diagonalization after convergence?")

    options.add_str(
        "PCI_FUNCTIONAL", "MAX", ["MAX", "SUM", "SQUARE", "SQRT", "SPECIFY-ORDER"],
        "The functional for determinant coupling importance evaluation"
    )

    options.add_double("PCI_FUNCTIONAL_ORDER", 1.0, "The functional order of PCI_FUNCTIONAL is SPECIFY-ORDER")


def register_fci_options(options):
    options.set_group("FCI")
    options.add_int('FCI_MAXITER', 30, 'Maximum number of iterations for FCI code')
    options.add_bool('FCI_TEST_RDMS', False, 'Test the FCI reduced density matrices?')
    options.add_bool('PRINT_NO', False, 'Print the NO from the rdm of FCI')
    options.add_int('NTRIAL_PER_ROOT', 10, 'The number of trial guess vectors to generate per root')


def register_sci_options(options):
    options.set_group("SCI")

    options.add_bool("SCI_ENFORCE_SPIN_COMPLETE", True, "Enforce determinant spaces (P and Q) to be spin-complete?")

    options.add_bool("SCI_ENFORCE_SPIN_COMPLETE_P", False, "Enforce determinant space P to be spin-complete?")

    options.add_bool(
        "SCI_PROJECT_OUT_SPIN_CONTAMINANTS", True, "Project out spin contaminants in Davidson-Liu's algorithm?"
    )

    options.add_str(
        "SCI_EXCITED_ALGORITHM", "NONE", ['AVERAGE', 'ROOT_ORTHOGONALIZE', 'ROOT_COMBINE', 'MULTISTATE'],
        "The selected CI excited state algorithm"
    )

    options.add_int("SCI_MAX_CYCLE", 20, "Maximum number of cycles")

    options.add_bool("SCI_QUIET_MODE", False, "Print during ACI procedure?")

    options.add_int("SCI_PREITERATIONS", 0, "Number of iterations to run SA-ACI before SS-ACI")

    options.add_bool("SCI_DIRECT_RDMS", False, "Computes RDMs without coupling lists?")

    options.add_bool("SCI_SAVE_FINAL_WFN", False, "Save final wavefunction to file?")

    options.add_bool("SCI_TEST_RDMS", False, "Run test for the RDMs?")

    options.add_bool("SCI_FIRST_ITER_ROOTS", False, "Compute all roots on first iteration?")

    options.add_bool("SCI_CORE_EX", False, "Use core excitation algorithm")


def register_aci_options(options):
    options.set_group("ACI")
    options.add_double("ACI_CONVERGENCE", 1e-9, "ACI Convergence threshold")

    options.add_str(
        "ACI_SCREEN_ALG", "AVERAGE", ['AVERAGE', 'SR', 'RESTRICTED', 'CORE', 'BATCH_HASH', 'BATCH_VEC', 'MULTI_GAS'],
        "The screening algorithm to use"
    )

    options.add_double("SIGMA", 0.01, "The energy selection threshold for the P space")

    options.add_double("GAMMA", 1.0, "The threshold for the selection of the Q space")

    options.add_double("ACI_PRESCREEN_THRESHOLD", 1e-12, "The SD space prescreening threshold")

    options.add_str(
        "ACI_PQ_FUNCTION", "AVERAGE", ['AVERAGE', 'MAX'], "Function of q-space criteria, per root for SA-ACI"
    )

    options.add_int(
        "ACI_SPIN_PROJECTION", 0, """Type of spin projection
     0 - None
     1 - Project initial P spaces at each iteration
     2 - Project only after converged PQ space
     3 - Do 1 and 2"""
    )

    options.add_bool("SPIN_PROJECT_FULL", False, "Project solution in full diagonalization algorithm?")

    options.add_bool(
        "ACI_ADD_AIMED_DEGENERATE", True, "Add degenerate determinants not included in the aimed selection"
    )

    options.add_int(
        "ACI_N_AVERAGE", 0, "Number of roots to average. When set to zero (default) it averages over all roots"
    )

    options.add_int("ACI_AVERAGE_OFFSET", 0, "Offset for state averaging")

    options.add_bool("ACI_PRINT_REFS", False, "Print the P space?")

    options.add_int("N_GUESS_VEC", 10, "Number of guess vectors for Sparse CI solver")

    options.add_double("ACI_NO_THRESHOLD", 0.02, "Threshold for active space prediction")

    options.add_double("ACI_SPIN_TOL", 0.02, "Tolerance for S^2 value")

    options.add_bool("ACI_APPROXIMATE_RDM", False, "Approximate the RDMs?")

    options.add_bool("ACI_PRINT_WEIGHTS", False, "Print weights for active space prediction?")

    options.add_bool("ACI_PRINT_NO", True, "Print the natural orbitals?")

    options.add_bool("ACI_NO", False, "Computes ACI natural orbitals?")

    options.add_bool("FULL_MRPT2", False, "Compute full PT2 energy?")

    options.add_bool("UNPAIRED_DENSITY", False, "Compute unpaired electron density?")

    options.add_bool("ACI_LOW_MEM_SCREENING", False, "Use low-memory screening algorithm?")

    options.add_bool("ACI_REF_RELAX", False, "Do reference relaxation in ACI?")

    options.add_int("ACI_NFROZEN_CORE", 0, "Number of orbitals to freeze for core excitations")

    options.add_int("ACI_ROOTS_PER_CORE", 1, "Number of roots to compute per frozen orbital")

    options.add_bool("SPIN_ANALYSIS", False, "Do spin correlation analysis?")

    options.add_bool("SPIN_TEST", False, "Do test validity of correlation analysis")

    options.add_bool("ACI_RELAXED_SPIN", False, "Do spin correlation analysis for relaxed wave function?")

    options.add_bool("PRINT_IAOS", True, "Print IAOs?")

    options.add_bool("PI_ACTIVE_SPACE", False, "Active space type?")

    options.add_bool("SPIN_MAT_TO_FILE", False, "Save spin correlation matrix to file?")

    options.add_str("SPIN_BASIS", "LOCAL", ['LOCAL', 'IAO', 'NO', 'CANONICAL'], "Basis for spin analysis")

    options.add_double("ACI_RELAX_SIGMA", 0.01, "Sigma for reference relaxation")

    options.add_int("ACI_NBATCH", 0, "Number of batches in screening")

    options.add_int("ACI_MAX_MEM", 1000, "Sets max memory for batching algorithm (MB)")

    options.add_double("ACI_SCALE_SIGMA", 0.5, "Scales sigma in batched algorithm")

    options.add_int("ACTIVE_GUESS_SIZE", 1000, "Number of determinants for CI guess")

    options.add_str("DIAG_ALGORITHM", "SPARSE", ["DYNAMIC", "FULL", "SPARSE"], "The diagonalization method")

    options.add_bool("FORCE_DIAG_METHOD", False, "Force the diagonalization procedure?")

    options.add_bool("ONE_CYCLE", False, "Doing only one cycle of ACI (FCI) ACI iteration?")

    options.add_bool("OCC_ANALYSIS", False, "Doing post calcualtion occupation analysis?")

    options.add_double(
        "OCC_LIMIT", 0.0001, "Occupation limit for considering"
        " if an orbital is occupied/unoccupied in the post calculation analysis."
    )

    options.add_double(
        "CORR_LIMIT", -0.01, "Correlation limit for considering"
        " if two orbitals are correlated in the post calculation analysis."
    )


def register_davidson_liu_options(options):
    options.set_group("Davidson-Liu")

    options.add_int("DL_MAXITER", 100, "The maximum number of Davidson-Liu iterations")

    options.add_int(
        "DL_GUESS_SIZE", 50, "Set the number of determinants in the initial guess"
        " space for the DL solver"
    )

    options.add_int("DL_COLLAPSE_PER_ROOT", 2, "The number of trial vector to retain after collapsing")

    options.add_int("DL_SUBSPACE_PER_ROOT", 10, "The maxim number of trial vectors")

    options.add_int(
        "SIGMA_VECTOR_MAX_MEMORY", 67108864,
        "The maximum number of doubles stored in memory in the sigma vector algorithm"
    )


def register_asci_options(options):
    options.set_group("ASCI")
    options.add_double("ASCI_E_CONVERGENCE", 1e-5, "ASCI energy convergence threshold")

    options.add_int("ASCI_TDET", 2000, "ASCI Max det")

    options.add_int("ASCI_CDET", 200, "ASCI Max reference det")

    options.add_double("ASCI_PRESCREEN_THRESHOLD", 1e-12, "ASCI prescreening threshold")

def register_tdci_options(options):

    options.add_int("TDCI_HOLE", 0,
            "Orbital used to ionize intial state. Number is indexed by the same ordering of orbitals in the determinants.")

    options.add_str("TDCI_PROPAGATOR", "EXACT", ['EXACT','CN','QCN','LINEAR','QUADRATIC',
                          'RK4', 'LANCZOS', 'EXACT_SELECT', 'RK4_SELECT', 'RK4_SELECT_LIST','ALL'],"Type of propagator")

    options.add_int("TDCI_NSTEP", 20, "Number of time-steps")

    options.add_double("TDCI_TIMESTEP", 1.0, "Timestep length in attosecond")

    options.add_double("TDCI_CN_CONVERGENCE", 1e-12, "Convergence threshold for CN iterations")

    options.add_bool("TDCI_PRINT_WFN", False, "Print coefficients to files")

    options.add_int_list("TDCI_OCC_ORB", "Print the occupation at integral time itervals for these orbitals")

    options.add_int("TDCI_KRYLOV_DIM", 5, "Dimension of Krylov subspace for Lanczos method")

    options.add_double("TDCI_ETA_P", 1e-12, "Path filtering threshold for P space")

    options.add_double("TDCI_ETA_PQ", 1e-12, "Path filtering threshold for Q space")

    options.add_double("TDCI_PRESCREEN_THRESH", 1e-12, "Prescreening threshold")

    options.add_bool("TDCI_TEST_OCC", False, "Test the occupation vectors")

def register_fci_mo_options(options):
    options.set_group("FCIMO")
    options.add_str("FCIMO_ACTV_TYPE", "COMPLETE", ["COMPLETE", "CIS", "CISD", "DOCI"], "The active space type")

    options.add_bool("FCIMO_CISD_NOHF", True, "Ground state: HF;"
                     " Excited states: no HF determinant in CISD space")

    options.add_str("FCIMO_IPEA", "NONE", ["NONE", "IP", "EA"], "Generate IP/EA CIS/CISD space")

    options.add_double("FCIMO_PRINT_CIVEC", 0.05, "The printing threshold for CI vectors")

    # options.add_bool("FCIMO_IAO_ANALYSIS", False, "Intrinsic atomic orbital analysis")


def register_detci_options(options):
    options.set_group("DETCI")

    options.add_double("DETCI_PRINT_CIVEC", 0.05, "The printing threshold for CI vectors")
    options.add_bool("DETCI_CISD_NO_HF", False, "Exclude HF determinant in active CID/CISD space")


def register_integral_options(options):
    options.set_group("Integrals")
    options.add_str(
        "INT_TYPE", "CONVENTIONAL", ["CONVENTIONAL", "CHOLESKY", "DF", "DISKDF", "FCIDUMP"],
        "The type of molecular integrals used in a computation"
        "- CONVENTIONAL Conventional four-index two-electron integrals"
        "- DF Density fitted two-electron integrals"
        "- CHOLESKY Cholesky decomposed two-electron integrals"
        "- FCIDUMP Read integrals from a file in the FCIDUMP format"
    )

    options.add_str('FCIDUMP_FILE', 'INTDUMP', 'The file that stores the FCIDUMP integrals')
    options.add_int_list(
        'FCIDUMP_DOCC',
        'The number of doubly occupied orbitals assumed for a FCIDUMP file. This information is used to build orbital energies.'
    )
    options.add_int_list(
        'FCIDUMP_SOCC',
        'The number of singly occupied orbitals assumed for a FCIDUMP file. This information is used to build orbital energies.'
    )

    options.add_bool("PRINT_INTS", False, "Print the one- and two-electron integrals?")


def register_dsrg_options(options):
    options.set_group("DSRG")

    options.add_double("DSRG_S", 0.5, "The value of the DSRG flow parameter s")

    options.add_double("DSRG_POWER", 2.0, "The power of the parameter s in the regularizer")

    options.add_str(
        "CORR_LEVEL", "PT2",
        ["PT2", "PT3", "LDSRG2", "LDSRG2_QC", "LSRG2", "SRG_PT2", "QDSRG2", "LDSRG2_P3", "QDSRG2_P3"],
        "Correlation level of MR-DSRG (used in mrdsrg code, "
        "LDSRG2_P3 and QDSRG2_P3 not implemented)"
    )

    options.add_str(
        "SOURCE", "STANDARD", ["STANDARD", "LABS", "DYSON", "AMP", "EMP2", "LAMP", "LEMP2"],
        "Source operator used in DSRG (AMP, EMP2, LAMP, LEMP2 "
        "only available in toy code mcsrgpt2)"
    )

    options.add_int(
        "DSRG_RSC_NCOMM", 20, "The maximum number of commutators in the recursive single commutator approximation"
    )

    options.add_double(
        "DSRG_RSC_THRESHOLD", 1.0e-12, "The treshold for terminating the recursive single commutator approximation"
    )

    options.add_str(
        "T_ALGORITHM", "DSRG", ["DSRG", "DSRG_NOSEMI", "SELEC", "ISA"],
        "The way of forming T amplitudes (DSRG_NOSEMI, SELEC, ISA "
        "only available in toy code mcsrgpt2)"
    )

    options.add_str(
        "DSRG_PT2_H0TH", "FDIAG", ["FDIAG", "FFULL", "FDIAG_VACTV", "FDIAG_VDIAG"],
        "Different Zeroth-order Hamiltonian of DSRG-MRPT (used in mrdsrg code)"
    )

    options.add_bool("DSRG_DIPOLE", False, "Compute (if true) DSRG dipole moments")

    options.add_int("DSRG_MAX_DIPOLE_LEVEL", 0,
                    "The max body level of DSRG transformed dipole moment (skip if < 1)")

    options.add_int("DSRG_MAX_QUADRUPOLE_LEVEL", 0,
                    "The max body level of DSRG transformed quadrupole moment (skip if < 1)")

    options.add_int("DSRG_MAXITER", 50, "Max iterations for nonperturbative"
                    " MR-DSRG amplitudes update")

    options.add_double("R_CONVERGENCE", 1.0e-6, "Residue convergence criteria for amplitudes")

    options.add_str("RELAX_REF", "NONE", ["NONE", "ONCE", "TWICE", "ITERATE"], "Relax the reference for MR-DSRG")

    options.add_int("MAXITER_RELAX_REF", 15, "Max macro iterations for DSRG reference relaxation")

    options.add_double("RELAX_E_CONVERGENCE", 1.0e-8, "The energy relaxation convergence criterion")

    options.add_bool(
        "DSRG_DUMP_RELAXED_ENERGIES", False, "Dump the energies after each reference relaxation step to JSON."
    )

    options.add_int("TAYLOR_THRESHOLD", 3, "DSRG Taylor expansion threshold for small denominator")

    options.add_int("NTAMP", 15, "Number of largest amplitudes printed in the summary")

    options.add_double("INTRUDER_TAMP", 0.10, "Threshold for amplitudes considered as intruders for printing")

    options.add_str("DSRG_TRANS_TYPE", "UNITARY", ["UNITARY", "CC"], "DSRG transformation type")

    options.add_str(
        "SMART_DSRG_S", "DSRG_S", ["DSRG_S", "MIN_DELTA1", "MAX_DELTA1", "DAVG_MIN_DELTA1", "DAVG_MAX_DELTA1"],
        "Automatically adjust the flow parameter according to denominators"
    )

    options.add_bool("PRINT_TIME_PROFILE", False, "Print detailed timings in dsrg-mrpt3")

    options.add_str(
        "DSRG_MULTI_STATE", "SA_FULL", ["SA_FULL", "SA_SUB", "MS", "XMS"],
        "Multi-state DSRG options (MS and XMS recouple states after single-state computations)\n"
        "  - State-average approach\n"
        "    - SA_SUB:  form H_MN = <M|Hbar|N>; M, N are CAS states of interest\n"
        "    - SA_FULL: redo a CASCI\n"
        "  - Multi-state approach (currently only for MRPT2)\n"
        "    - MS:  form 2nd-order Heff_MN = <M|H|N> + 0.5 * [<M|(T_M)^+ H|N> + <M|H T_N|N>]\n"
        "    - XMS: rotate references such that <M|F|N> is diagonal before MS procedure"
    )

    options.add_bool("FORM_HBAR3", False, "Form 3-body Hbar (only used in dsrg-mrpt2 with SA_SUB for testing)")

    options.add_bool("FORM_MBAR3", False, "Form 3-body mbar (only used in dsrg-mrpt2 for testing)")

    options.add_bool(
        "DSRGPT", True, "Renormalize (if true) the integrals for purturbitive"
        " calculation (only in toy code mcsrgpt2)"
    )

    options.add_str(
        "INTERNAL_AMP", "NONE", ["NONE", "SINGLES_DOUBLES", "SINGLES", "DOUBLES"],
        "Include internal amplitudes for VCIS/VCISD-DSRG acording"
        " to excitation level"
    )

    options.add_str(
        "INTERNAL_AMP_SELECT", "AUTO", ["AUTO", "ALL", "OOVV"],
        "Excitation types considered when internal amplitudes are included\n"
        "- Select only part of the asked internal amplitudes (IAs) in V-CIS/CISD\n"
        "  - AUTO: all IAs that changes excitations (O->V; OO->VV, OO->OV, OV->VV)\n"
        "  - ALL:  all IAs (O->O, V->V, O->V; OO->OO, OV->OV, VV->VV, OO->VV, OO->OV, OV->VV)\n"
        "  - OOVV: pure external (O->V; OO->VV)"
    )

    options.add_str(
        "T1_AMP", "DSRG", ["DSRG", "SRG", "ZERO"], "The way of forming T1 amplitudes (only in toy code mcsrgpt2)"
    )

    options.add_double(
        "ISA_B", 0.02, "Intruder state avoidance parameter when use ISA to"
        " form amplitudes (only in toy code mcsrgpt2)"
    )

    options.add_str(
        "CCVV_SOURCE", "NORMAL", ["ZERO", "NORMAL"],
        "Definition of source operator: special treatment for the CCVV term"
    )

    options.add_str(
        "CCVV_ALGORITHM", "FLY_AMBIT", [
            "CORE", "FLY_AMBIT", "FLY_LOOP", "BATCH_CORE", "BATCH_VIRTUAL", "BATCH_CORE_GA", "BATCH_VIRTUAL_GA",
            "BATCH_VIRTUAL_MPI", "BATCH_CORE_MPI", "BATCH_CORE_REP", "BATCH_VIRTUAL_REP"
        ], "Algorithm to compute the CCVV term in DSRG-MRPT2 (only in three-dsrg-mrpt2 code)"
    )

    options.add_bool("AO_DSRG_MRPT2", False, "Do AO-DSRG-MRPT2 if true (not available)")

    options.add_int("CCVV_BATCH_NUMBER", -1, "Batches for CCVV_ALGORITHM")

    options.add_bool("DSRG_MRPT2_DEBUG", False, "Excssive printing for three-dsrg-mrpt2")

    options.add_str(
        "THREEPDC_ALGORITHM", "CORE", ["CORE", "BATCH"], "Algorithm for evaluating 3-body cumulants in three-dsrg-mrpt2"
    )

    options.add_bool("THREE_MRPT2_TIMINGS", False, "Detailed printing (if true) in three-dsrg-mrpt2")

    options.add_bool(
        "PRINT_DENOM2", False, "Print (if true) (1 - exp(-2*s*D)) / D, renormalized denominators in DSRG-MRPT2"
    )

    options.add_bool("DSRG_HBAR_SEQ", False, "Evaluate H_bar sequentially if true")

    options.add_bool("DSRG_NIVO", False, "NIVO approximation: Omit tensor blocks with >= 3 virtual indices if true")

    options.add_bool("PRINT_1BODY_EVALS", False, "Print eigenvalues of 1-body effective H")

    options.add_bool("DSRG_MRPT3_BATCHED", False, "Force running the DSRG-MRPT3 code using the batched algorithm")

    options.add_bool("IGNORE_MEMORY_ERRORS", False, "Continue running DSRG-MRPT3 even if memory exceeds")

    options.add_int(
        "DSRG_DIIS_START", 2, "Iteration cycle to start adding error vectors for"
        " DSRG DIIS (< 1 for not doing DIIS)"
    )

    options.add_int("DSRG_DIIS_FREQ", 1, "Frequency of extrapolating error vectors for DSRG DIIS")

    options.add_int("DSRG_DIIS_MIN_VEC", 3, "Minimum size of DIIS vectors")

    options.add_int("DSRG_DIIS_MAX_VEC", 8, "Maximum size of DIIS vectors")

    options.add_bool("DSRG_RESTART_AMPS", True, "Restart DSRG amplitudes from a previous step")

    options.add_bool("DSRG_READ_AMPS", False, "Read initial amplitudes from the current directory")

    options.add_bool("DSRG_DUMP_AMPS", False, "Dump converged amplitudes to the current directory")

    options.add_str(
        "DSRG_T1_AMPS_GUESS", "PT2", ["PT2", "ZERO"],
        "The initial guess of T1 amplitudes for nonperturbative DSRG methods"
    )

    options.add_str(
        "DSRG_3RDM_ALGORITHM", "EXPLICIT", ["EXPLICIT", "DIRECT"],
        "Algorithm to compute 3-RDM contributions in fully contracted [H2, T2]"
    )

    options.add_bool("DSRG_RDM_MS_AVG", False, "Form Ms-averaged density if true")

    options.add_bool("SAVE_SA_DSRG_INTS", False, "Save SA-DSRG dressed integrals to dsrg_ints.json")


def register_dwms_options(options):
    options.set_group("DWMS")
    options.add_double(
        "DWMS_ZETA", 0.0, "Automatic Gaussian width cutoff for the density weights\n"
        "Weights of state α:\n"
        "Wi = exp(-ζ * (Eα - Ei)^2) / sum_j exp(-ζ * (Eα - Ej)^2)"
        "Energies (Eα, Ei, Ej) can be CASCI or SA-DSRG-PT2/3 energies."
    )

    options.add_str("DWMS_CORRLV", "PT2", ["PT2", "PT3"], "DWMS-DSRG-PT level")

    options.add_str(
        "DWMS_REFERENCE", "CASCI", ["CASCI", "PT2", "PT3", "PT2D"],
        "Energies to compute dynamic weights and CI vectors to do multi-state\n"
        "  CAS: CASCI energies and CI vectors\n"
        "  PT2: SA-DSRG-PT2 energies and SA-DSRG-PT2/CASCI vectors\n"
        "  PT3: SA-DSRG-PT3 energies and SA-DSRG-PT3/CASCI vectors\n"
        "  PT2D: Diagonal SA-DSRG-PT2c effective Hamiltonian elements and original CASCI vectors"
    )

    options.add_str(
        "DWMS_ALGORITHM", "SA", ["MS", "XMS", "SA", "XSA", "SH-0", "SH-1"], "DWMS algorithms:\n"
        "  - SA: state average Hαβ = 0.5 * ( <α|Hbar(β)|β> + <β|Hbar(α)|α> )\n"
        "  - XSA: extended state average (rotate Fαβ to a diagonal form)\n"
        "  - MS: multi-state (single-state single-reference)\n"
        "  - XMS: extended multi-state (single-state single-reference)\n"
        "  - To Be Deprecated:\n"
        "    - SH-0: separated diagonalizations, non-orthogonal final solutions\n"
        "    - SH-1: separated diagonalizations, orthogonal final solutions"
    )

    options.add_bool(
        "DWMS_DELTA_AMP", False, "Consider (if true) amplitudes difference between states"
        " X(αβ) = A(β) - A(α) in SA algorithm,"
        " testing in non-DF DSRG-MRPT2"
    )

    options.add_bool(
        "DWMS_ITERATE", False, "Iterative update the reference CI coefficients in"
        " SA algorithm, testing in non-DF DSRG-MRPT2"
    )

    options.add_int(
        "DWMS_MAXITER", 10, "Max number of iteration in the update of the reference"
        " CI coefficients in SA algorithm,"
        " testing in non-DF DSRG-MRPT2"
    )

    options.add_double("DWMS_E_CONVERGENCE", 1.0e-7, "Energy convergence criteria for DWMS iteration")


def register_localize_options(options):
    options.set_group("Localize")
    options.add_str("LOCALIZE", "PIPEK_MEZEY", ["PIPEK_MEZEY", "BOYS"], "The method used to localize the orbitals")
    options.add_int_list("LOCALIZE_SPACE", "Sets the orbital space for localization")


def register_casscf_options(options):
    options.set_group("CASSCF")

    options.add_int("CASSCF_MAXITER", 100, "The maximum number of CASSCF macro iterations")

    options.add_int("CASSCF_MICRO_MAXITER", 40, "The maximum number of CASSCF micro iterations")

    options.add_int("CASSCF_MICRO_MINITER", 6, "The minimum number of CASSCF micro iterations")

    options.add_int("CPSCF_MAXITER", 50, "Max iteration of solving coupled perturbed SCF equation")

    options.add_double("CPSCF_CONVERGENCE", 1e-8, "Convergence criterion for CP-SCF equation")

    options.add_double("CASSCF_E_CONVERGENCE", 1e-8, "The energy convergence criterion (two consecutive energies)")

    options.add_double(
        "CASSCF_G_CONVERGENCE", 1e-7, "The orbital gradient convergence criterion (RMS of gradient vector)"
    )

    options.add_bool("CASSCF_DEBUG_PRINTING", False, "Enable debug printing if True")

    options.add_bool("CASSCF_NO_ORBOPT", False, "No orbital optimization if true")

    options.add_bool("CASSCF_INTERNAL_ROT", False, "Keep GASn-GASn orbital rotations if true")

    # Zero mixing for orbital pairs
    # Format: [[irrep1, mo1, mo2], [irrep1, mo3, mo4], ...]
    # Irreps are 0-based, while MO indices are 1-based!
    # MO indices are relative indices within the irrep, e.g., 3A1 and 2A1: [[0, 3, 2]]
    options.add_list("CASSCF_ZERO_ROT", "An array of MOs [[irrep1, mo1, mo2], [irrep2, mo3, mo4], ...]")

    options.add_str(
        "CASSCF_FINAL_ORBITAL", "CANONICAL", ["CANONICAL", "NATURAL", "UNSPECIFIED"],
        "Constraints for redundant orbital pairs at the end of macro iteration"
    )

    options.add_str("CASSCF_CI_SOLVER", "FCI", "The active space solver to use in CASSCF")

    options.add_int(
        "CASSCF_CI_FREQ", 1, "How often to solve CI?\n"
        "< 1: do CI in the first macro iteration ONLY\n"
        "= n: do CI every n macro iteration"
    )

    options.add_bool("CASSCF_REFERENCE", False, "Run a FCI followed by CASSCF computation?")

    options.add_int(
        "CASSCF_MULTIPLICITY", 0, """Multiplicity for the CASSCF solution (if different from multiplicity)
    You should not use this if you are interested in having a CASSCF
    solution with the same multiplicitity as the DSRG-MRPT2"""
    )

    options.add_bool("CASSCF_SOSCF", False, "Run a complete SOSCF (form full Hessian)?")
    options.add_bool("OPTIMIZE_FROZEN_CORE", False, "Ignore frozen core option and optimize orbitals?")

    options.add_bool("RESTRICTED_DOCC_JK", True, "Use JK builder for restricted docc (EXPERT)?")

    options.add_double("CASSCF_MAX_ROTATION", 0.2, "Max value in orbital update vector")

    options.add_str(
        "CASSCF_ORB_ORTHO_TRANS", "CAYLEY", ["CAYLEY", "POWER", "PADE"],
        "Ways to compute the orthogonal transformation U from orbital rotation R"
    )

    options.add_str(
        "ORB_ROTATION_ALGORITHM", "DIAGONAL", ["DIAGONAL", "AUGMENTED_HESSIAN"], "Orbital rotation algorithm"
    )

    options.add_bool("CASSCF_DO_DIIS", True, "Use DIIS in CASSCF orbital optimization")
    options.add_int("CASSCF_DIIS_MIN_VEC", 3, "Minimum size of DIIS vectors for orbital rotations")
    options.add_int("CASSCF_DIIS_MAX_VEC", 8, "Maximum size of DIIS vectors for orbital rotations")
    options.add_int("CASSCF_DIIS_START", 15, "Iteration number to start adding error vectors (< 1 will not do DIIS)")
    options.add_int("CASSCF_DIIS_FREQ", 1, "How often to do DIIS extrapolation")
    options.add_double("CASSCF_DIIS_NORM", 1e-3, "Do DIIS when the orbital gradient norm is below this value")

    options.add_bool("CASSCF_CI_STEP", False, "Do a CAS step for every CASSCF_CI_FREQ")

    options.add_int("CASSCF_CI_STEP_START", -1, "When to start skipping CI steps")

    options.add_bool("MONITOR_SA_SOLUTION", False, "Monitor the CAS-CI solutions through iterations")

    options.add_int_list(
        "CASSCF_ACTIVE_FROZEN_ORBITAL",
        "A list of active orbitals to be frozen in the MCSCF optimization (in Pitzer order,"
        " zero based). Useful when doing core-excited state computations."
    )

    options.add_bool("CASSCF_DIE_IF_NOT_CONVERGED", True, "Stop Forte if MCSCF is not converged")


def register_old_options(options):
    options.set_group("Old")
    options.add_bool("NAT_ORBS_PRINT", False, "View the natural orbitals with their symmetry information")

    options.add_bool("NAT_ACT", False, "Use Natural Orbitals to suggest active space?")

    options.add_double("PT2NO_OCC_THRESHOLD", 0.98, "Occupancy smaller than which is considered as active")
    options.add_double("PT2NO_VIR_THRESHOLD", 0.02, "Occupancy greater than which is considered as active")

    options.add_bool("MEMORY_SUMMARY", False, "Print summary of memory")

    options.add_str("REFERENCE", "", "The SCF refernce type")

    options.add_int("DIIS_MAX_VECS", 5, "The maximum number of DIIS vectors")
    options.add_int("DIIS_MIN_VECS", 2, "The minimum number of DIIS vectors")
    options.add_int("MAXITER", 100, "The maximum number of iterations")

    options.add_bool("USE_DMRGSCF", False, "Use the older DMRGSCF algorithm?")

    #    /*- Semicanonicalize orbitals -*/
    options.add_bool("SEMI_CANONICAL", True, "Semicanonicalize orbitals for each elementary orbital space")
    options.add_bool(
        "SEMI_CANONICAL_MIX_INACTIVE", False, "Treat frozen and restricted orbitals together for semi-canonicalization"
    )
    options.add_bool("SEMI_CANONICAL_MIX_ACTIVE", False, "Treat all GAS orbitals together for semi-canonicalization")

    #    /*- Two-particle density cumulant -*/
    options.add_str("TWOPDC", "MK", ["MK", "ZERO"], "The form of the two-particle density cumulant")
    options.add_str("THREEPDC", "MK", ["MK", "MK_DECOMP", "ZERO"], "The form of the three-particle density cumulant")
    #    /*- Select a modified commutator -*/
    options.add_str("SRG_COMM", "STANDARD", "STANDARD FO FO2")

    #    /*- The initial time step used by the ode solver -*/
    options.add_double("SRG_DT", 0.001, "The initial time step used by the ode solver")
    #    /*- The absolute error tollerance for the ode solver -*/
    options.add_double("SRG_ODEINT_ABSERR", 1.0e-12, "The absolute error tollerance for the ode solver")
    #    /*- The absolute error tollerance for the ode solver -*/
    options.add_double("SRG_ODEINT_RELERR", 1.0e-12, "The absolute error tollerance for the ode solver")
    #    /*- Select a modified commutator -*/
    options.add_str("SRG_COMM", "STANDARD", ["STANDARD", "FO", "FO2"], "Select a modified commutator")

    options.add_str(
        "SRG_ODEINT", "FEHLBERG78", ["DOPRI5", "CASHKARP", "FEHLBERG78"],
        "The integrator used to propagate the SRG equations"
    )
    #    /*- The end value of the integration parameter s -*/
    options.add_double("SRG_SMAX", 10.0, "The end value of the integration parameter s")


def register_psi_options(options):
    options.add_str('BASIS', '', 'The primary basis set')
    options.add_str('BASIS_RELATIVISTIC', '', 'The basis set used to run relativistic computations')
    options.add_str("DF_INTS_IO", "NONE", ['NONE', 'SAVE', 'LOAD'], 'IO caching for CP corrections')
    options.add_str('DF_BASIS_MP2', '', 'Auxiliary basis set for density fitting computations')
    options.add_double("INTS_TOLERANCE", 1.0e-12, "Schwarz screening threshold")
    options.add_double("DF_FITTING_CONDITION", 1.0e-10, "Eigenvalue threshold for RI basis")
    options.add_double("CHOLESKY_TOLERANCE", 1.0e-6, "Tolerance for Cholesky integrals")


def register_gas_options(options):
    options.set_group("GAS")
    options.add_int_list("GAS1MAX", "The maximum number of electrons in GAS1 for different states")
    options.add_int_list("GAS1MIN", "The minimum number of electrons in GAS1 for different states")
    options.add_int_list("GAS2MAX", "The maximum number of electrons in GAS2 for different states")
    options.add_int_list("GAS2MIN", "The minimum number of electrons in GAS2 for different states")
    options.add_int_list("GAS3MAX", "The maximum number of electrons in GAS3 for different states")
    options.add_int_list("GAS3MIN", "The minimum number of electrons in GAS3 for different states")
    options.add_int_list("GAS4MAX", "The maximum number of electrons in GAS4 for different states")
    options.add_int_list("GAS4MIN", "The minimum number of electrons in GAS4 for different states")
    options.add_int_list("GAS5MAX", "The maximum number of electrons in GAS5 for different states")
    options.add_int_list("GAS5MIN", "The minimum number of electrons in GAS5 for different states")
    options.add_int_list("GAS6MAX", "The maximum number of electrons in GAS6 for different states")
    options.add_int_list("GAS6MIN", "The minimum number of electrons in GAS6 for different states")


def register_dmrg_options(options):
    options.set_group("DMRG")
    options.add_int_list(
        "DMRG_SWEEP_STATES", "Number of reduced renormalized basis states kept during successive DMRG instructions"
    )
    options.add_int_list(
        "DMRG_SWEEP_MAX_SWEEPS", "Max number of sweeps to stop an instruction during successive DMRG instructions"
    )
    options.add_double_list(
        "DMRG_SWEEP_ENERGY_CONV", "Energy convergence to stop an instruction during successive DMRG instructions"
    )
    options.add_double_list("DMRG_SWEEP_NOISE_PREFAC", "The noise prefactors for successive DMRG instructions")
    options.add_double_list(
        "DMRG_SWEEP_DVDSON_RTOL", "The residual tolerances for the Davidson diagonalization during DMRG instructions"
    )
    options.add_bool(
        "DMRG_PRINT_CORR", False, "Whether or not to print the correlation functions after the DMRG calculation"
    )

    #    /*- The minimum excitation level (Default value: 0) -*/
    #    options.add_int("MIN_EXC_LEVEL", 0)

    #    /*- The maximum excitation level (Default value: 0 = number of
    #     * electrons) -*/
    #    options.add_int("MAX_EXC_LEVEL", 0)

    #    /*- The algorithm used to screen the determinant
    #     *  - DENOMINATORS uses the MP denominators to screen strings
    #     *  - SINGLES generates the space by a series of single excitations -*/
    #    options.add_str("EXPLORER_ALGORITHM", "DENOMINATORS", "DENOMINATORS SINGLES")

    #    /*- The energy threshold for the determinant energy in Hartree -*/
    #    options.add_double("DET_THRESHOLD", 1.0)

    #    /*- The energy threshold for the MP denominators energy in Hartree -*/
    #    options.add_double("DEN_THRESHOLD", 1.5)

    #    /*- The criteria used to screen the strings -*/
    #    options.add_str("SCREENING_TYPE", "MP", "MP DET")

    #    // Options for the diagonalization of the Hamiltonian //
    #    /*- Determines if this job will compute the energy -*/
    #    options.add_bool("COMPUTE_ENERGY", True)

    #    /*- The form of the Hamiltonian matrix.
    #     *  - FIXED diagonalizes a matrix of fixed dimension
    #     *  - SMOOTH forms a matrix with smoothed matrix elements -*/
    #    options.add_str("H_TYPE", "FIXED_ENERGY", "FIXED_ENERGY FIXED_SIZE")

    #    /*- Determines if this job will compute the energy -*/
    #    options.add_str("ENERGY_TYPE", "FULL",
    #                    "FULL SELECTED LOWDIN SPARSE RENORMALIZE "
    #                    "RENORMALIZE_FIXED LMRCISD LMRCIS IMRCISD "
    #                    "IMRCISD_SPARSE LMRCISD_SPARSE LMRCIS_SPARSE "
    #                    "FACTORIZED_CI")

    #    /*- The form of the Hamiltonian matrix.
    #     *  - FIXED diagonalizes a matrix of fixed dimension
    #     *  - SMOOTH forms a matrix with smoothed matrix elements -*/

    #    //    options.add_int("IMRCISD_TEST_SIZE", 0)
    #    //    options.add_int("IMRCISD_SIZE", 0)

    #    /*- The number of determinants used to build the Hamiltonian -*/
    #    options.add_int("NDETS", 100)

    #    /*- The maximum dimension of the Hamiltonian -*/
    #    options.add_int("MAX_NDETS", 1000000)

    #    /*- The energy threshold for the model space -*/
    #    options.add_double("SPACE_M_THRESHOLD", 1000.0)

    #    /*- The energy threshold for the intermdiate space -*/
    #    options.add_double("SPACE_I_THRESHOLD", 1000.0)

    #    /*- The energy threshold for the intermdiate space -*/
    #    options.add_double("T2_THRESHOLD", 0.000001)

    #    /*- The number of steps used in the renormalized Lambda CI -*/
    #    options.add_int("RENORMALIZATION_STEPS", 10)

    #    /*- The energy threshold for smoothing the Hamiltonian.
    #     *  Determinants with energy < DET_THRESHOLD - SMO_THRESHOLD will be
    #     * included in H
    #     *  Determinants with DET_THRESHOLD - SMO_THRESHOLD < energy <
    #     * DET_THRESHOLD will be included in H but smoothed
    #     *  Determinants with energy > DET_THRESHOLD will not be included in H
    #     * -*/
    #    options.add_double("SMO_THRESHOLD", 0.0)

    #    /*- The method used to smooth the Hamiltonian -*/
    #    options.add_bool("SMOOTH", False)

    #    /*- The method used to smooth the Hamiltonian -*/
    #    options.add_bool("SELECT", False)

    #    /*- The energy convergence criterion -*/
    #    options.add_double("E_CONVERGENCE", 1.0e-8)

    #    options.add_bool("MOLDEN_WRITE_FORTE", False)
    #    // Natural Orbital selection criteria.  Used to fine tune how many
    #    // active orbitals there are

    #    /*- Typically, a occupied orbital with a NO occupation of <0.98 is
    #     * considered active -*/
    #    options.add_double("OCC_NATURAL", 0.98)
    #    /*- Typically, a virtual orbital with a NO occupation of > 0.02 is
    #     * considered active -*/
    #    options.add_double("VIRT_NATURAL", 0.02)

    #    /*- The amount of information printed
    #        to the output file -*/
    #    options.add_int("PRINT", 0)
    #    /*-  -*/


#    // Options for the Cartographer class //
#    /*- Density of determinants format -*/
#    options.add_str("DOD_FORMAT", "HISTOGRAM", "GAUSSIAN HISTOGRAM")
#    /*- Number of bins used to form the DOD plot -*/
#    options.add_int("DOD_BINS", 2000)
#    /*- Width of the DOD Gaussian/histogram.  Default 0.02 Hartree ~ 0.5 eV
#     * -*/
#    options.add_double("DOD_BIN_WIDTH", 0.05)
#    /*- Write the determinant occupation? -*/
#    options.add_bool("WRITE_OCCUPATION", True)
#    /*- Write the determinant energy? -*/
#    options.add_bool("WRITE_DET_ENERGY", True)
#    /*- Write the denominator energy? -*/
#    options.add_bool("WRITE_DEN_ENERGY", False)
#    /*- Write the excitation level? -*/
#    options.add_bool("WRITE_EXC_LEVEL", False)
#    /*- Write information only for a given excitation level.
#        0 (default) means print all -*/
#    options.add_int("RESTRICT_EXCITATION", 0)
#    /*- The energy buffer for building the Hamiltonian matrix in Hartree -*/
#    options.add_double("H_BUFFER", 0.0)

#    /*- The maximum number of iterations -*/
#    options.add_int("MAXITER", 100)

#    // Options for the Genetic Algorithm CI //
#    /*- The size of the population -*/
#    //    options.add_int("NPOP", 100)

#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR ALTERNATIVES FOR CASSCF ORBITALS
#    //////////////////////////////////////////////////////////////
#    /*- What type of alternative CASSCF Orbitals do you want -*/
#    options.add_str("ALTERNATIVE_CASSCF", "NONE", "IVO FTHF NONE")
#    options.add_double("TEMPERATURE", 50000)

#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE CASSCF CODE
#    //////////////////////////////////////////////////////////////

#    /*- The CI solver to use -*/

#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE DMRGSOLVER
#    //////////////////////////////////////////////////////////////

#    options.add_int("DMRG_WFN_MULTP", -1)

#    /*- The DMRGSCF wavefunction irrep uses the same conventions as PSI4.
#    How convenient :-).
#        Just to avoid confusion, it's copied here. It can also be found on
#        http://sebwouters.github.io/CheMPS2/classCheMPS2_1_1Irreps.html .

#        Symmetry Conventions        Irrep Number & Name
#        Group Number & Name         0 	1 	2 	3 	4 5
#    6
#    7
#        0: c1                       A
#        1: ci                       Ag 	Au
#        2: c2                       A 	B
#        3: cs                       A' 	A''
#        4: d2                       A 	B1 	B2 	B3
#        5: c2v                      A1 	A2 	B1 	B2
#        6: c2h                      Ag 	Bg 	Au 	Bu
#        7: d2h                      Ag 	B1g 	B2g 	B3g 	Au
#    B1u 	B2u 	B3u
#    -*/
#    options.add_int("DMRG_WFN_IRREP", -1)
#    /*- FrozenDocc for DMRG (frozen means restricted) -*/
#    options.add_list("DMRG_FROZEN_DOCC")

#    /*- The number of reduced renormalized basis states to be
#        retained during successive DMRG instructions -*/
#    options.add_list("DMRG_STATES")

#    /*- The energy convergence to stop an instruction
#        during successive DMRG instructions -*/
#    options.add_list("DMRG_ECONV")

#    /*- The maximum number of sweeps to stop an instruction
#        during successive DMRG instructions -*/
#    options.add_list("DMRG_MAXSWEEPS")
#    /*- The Davidson R tolerance (Wouters says this will cause RDms to be
#     * close to exact -*/
#    options.add_list("DMRG_DAVIDSON_RTOL")

#    /*- The noiseprefactors for successive DMRG instructions -*/
#    options.add_list("DMRG_NOISEPREFACTORS")

#    /*- Whether or not to print the correlation functions after the DMRG
#     * calculation -*/
#    options.add_bool("DMRG_PRINT_CORR", False)

#    /*- Whether or not to create intermediary MPS checkpoints -*/
#    options.add_bool("MPS_CHKPT", False)

#    /*- Convergence threshold for the gradient norm. -*/
#    options.add_double("DMRG_CONVERGENCE", 1e-6)

#    /*- Whether or not to store the unitary on disk (convenient for
#     * restarting). -*/
#    options.add_bool("DMRG_STORE_UNIT", True)

#    /*- Whether or not to use DIIS for DMRGSCF. -*/
#    options.add_bool("DMRG_DO_DIIS", False)

#    /*- When the update norm is smaller than this value DIIS starts. -*/
#    options.add_double("DMRG_DIIS_BRANCH", 1e-2)

#    /*- Whether or not to store the DIIS checkpoint on disk (convenient for
#     * restarting). -*/
#    options.add_bool("DMRG_STORE_DIIS", True)

#    /*- Maximum number of DMRGSCF iterations -*/
#    options.add_int("DMRGSCF_MAX_ITER", 100)

#    /*- Which root is targeted: 1 means ground state, 2 first excited state,
#     * etc. -*/
#    options.add_int("DMRG_WHICH_ROOT", 1)

#    /*- Whether or not to use state-averaging for roots >=2 with DMRG-SCF.
#     * -*/
#    options.add_bool("DMRG_AVG_STATES", True)

#    /*- Which active space to use for DMRGSCF calculations:
#           --> input with SCF rotations (INPUT)
#           --> natural orbitals (NO)
#           --> localized and ordered orbitals (LOC) -*/
#    options.add_str("DMRG_ACTIVE_SPACE", "INPUT", "INPUT NO LOC")

#    /*- Whether to start the active space localization process from a random
#     * unitary or the unit matrix. -*/
#    options.add_bool("DMRG_LOC_RANDOM", True)
#    /*-  -*/
#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE FULL CI QUANTUM MONTE-CARLO
#    //////////////////////////////////////////////////////////////
#    /*- The maximum value of beta -*/
#    options.add_double("START_NUM_WALKERS", 1000.0)
#    /*- Spawn excitation type -*/
#    options.add_str("SPAWN_TYPE", "RANDOM", "RANDOM ALL GROUND_AND_RANDOM")
#    /*- The number of walkers for shift -*/
#    options.add_double("SHIFT_NUM_WALKERS", 10000.0)
#    options.add_int("SHIFT_FREQ", 10)
#    options.add_double("SHIFT_DAMP", 0.1)
#    /*- Clone/Death scope -*/
#    options.add_bool("DEATH_PARENT_ONLY", False)
#    /*- initiator -*/
#    options.add_bool("USE_INITIATOR", False)
#    options.add_double("INITIATOR_NA", 3.0)
#    /*- Iterations in between variational estimation of the energy -*/
#    options.add_int("VAR_ENERGY_ESTIMATE_FREQ", 1000)
#    /*- Iterations in between printing information -*/
#    options.add_int("PRINT_FREQ", 100)

#    //////////////////////////////////////////////////////////////
#    ///
#    ///              OPTIONS FOR THE SRG MODULE
#    ///
#    //////////////////////////////////////////////////////////////
#    /*- The type of operator to use in the SRG transformation -*/
#    options.add_str("SRG_MODE", "DSRG", "DSRG CT")
#    /*- The type of operator to use in the SRG transformation -*/
#    options.add_str("SRG_OP", "UNITARY", "UNITARY CC")
#    /*- The flow generator to use in the SRG equations -*/
#    options.add_str("SRG_ETA", "WHITE", "WEGNER_BLOCK WHITE")
#    /*- The integrator used to propagate the SRG equations -*/

#    /*-  -*/

#    // --------------------------- SRG EXPERT OPTIONS
#    // ---------------------------

#    /*- Save Hbar? -*/
#    options.add_bool("SAVE_HBAR", False)

#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE PILOT FULL CI CODE
#    //////////////////////////////////////////////////////////////

#    /*- The density convergence criterion -*/
#    options.add_double("D_CONVERGENCE", 1.0e-8)

#    //////////////////////////////////////////////////////////////
#    ///         OPTIONS FOR THE V2RDM INTERFACE
#    //////////////////////////////////////////////////////////////
#    /*- Write Density Matrices or Cumulants to File -*/
#    options.add_str("WRITE_DENSITY_TYPE", "NONE", "NONE DENSITY CUMULANT")
#    /*- Average densities of different spins in V2RDM -*/
#    options.add_bool("AVG_DENS_SPIN", False)

#    //////////////////////////////////////////////////////////////
#    ///              OPTIONS FOR THE MR-DSRG MODULE
#    //////////////////////////////////////////////////////////////

#    /*- The code used to do CAS-CI.
#     *  - CAS   determinant based CI code
#     *  - FCI   string based FCI code
#     *  - DMRG  DMRG code
#     *  - V2RDM V2RDM interface -*/
#    options.add_str("CAS_TYPE", "FCI", "CAS FCI ACI DMRG V2RDM")
