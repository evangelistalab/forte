# -*- coding: utf-8 -*-

def register_forte_options(forte_options):
    register_driver_options(forte_options)
    register_mo_space_info_options(forte_options)
    register_integral_options(forte_options)
    register_pt2_options(forte_options)
    register_pci_options(forte_options)
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

    forte_options.add_bool("PCI_FAST_EVAR", False, "Use a fast (sparse) estimate of the energy")

    forte_options.add_double("PCI_EVAR_MAX_ERROR", 0.0, "The max allowed error for variational energy")

    forte_options.add_int("PCI_ENERGY_ESTIMATE_FREQ", 1,
                     "Iterations in between variational estimation of the energy")

    forte_options.add_bool("PCI_ADAPTIVE_BETA", False, "Use an adaptive time step?")

    forte_options.add_bool("PCI_USE_INTER_NORM", False, "Use intermediate normalization")

    forte_options.add_bool("PCI_USE_SHIFT", False, "Use a shift in the exponential")

    forte_options.add_bool("PCI_VAR_ESTIMATE", False, "Estimate variational energy during calculation")

    forte_options.add_bool("PCI_PRINT_FULL_WAVEFUNCTION", False, "Print full wavefunction when finish")

    forte_options.add_bool("PCI_SIMPLE_PRESCREENING", False, "Prescreen the spawning of excitations")

    forte_options.add_bool("PCI_DYNAMIC_PRESCREENING", False, "Use dynamic prescreening")

    forte_options.add_bool("PCI_SCHWARZ_PRESCREENING", False, "Use schwarz prescreening")

    forte_options.add_bool("PCI_INITIATOR_APPROX", False, "Use initiator approximation")

    forte_options.add_double("PCI_INITIATOR_APPROX_FACTOR", 1.0, "The initiator approximation factor")

    forte_options.add_bool("PCI_PERTURB_ANALYSIS", False, "Do result perturbation analysis")

    forte_options.add_bool("PCI_SYMM_APPROX_H", False, "Use Symmetric Approximate Hamiltonian")

    forte_options.add_bool("PCI_STOP_HIGHER_NEW_LOW", False,
                      "Stop iteration when higher new low detected")

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

    forte_options.add_bool("PCI_REFERENCE_SPAWNING", False, "Do spawning according to reference")

    forte_options.add_bool("PCI_POST_DIAGONALIZE", False, "Do a post diagonalization?")

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
