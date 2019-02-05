# -*- coding: utf-8 -*-

def register_forte_options(forte_options):
    register_driver_options(forte_options)
    register_mo_space_info_options(forte_options)
    register_avas_options(forte_options)
    register_cino_options(forte_options)
    register_mrcino_options(forte_options)
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

def register_avas_options(forte_options):
    forte_options.add_double("AVAS_SIGMA", 0.98, "Threshold that controls the size of the active space")
    forte_options.add_int("AVAS_NUM_ACTIVE", 0,
                     "Allows the user to specify the "
                     "total number of active orbitals. "
                     "It takes priority over the "
                     "threshold based selection.")
    forte_options.add_int("AVAS_NUM_ACTIVE_OCC", 0,
                     "Allows the user to specify the "
                     "number of active occupied orbitals. "
                     "It takes priority over the "
                     "threshold based selection.")
    forte_options.add_int("AVAS_NUM_ACTIVE_VIR", 0,
                     "Allows the user to specify the "
                     "number of active occupied orbitals. "
                     "It takes priority over the "
                     "threshold based selection.")
    forte_options.add_bool("AVAS_DIAGONALIZE", True,
                      "Allow the users to specify"
                      "diagonalization of Socc and Svir"
                      "It takes priority over the"
                      "threshold based selection.")

def register_cino_options(forte_options):
    forte_options.add_bool("CINO", False, "Do a CINO computation?")
    forte_options.add_str("CINO_TYPE", "CIS", ["CIS", "CISD"], "The type of wave function.")
    forte_options.add_int("CINO_NROOT", 1, "The number of roots computed")
    forte_options.add_array("CINO_ROOTS_PER_IRREP",
                       "The number of excited states per irreducible representation")
    forte_options.add_double("CINO_THRESHOLD", 0.99,
                        "The fraction of NOs to include in the active space")
#   forte_options.add_int("ACI_MAX_RDM", 1, "Order of RDM to compute
#     Type of spin projection
#     * 0 - None
#     * 1 - Project initial P spaces at each iteration
#     * 2 - Project only after converged PQ space
#     * 3 - Do 1 and 2 ")
    forte_options.add_bool("CINO_AUTO", False,
                      "Allow the users to choose"
                      "whether pass frozen_docc"
                      "actice_docc and restricted_docc"
                      "or not")

def register_mrcino_options(forte_options):
    forte_options.add_bool("MRCINO", False, "Do a MRCINO computation?")
    forte_options.add_str("MRCINO_TYPE", "CIS", ["CIS", "CISD"], "The type of wave function.")
    forte_options.add_int("MRCINO_NROOT", 1, "The number of roots computed")
    forte_options.add_array("MRCINO_ROOTS_PER_IRREP",
                       "The number of excited states per irreducible representation")
    forte_options.add_double("MRCINO_THRESHOLD", 0.99,
                        "The fraction of NOs to include in the active space")
#    forte_options.add_int("ACI_MAX_RDM", 1, "Order of RDM to compute 
#     Type of spin projection
#     * 0 - None
#     * 1 - Project initial P spaces at each iteration
#     * 2 - Project only after converged PQ space
#     * 3 - Do 1 and 2 ")

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
