# -*- coding: utf-8 -*-

def register_forte_options(forte_options):
    register_fci_options(forte_options)

def register_fci_options(forte_options):
    forte_options.add_int("FCI_NROOT", 1, "The number of roots computed")
    forte_options.add_int("FCI_ROOT", 0, "The root selected for state-specific computations")
    forte_options.add_int("FCI_MAXITER", 30, "Maximum number of iterations for FCI code")
    forte_options.add_int("FCI_MAX_RDM", 1, "The number of trial guess vectors to generate per root")
    forte_options.add_bool("FCI_TEST_RDMS", False, "Test the FCI reduced density matrices?")
    forte_options.add_bool("FCI_PRINT_NO", False, "Print the NO from the rdm of FCI")
    forte_options.add_int("FCI_NTRIAL_PER_ROOT", 10,
                     "The number of trial guess vectors to generate per root")


