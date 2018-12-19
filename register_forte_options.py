# -*- coding: utf-8 -*-

def register_forte_options(forte_options):
    register_driver_options(forte_options)
    register_fci_options(forte_options)

def register_driver_options(forte_options):
    forte_options.add_str('JOB_TYPE', 'NEWDRIVER',
                     ['NONE',
                      'ACI',
                      'PCI',
                      'CAS',
                      'DMRG',
                      'SR-DSRG',
                      'SR-DSRG-ACI',
                      'SR-DSRG-PCI',
                      'DSRG-MRPT2',
                      'DSRG-MRPT3',
                      'MR-DSRG-PT2',
                      'THREE-DSRG-MRPT2',
                      'SOMRDSRG',
                      'MRDSRG',
                      'MRDSRG_SO',
                      'CASSCF',
                      'ACTIVE-DSRGPT2',
                      'DWMS-DSRGPT2',
                      'DSRG_MRPT',
                      'TASKS',
                      'CC',
                      'NOJOB'], 'Specify the job type')

    forte_options.add_str('ACTIVE_SPACE_SOLVER','',['FCI','ACI'],'Active space solver type')
    forte_options.add_str('DYNCORR_SOLVER','',[],'Dynamical correlation solver type')

def register_fci_options(forte_options):
    forte_options.add_int('FCI_NROOT', 1, 'The number of roots computed')
    forte_options.add_int('FCI_ROOT', 0, 'The root selected for state-specific computations')
    forte_options.add_int('FCI_MAXITER', 30, 'Maximum number of iterations for FCI code')
    forte_options.add_int('FCI_MAX_RDM', 1, 'The number of trial guess vectors to generate per root')
    forte_options.add_bool('FCI_TEST_RDMS', False, 'Test the FCI reduced density matrices?')
    forte_options.add_bool('FCI_PRINT_NO', False, 'Print the NO from the rdm of FCI')
    forte_options.add_int('FCI_NTRIAL_PER_ROOT', 10,
                     'The number of trial guess vectors to generate per root')


