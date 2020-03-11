import psi4
import forte

def psi4_scf(geom, basis, reference, functional = 'hf', options = {}) -> (double, psi4Wavefunction):
    """
    Run a psi4 scf computation and return the energy and the Wavefunction object

    Parameters
    ----------
    geom : str
        The molecular geometry (in xyz or zmat)
    basis : str
        The computational basis set
    reference : str
        The type of reference (rhf, uhf, rohf)
    functional : str
        The functional type for DFT (default = HF exchange)

    Returns
    -------
    tuple(double, psi4::Wavefunction)
        a tuple containing the energy and the Wavefunction object
    """
    # build the molecule object
    mol = psi4.geometry(geom)

    # add basis/reference/scf_type to options passed by the user
    options['basis'] = basis
    options['reference'] = reference
    options['scf_type'] = 'pk'

    psi4.set_options(options)

    # pipe output to the file output.dat
    psi4.core.set_output_file('output.dat', True)

    # run scf and return the energy and a wavefunction object (will work only if pass return_wfn=True)
    E_scf, wfn = psi4.energy(functional, return_wfn=True)

    return (E_scf, wfn)


def prepare_forte_objects(wfn):
    """
    Take a psi4 wavefunction object and prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects

    Parameters
    ----------
    wfn : psi4Wavefunction
        A psi4 Wavefunction object

    Returns
    -------
    tuple(ForteIntegrals, SCFInfo, MOSpaceInfo)
        a tuple containing the ForteIntegrals, SCFInfo, and MOSpaceInfo objects
    """
    # fill in the options object
    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')
    options = forte.forte_options
    options.get_options_from_psi4(psi4_options)

    if ('DF' in options.get_str('INT_TYPE')):
        aux_basis = psi4.core.BasisSet.build(wfn.molecule(), 'DF_BASIS_MP2',
                                         psi4.core.get_global_option('DF_BASIS_MP2'),
                                         'RIFIT', psi4.core.get_global_option('BASIS'))
        wfn.set_basisset('DF_BASIS_MP2', aux_basis)

    if (options.get_str('MINAO_BASIS')):
        minao_basis = psi4.core.BasisSet.build(wfn.molecule(), 'MINAO_BASIS',
                                               psi4_options.get_str('MINAO_BASIS'))
        wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Prepare base objects
    scf_info = forte.SCFInfo(wfn)
    mo_space_info = forte.make_mo_space_info(wfn, options)
    ints = forte.make_forte_integrals(wfn, options, mo_space_info)

    return (ints, scf_info, mo_space_info)
