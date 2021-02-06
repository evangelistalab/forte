import psi4
import forte


def psi4_scf(geom, basis, reference, functional='hf',
             options={}) -> (float, psi4.core.Wavefunction):
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
    default_options = {
        'SCF_TYPE': 'pk',
        'E_CONVERGENCE': 1.0e-10,
        'D_CONVERGENCE': 1.0e-6
    }

    # capitalize the options
    options = {k.upper(): v for k, v in options.items()}
    default_options = {k.upper(): v for k, v in default_options.items()}

    # merge the two dictionaries. The user-provided options will overwrite the default ones
    merged_options = {**default_options, **options}

    # add the mandatory arguments
    merged_options['BASIS'] = basis
    merged_options['REFERENCE'] = reference

    psi4.set_options(merged_options)

    # pipe output to the file output.dat
    psi4.core.set_output_file('output.dat', True)

    # run scf and return the energy and a wavefunction object (will work only if pass return_wfn=True)
    E_scf, wfn = psi4.energy(functional, return_wfn=True)

    return (E_scf, wfn)


def psi4_cubeprop(wfn,
                  path='.',
                  orbs=[],
                  nocc=0,
                  nvir=0,
                  density=False,
                  frontier_orbitals=False,
                  load=False):
    """
    Run a psi4 cubeprop computation to generate cube files from a given Wavefunction object
    By default this function plots from the HOMO -2 to the LUMO + 2

    Parameters
    ----------
    wfn : psi4Wavefunction
        A psi4 Wavefunction object
    path : str
        The path of the directory that will contain the cube files
    orbs : list or string
        The list of orbitals to convert to cube files (one based).
    nocc : int
        The number of occupied orbitals
    nvir : int
        The number of virtual orbitals
    """

    import os.path

    cubeprop_tasks = []

    if isinstance(orbs, str):
        if (orbs == 'frontier_orbitals'):
            cubeprop_tasks.append('FRONTIER_ORBITALS')
    else:
        cubeprop_tasks.append('ORBITALS')
        if nocc + nvir > 0:
            na = wfn.nalpha()
            nmo = wfn.nmo()
            min_orb = max(1, na + 1 - nocc)
            max_orb = min(nmo, na + nvir)
            orbs = [k for k in range(min_orb, max_orb + 1)]
        print(
            f'Preparing cube files for orbitals: {", ".join([str(orb) for orb in orbs])}'
        )

    if density:
        cubeprop_tasks.append('DENSITY')

    if not os.path.exists(path):
        os.makedirs(path)

    psi4.set_options({
        'CUBEPROP_TASKS': cubeprop_tasks,
        'CUBEPROP_ORBITALS': orbs,
        'CUBEPROP_FILEPATH': path
    })
    psi4.cubeprop(wfn)
    if load:
        return load_cubes(path)


def prepare_forte_objects(wfn,mo_spaces = None, active_space = 'ACTIVE',core_spaces = ['RESTRICTED_DOCC'],localize = False,localize_spaces = []):
    """
    Take a psi4 wavefunction object and prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects

    Parameters
    ----------
    wfn : psi4 Wavefunction
        A psi4 Wavefunction object
    mo_spaces : dict
        A dictionary with the size of each space (e.g., {'ACTIVE' : [3]})
    Returns
    -------
    tuple(ForteIntegrals, ActiveSpaceIntegrals, SCFInfo, MOSpaceInfo, map(StateInfo : list)
        a tuple containing the ForteIntegrals, SCFInfo, and MOSpaceInfo objects and a map of states and weights
    """
    # fill in the options object
    psi4_options = psi4.core.get_options()
    psi4_options.set_current_module('FORTE')
    options = forte.forte_options
    options.get_options_from_psi4(psi4_options)

    if ('DF' in options.get_str('INT_TYPE')):
        aux_basis = psi4.core.BasisSet.build(
            wfn.molecule(), 'DF_BASIS_MP2',
            psi4.core.get_global_option('DF_BASIS_MP2'), 'RIFIT',
            psi4.core.get_global_option('BASIS'))
        wfn.set_basisset('DF_BASIS_MP2', aux_basis)

    if (options.get_str('MINAO_BASIS')):
        minao_basis = psi4.core.BasisSet.build(
            wfn.molecule(), 'MINAO_BASIS', psi4_options.get_str('MINAO_BASIS'))
        wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Prepare base objects
    scf_info = forte.SCFInfo(wfn)

    nmopi = wfn.nmopi()
    point_group = wfn.molecule().point_group().symbol()

    if mo_spaces == None:
        mo_space_info = forte.make_mo_space_info(nmopi, point_group, options)
    else:
        mo_space_info = forte.make_mo_space_info_from_map(nmopi,point_group,mo_spaces,[])

    state_weights_map = forte.make_state_weights_map(options, mo_space_info)

    ints = forte.make_ints_from_psi4(wfn, options, mo_space_info)

    if localize:
        localizer = forte.Localize(forte.forte_options, ints, mo_space_info)
        localizer.set_orbital_space(localize_spaces)
        localizer.compute_transformation()
        Ua = localizer.get_Ua()
        ints.rotate_orbitals(Ua,Ua)

    # the space that defines the active orbitals. We select only the 'ACTIVE' part
    # the space(s) with non-active doubly occupied orbitals

    as_ints = forte.make_active_space_ints(mo_space_info, ints, active_space, core_spaces)

    return (ints, as_ints, scf_info, mo_space_info, state_weights_map)
