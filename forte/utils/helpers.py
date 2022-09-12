import psi4
import forte


def psi4_scf(geom, basis, reference, functional='hf', options={}) -> (float, psi4.core.Wavefunction):
    """Run a psi4 scf computation and return the energy and the Wavefunction object

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

    # clean psi4
    psi4.core.clean()

    # build the molecule object
    mol = psi4.geometry(geom)

    # add basis/reference/scf_type to options passed by the user
    default_options = {'SCF_TYPE': 'PK', 'E_CONVERGENCE': 1.0e-11, 'D_CONVERGENCE': 1.0e-6}

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
    E_scf, wfn = psi4.energy(functional, molecule=mol, return_wfn=True)

    return (E_scf, wfn)


def psi4_casscf(geom, basis, reference, restricted_docc, active, options={}) -> (float, psi4.core.Wavefunction):
    """Run a psi4 casscf computation and return the energy and the Wavefunction object

    Parameters
    ----------
    geom : str
        The molecular geometry (in xyz or zmat)
    basis : str
        The computational basis set
    reference : str
        The type of reference (rhf, uhf, rohf)

    Returns
    -------
    tuple(double, psi4::Wavefunction)
        a tuple containing the energy and the Wavefunction object
    """
    # build the molecule object
    mol = psi4.geometry(geom)

    # add basis/reference/scf_type to options passed by the user
    default_options = {'SCF_TYPE': 'pk', 'E_CONVERGENCE': 1.0e-10, 'D_CONVERGENCE': 1.0e-6}

    # capitalize the options
    options = {k.upper(): v for k, v in options.items()}
    default_options = {k.upper(): v for k, v in default_options.items()}

    # merge the two dictionaries. The user-provided options will overwrite the default ones
    merged_options = {**default_options, **options}

    # add the mandatory arguments
    merged_options['BASIS'] = basis
    merged_options['REFERENCE'] = reference
    merged_options['RESTRICTED_DOCC'] = restricted_docc
    merged_options['ACTIVE'] = active
    merged_options['MCSCF_MAXITER'] = 100
    merged_options['MCSCF_E_CONVERGENCE'] = 1.0e-10
    merged_options['MCSCF_R_CONVERGENCE'] = 1.0e-6
    merged_options['MCSCF_DIIS_START'] = 20

    psi4.set_options(merged_options)

    # pipe output to the file output.dat
    psi4.core.set_output_file('output.dat', True)

    # psi4.core.clean()

    # run scf and return the energy and a wavefunction object (will work only if pass return_wfn=True)
    E_scf, wfn = psi4.energy('casscf', molecule=mol, return_wfn=True)
    return (E_scf, wfn)


def psi4_casscf(geom, basis, mo_spaces):
    """
    Run a Psi4 SCF.
    :param geom: a string for molecular geometry
    :param basis: a string for basis set
    :param reference: a string for the type of reference
    :return: a tuple of (scf energy, psi4 Wavefunction)
    """
    psi4.core.clean()
    mol = psi4.geometry(geom)

    psi4.set_options(
        {
            'basis': basis,
            'scf_type': 'pk',
            'e_convergence': 1e-13,
            'd_convergence': 1e-6,
            'restricted_docc': mo_spaces['RESTRICTED_DOCC'],
            'active': mo_spaces['ACTIVE'],
            'mcscf_maxiter': 100,
            'mcscf_e_convergence': 1.0e-11,
            'mcscf_r_convergence': 1.0e-6,
            'mcscf_diis_start': 20
        }
    )
    psi4.core.set_output_file('output.dat', False)

    Escf, wfn = psi4.energy('casscf', return_wfn=True)
    psi4.core.clean()
    return Escf, wfn


def psi4_cubeprop(wfn, path='.', orbs=[], nocc=0, nvir=0, density=False, frontier_orbitals=False, load=False):
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
        print(f'Preparing cube files for orbitals: {", ".join([str(orb) for orb in orbs])}')

    if density:
        cubeprop_tasks.append('DENSITY')

    if not os.path.exists(path):
        os.makedirs(path)

    psi4.set_options({'CUBEPROP_TASKS': cubeprop_tasks, 'CUBEPROP_ORBITALS': orbs, 'CUBEPROP_FILEPATH': path})
    psi4.cubeprop(wfn)


def prepare_forte_objects(
    wfn, mo_spaces=None, active_space='ACTIVE', core_spaces=['RESTRICTED_DOCC'], localize=False, localize_spaces=[]
):
    """Take a psi4 wavefunction object and prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects

    Parameters
    ----------
    wfn : psi4 Wavefunction
        A psi4 Wavefunction object
    mo_spaces : dict
        A dictionary with the size of each space (e.g., {'ACTIVE' : [3]})
    active_space : str
        The MO space treated as active (default: 'ACTIVE')
    core_spaces : list(str)
        The MO spaces treated as active (default: ['RESTRICTED_DOCC'])
    localize : bool
        Do localize the orbitals? (defaul: False)
    localize_spaces : list(str)
        A list of spaces to localize (default: [])
    Returns
    -------
    dict(ForteIntegrals, ActiveSpaceIntegrals, SCFInfo, MOSpaceInfo, map(StateInfo : list))
        a dictionary containing the ForteIntegrals, SCFInfo, and MOSpaceInfo objects and a map of states and weights
    """
    # fill in the options object
    options = forte.forte_options

    if ('DF' in options.get_str('INT_TYPE')):
        aux_basis = psi4.core.BasisSet.build(
            wfn.molecule(), 'DF_BASIS_MP2', psi4.core.get_global_option('DF_BASIS_MP2'), 'RIFIT',
            psi4.core.get_global_option('BASIS'), puream=wfn.basisset().has_puream()
        )
        wfn.set_basisset('DF_BASIS_MP2', aux_basis)

    if (options.get_str('MINAO_BASIS')):
        minao_basis = psi4.core.BasisSet.build(wfn.molecule(), 'MINAO_BASIS', options.get_str('MINAO_BASIS'))
        wfn.set_basisset('MINAO_BASIS', minao_basis)

    # Prepare base objects
    scf_info = forte.SCFInfo(wfn)

    # Grab the number of MOs per irrep
    nmopi = wfn.nmopi()

    # Grab the point group symbol (e.g. "C2V")
    point_group = wfn.molecule().point_group().symbol()

    # create a MOSpaceInfo object
    if mo_spaces is None:
        mo_space_info = forte.make_mo_space_info(nmopi, point_group, options)
    else:
        mo_space_info = forte.make_mo_space_info_from_map(nmopi, point_group, mo_spaces, [])

    state_weights_map = forte.make_state_weights_map(options, mo_space_info)

    # make a ForteIntegral object
    ints = forte.make_ints_from_psi4(wfn, options, mo_space_info)

    if localize:
        localizer = forte.Localize(forte.forte_options, ints, mo_space_info)
        localizer.set_orbital_space(localize_spaces)
        localizer.compute_transformation()
        Ua = localizer.get_Ua()
        ints.rotate_orbitals(Ua, Ua)

    # the space that defines the active orbitals. We select only the 'ACTIVE' part
    # the space(s) with non-active doubly occupied orbitals

    # create active space integrals
    as_ints = forte.make_active_space_ints(mo_space_info, ints, active_space, core_spaces)

    return {
        'ints': ints,
        'as_ints': as_ints,
        'scf_info': scf_info,
        'mo_space_info': mo_space_info,
        'state_weights_map': state_weights_map
    }


def prepare_ints_rdms(wfn, mo_spaces, rdm_level=3, rdm_type=forte.RDMsType.spin_dependent):
    """
    Preparation step for DSRG: compute a CAS and its RDMs.
    :param wfn: reference wave function from psi4
    :param mo_spaces: a dictionary {mo_space: occupation}, e.g., {'ACTIVE': [0,0,0,0]}
    :param rdm_level: max RDM to be computed
    :param rdm_type: RDMs type: spin_dependent or spin_free
    :return: a tuple of (reference energy, MOSpaceInfo, ForteIntegrals, RDMs)
    """

    forte_objects = prepare_forte_objects(wfn, mo_spaces)

    ints = forte_objects['ints']
    as_ints = forte_objects['as_ints']
    scf_info = forte_objects['scf_info']
    mo_space_info = forte_objects['mo_space_info']
    state_weights_map = forte_objects['state_weights_map']

    # build a map {StateInfo: a list of weights} for multi-state computations
    state_weights_map = forte.make_state_weights_map(forte.forte_options, mo_space_info)

    # converts {StateInfo: weights} to {StateInfo: nroots}
    state_map = forte.to_state_nroots_map(state_weights_map)

    # create an active space solver object and compute the energy
    as_solver_type = 'FCI'
    as_solver = forte.make_active_space_solver(
        as_solver_type, state_map, scf_info, mo_space_info, as_ints, forte.forte_options
    )

    state_energies_list = as_solver.compute_energy()  # a map {StateInfo: a list of energies}

    # compute averaged energy --- reference energy for DSRG
    Eref = forte.compute_average_state_energy(state_energies_list, state_weights_map)

    # compute RDMs
    rdms = as_solver.compute_average_rdms(state_weights_map, rdm_level, rdm_type)

    # semicanonicalize orbitals
    semi = forte.SemiCanonical(mo_space_info, ints, forte.forte_options)
    semi.semicanonicalize(rdms, rdm_level)

    return {'reference_energy': Eref, 'mo_space_info': mo_space_info, 'ints': ints, 'rdms': rdms}
