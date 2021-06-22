from abc import ABC, abstractmethod

from forte.core import flog, increase_log_depth
from forte.data import Data
from forte.model import MolecularModel
from forte.results import Results
from forte.molecule import Molecule
from forte.basis import Basis

import forte


class Solver(ABC):
    """
    A class used to implement a quantum chemistry solver.

    Solver stores Forte base objects in a data attribute
    and a results object.
    """
    def __init__(self):
        self._executed = False
        self._data = Data()
        self._results = Results()
        self._output_file = 'output.dat'

    # decorate to icrease the log depth
    @increase_log_depth
    def run(self):
        """
        A general solver interface.

        This method is common to all solvers, and in turn it is routed to
        the method ``_run()`` implemented differently in each solver.
        """
        # log call to run()
        flog('info', f'{type(self).__name__}: calling run()')

        # call derived class implementation of _run()
        self._run()

        # log end of run()
        flog('info', f'{type(self).__name__}: run() finished executing')

        # set executed flag
        self._executed = True

    @abstractmethod
    def _run():
        """The actual run function implemented by each method"""

    @property
    def results(self):
        return self._results

    @property
    def data(self):
        return self._data

    @property
    def executed(self):
        return self._executed

    @property
    def psi_wfn(self):
        return self.data.psi_wfn

    @psi_wfn.setter
    def psi_wfn(self, val):
        self.data.psi_wfn = val

    @property
    def scf_info(self):
        return self.data.scf_info

    @scf_info.setter
    def scf_info(self, val):
        self.data.scf_info = val

    @property
    def mo_space_info(self):
        return self.data.mo_space_info

    @mo_space_info.setter
    def mo_space_info(self, val):
        self.data.mo_space_info = val

    @property
    def ints(self):
        return self.data.ints

    @ints.setter
    def ints(self, val):
        self.data.ints = val

    @property
    def as_ints(self):
        return self.data.as_ints

    @as_ints.setter
    def as_ints(self, val):
        self.data.as_ints = val

    @property
    def model(self):
        return self.data.model

    @property
    def output_file(self):
        return self._output_file

    @output_file.setter
    def output_file(self, val):
        self._output_file = val

    def value(self, label):
        return self._results.value(label)

    def state(self, charge: int, multiplicity: int, ms: float = None, sym: str = None):
        return self.data.model.state(charge, multiplicity, ms, sym)

    def _mo_space_info_map(
        self, frozen_docc=None, restricted_docc=None, active=None, restricted_uocc=None, frozen_uocc=None
    ):
        mo_space = {}
        if frozen_docc is not None:
            mo_space['FROZEN_DOCC'] = frozen_docc
        if restricted_docc is not None:
            mo_space['RESTRICTED_DOCC'] = restricted_docc
        if active is not None:
            mo_space['ACTIVE'] = active
        if restricted_uocc is not None:
            mo_space['RESTRICTED_UOCC'] = restricted_uocc
        if frozen_uocc is not None:
            mo_space['FROZEN_UOCC'] = frozen_uocc

        return mo_space

    def make_mo_space_info(self, mo_spaces):
        nmopi = self.data.scf_info.nmopi()
        point_group = self.model.point_group
        reorder = []  # TODO: enable reorder
        self.data.mo_space_info = forte.make_mo_space_info_from_map(nmopi, point_group, mo_spaces, reorder)

    def prepare_forte_options(self):
        """
        Return a ForteOptions object.
        """
        import psi4
        # Get the option object
        psi4_options = psi4.core.get_options()
        psi4_options.set_current_module('FORTE')

        # Get the forte option object
        options = forte.forte_options
        options.get_options_from_psi4(psi4_options)

        # Averaging spin multiplets if doing spin-adapted computation
        if options.get_str('CORRELATION_SOLVER') in ('SA-MRDSRG', 'SA_MRDSRG'):
            options.set_bool('SPIN_AVG_DENSITY', True)

        return options

    def prepare_forte_objects(self, options, name, **kwargs):
        """
        Prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects.

        Parameters
        ----------
        options
            the ForteOptions object
        name
            the name of the module associated with Psi4
        kwargs
            named arguments associated with Psi4
        Return
        ------
            a tuple of (Wavefunction, ForteIntegrals, SCFInfo, MOSpaceInfo, FCIDUMP)
        """
        lowername = name.lower().strip()

        # if 'FCIDUMP' in options.get_str('INT_TYPE'):
        # if 'FIRST' in options.get_str('DERTYPE'):
        #     raise Exception("Energy gradients NOT available for custom integrals!")

        # psi4.core.print_out('\n  Preparing forte objects from a custom source\n')
        # forte_objects = prepare_forte_objects_from_fcidump(options)
        # state_weights_map, mo_space_info, scf_info, fcidump = forte_objects
        # ref_wfn = None
        # else:
        if isinstance(self.model, MolecularModel):
            psi4.core.print_out('\n\n  Preparing forte objects from a Psi4 Wavefunction object')
            ref_wfn, mo_space_info = prepare_psi4_ref_wfn(options, **kwargs)
            forte_objects = prepare_forte_objects_from_psi4_wfn(options, ref_wfn, mo_space_info)
            state_weights_map, mo_space_info, scf_info = forte_objects
            fcidump = None
        else:
            raise RuntimeError('The new driver does not yet implement FCIDUMP')

        return ref_wfn, state_weights_map, mo_space_info, scf_info, fcidump

    def prepare_psi4_ref_wfn(options, **kwargs):
        """
        Prepare a Psi4 Wavefunction as reference for Forte.
        :param options: a ForteOptions object for options
        :param kwargs: named arguments associated with Psi4
        :return: (the processed Psi4 Wavefunction, a Forte MOSpaceInfo object)

        Notes:
            We will create a new Psi4 Wavefunction (wfn_new) if necessary.

            1. For an empty ref_wfn, wfn_new will come from Psi4 SCF or MCSCF.

            2. For a valid ref_wfn, we will test the orbital orthonormality against molecule.
            If the orbitals from ref_wfn are consistent with the active geometry,
            wfn_new will simply be a link to ref_wfn.
            If not, we will rerun a Psi4 SCF and orthogonalize orbitals, where
            wfn_new comes from this new Psi4 SCF computation.
        """
        p4print = psi4.core.print_out

        # grab reference Wavefunction and Molecule from kwargs
        kwargs = p4util.kwargs_lower(kwargs)

        ref_wfn = kwargs.get('ref_wfn', None)

        molecule = kwargs.pop('molecule', psi4.core.get_active_molecule())
        point_group = molecule.point_group().symbol()

        # try to read orbitals from file
        Ca = read_orbitals() if options.get_bool('READ_ORBITALS') else None

        need_orbital_check = True
        fresh_ref_wfn = True if ref_wfn is None else False

        if ref_wfn is None:
            ref_type = options.get_str('REF_TYPE')
            p4print(
                '\n  No reference wave function provided for Forte.'
                f' Computing {ref_type} orbitals using Psi4 ...\n'
            )

            # no warning printing for MCSCF
            job_type = options.get_str('JOB_TYPE')
            do_mcscf = (job_type in ["CASSCF", "MCSCF_TWO_STEP"] or options.get_bool("CASSCF_REFERENCE"))

            # run Psi4 SCF or MCSCF
            ref_wfn = run_psi4_ref(ref_type, molecule, not do_mcscf, **kwargs)

            need_orbital_check = False if Ca is None else True
        else:
            # Ca from file has higher priority than that of ref_wfn
            Ca = ref_wfn.Ca().clone() if Ca is None else Ca

        # build Forte MOSpaceInfo
        nmopi = ref_wfn.nmopi()
        mo_space_info = forte.make_mo_space_info(nmopi, point_group, options)

        # do we need to check MO overlap?
        if not need_orbital_check:
            wfn_new = ref_wfn
        else:
            # test if input Ca has the correct dimension
            if Ca.rowdim() != nmopi or Ca.coldim() != nmopi:
                raise ValueError("Invalid orbitals: different basis set / molecule")

            new_S = psi4.core.Wavefunction.build(molecule, options.get_str("BASIS")).S()

            if check_MO_orthonormality(new_S, Ca):
                wfn_new = ref_wfn
                wfn_new.Ca().copy(Ca)
            else:
                if fresh_ref_wfn:
                    wfn_new = ref_wfn
                    wfn_new.Ca().copy(ortho_orbs_forte(wfn_new, mo_space_info, Ca))
                else:
                    p4print("\n  Perform new SCF at current geometry ...\n")

                    kwargs_copy = {k: v for k, v in kwargs.items() if k != 'ref_wfn'}
                    wfn_new = run_psi4_ref('scf', molecule, False, **kwargs_copy)

                    # orthonormalize orbitals
                    wfn_new.Ca().copy(ortho_orbs_forte(wfn_new, mo_space_info, Ca))

                    # copy wfn_new to ref_wfn
                    ref_wfn.shallow_copy(wfn_new)

        # set DF and MINAO basis
        if 'DF' in options.get_str('INT_TYPE'):
            aux_basis = psi4.core.BasisSet.build(
                molecule, 'DF_BASIS_MP2', options.get_str('DF_BASIS_MP2'), 'RIFIT', options.get_str('BASIS')
            )
            wfn_new.set_basisset('DF_BASIS_MP2', aux_basis)

        if options.get_str('MINAO_BASIS'):
            minao_basis = psi4.core.BasisSet.build(molecule, 'MINAO_BASIS', options.get_str('MINAO_BASIS'))
            wfn_new.set_basisset('MINAO_BASIS', minao_basis)

        return wfn_new, mo_space_info


class BasicSolver(Solver):
    """
    This solver class is used as a starting point of computations.
    
    When initialized, this solver does not contain any information.
    It is used by the function `solver_factory` which fills it with
    information about a model.
    """
    def __init__(self):
        super().__init__()

    def _run(self):
        pass


def solver_factory(molecule, basis, scf_aux_basis=None, corr_aux_basis=None):
    """A factory to build a basic solver object"""
    flog('info', 'Calling solver factory')

    # TODO: generalize to other type of models (e.g. if molecule/basis are not provided)

    # convert string arguments to objects if necessary
    if isinstance(molecule, str):
        molecule = Molecule.from_geom(molecule)
    if isinstance(basis, str):
        basis = Basis(basis)
    if isinstance(scf_aux_basis, str):
        scf_aux_basis = Basis(scf_aux_basis)
    if isinstance(corr_aux_basis, str):
        corr_aux_basis = Basis(corr_aux_basis)

    # create an empty solver and pass the model in
    solver = BasicSolver()
    solver.data.model = MolecularModel(
        molecule=molecule, basis=basis, scf_aux_basis=scf_aux_basis, corr_aux_basis=corr_aux_basis
    )
    return solver
