from forte.core import flog

from forte.solvers.feature import Feature
from forte.solvers.solver import Solver

from forte import ForteOptions
from forte import forte_options
from forte import make_mcscf_two_step, make_mo_space_info_from_map


class MCSCF(Solver):
    """
    A class to perform orbital optimization
    """
    def __init__(
        self, input_nodes, e_convergence=1.0e-10, r_convergence=1.0e-6, freeze_core=False, options=None, cbh=None
    ):
        """
        Initialize an ActiveSpaceSolver object

        Parameters
        ----------
        input_nodes: Solver
            The solver that will provide the active space functionality and the starting molecular orbitals
        e_convergence: float
            energy convergence criterion
        r_convergence: float
            residual convergence criterion
        freeze_core: bool
            freeze rotations of the core orbitals? (default = False)
        options: dict()
            Additional options passed to control the active space solver
        cbh: CallbackHandler
            A callback object used to inject code into the HF class
        """
        # initialize the base class
        super().__init__(
            input_nodes=input_nodes,
            needs=[Feature.MODEL, Feature.ORBITALS, Feature.ACTIVESPACESOLVER],
            provides=[Feature.MODEL, Feature.ORBITALS, Feature.ACTIVESPACESOLVER, Feature.RDMS],
            options=options,
            cbh=cbh
        )
        # grab a pointer to the data from the input node
        self._data = self.input_nodes[0].data
        self._e_convergence = e_convergence
        self._r_convergence = r_convergence
        self._freeze_core = freeze_core

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'MCSCF(e_convergence={self._e_convergence},r_convergence={self._r_convergence})'

    def __str__(self):
        """
        return a string representation of this object
        """
        return repr(self)

    @property
    def e_convergence(self):
        return self._e_convergence

    @property
    def r_convergence(self):
        return self._r_convergence

    def _run(self):
        """Run an MCSCF computation"""
        # make sure the active space solver executed
        if not self.input_nodes[0].executed:
            flog('info', 'MCSCF: reference not available. Calling run() on input node')
            self.input_nodes[0].run()
        else:
            flog('info', 'MCSCF: reference read from input node')

        # prepare the options
        options = {'E_CONVERGENCE': self.e_convergence, 'R_CONVERGENCE': self.r_convergence}

        # values from self._options (user specified) replace those from options
        full_options = {**options, **self._options}

        flog('info', 'MCSCF: adding options')
        local_options = ForteOptions(forte_options)
        local_options.set_from_dict(full_options)

        if not self._freeze_core:
            mcscf_mo_space_info = self._make_mcscf_mo_space_info()
        else:
            mcscf_mo_space_info = self.mo_space_info

        flog('info', 'MCSCF: making the mcscf object')
        mcscf = make_mcscf_two_step(
            self.input_nodes[0]._states, mcscf_mo_space_info, local_options, self.ints,
            self.input_nodes[0].active_space_solver
        )
        flog('info', 'MCSCF: computing the energy')
        average_energy, energies = mcscf.compute_energy()
        flog('info', f'MCSCF: mcscf average energy = {average_energy}')
        flog('info', f'MCSCF: mcscf energy = {energies}')

        self._results.add('mcscf energy', energies, 'MCSCF energy', 'Eh')

        return self

    def _make_mcscf_mo_space_info(self):
        """This function prepares a MOSpaceInfo info object for a MCSCF computation

        We basically take frozen orbitals and combined them with the restricted ones
        """
        mo_space_dict = {}
        for space in self.mo_space_info.space_names():
            mo_space_dict[space] = list(self.mo_space_info.dimension(space).to_tuple())
        docc = []
        uocc = []
        for x, y in zip(mo_space_dict['FROZEN_DOCC'], mo_space_dict['RESTRICTED_DOCC']):
            docc.append(x + y)
        for x, y in zip(mo_space_dict['FROZEN_UOCC'], mo_space_dict['RESTRICTED_UOCC']):
            uocc.append(x + y)
        mo_space_dict.pop('FROZEN_DOCC')
        mo_space_dict.pop('FROZEN_UOCC')
        mo_space_dict['RESTRICTED_DOCC'] = docc
        mo_space_dict['RESTRICTED_UOCC'] = uocc

        # build a MOSpaceInfo object with frozen MOs merged with the restricted MOs
        nmopi = self.data.scf_info.nmopi()
        point_group = self.model.point_group
        mcscf_mo_info = make_mo_space_info_from_map(nmopi, point_group, mo_space_dict, self.mo_space_info.reorder())

        return mcscf_mo_info
