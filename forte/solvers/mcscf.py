import psi4

from forte.solvers.solver import Solver
from forte.solvers.hf import HF
from forte.model import Model
from forte.forte import StateInfo
from forte import prepare_forte_options, make_ints_from_psi4, make_mcscf_two_step, make_mcscf


class MCSCF(Solver):
    """
    A class to run MCSCF computations

    Attributes
    ----------
    restricted : bool
        is this a restricted computation
    """
    def __init__(
        self, parent_solver, states, active=None, restricted_docc=None, e_convergence=1.0e-10, d_convergence=1.0e-6
    ):
        """
        initialize a Basis object

        Parameters
        ----------
        restricted : bool
            a basis object
        """
        super().__init__(parent_solver.model)
        # allow passing a single StateInfo object
        if isinstance(states, StateInfo):
            self._states = {states: [1.0]}
        else:
            self._states = states
        self._parent_solver = parent_solver
        self._active = active
        self._restricted_docc = restricted_docc
        self._e_convergence = e_convergence
        self._d_convergence = d_convergence

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'MCSCF(e_convergence={self.e_convergence},d_convergence={self.d_convergence})'

    def __str__(self):
        """
        return a string representation of this object
        """
        return repr(self)

    @property
    def e_convergence(self):
        return self._e_convergence

    @property
    def d_convergence(self):
        return self._d_convergence

    @property
    def guess(self):
        return self._parent_solver

    def run(self):
        # compute the guess orbitals
        self.guess.run()

        mo_space = {}
        if self._active is not None:
            mo_space['ACTIVE'] = self._active
        if self._restricted_docc is not None:
            mo_space['RESTRICTED_DOCC'] = self._restricted_docc
        mo_space_info = self.guess.make_mo_space_info(mo_space)

        options = prepare_forte_options()

        ints = make_ints_from_psi4(self.guess.psi_wfn, options, mo_space_info)

        # pipe output to the file self._output_file
        psi4.core.set_output_file(self._output_file, True)

        casscf = make_mcscf_two_step(self._states, self.guess.scf_info, options, mo_space_info, ints)
        energy = casscf.compute_energy()
        self._results.add('mcscf energy', [energy], 'MCSCF energy', 'Eh')
