from forte.solvers.solver import Solver
from forte.model import Model, MolecularModel
from forte.forte import SCFInfo


class HF(Solver):
    """
    A class to run Hartree-Fock computations

    Attributes
    ----------
    restricted : bool
        is this a restricted computation
    """
    def __init__(
        self,
        parent_solver,
        state,
        restricted=True,
        e_convergence=1.0e-10,
        d_convergence=1.0e-6,
    ):
        """
        initialize a Basis object

        Parameters
        ----------
        restricted : bool
            do restricted HF?
        e_convergence: float
            energy convergence criterion
        d_convergence: float
            density matrix convergence criterion                        
        """
        # initialize common objects
        super().__init__()
        self.copy(parent_solver)
        self._state = state
        self._restricted = restricted
        self._e_convergence = e_convergence
        self._d_convergence = d_convergence

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'HF(restricted={self._restricted},e_convergence={self._e_convergence},d_convergence={self._d_convergence})'

    def __str__(self):
        """
        return a string representation of this object
        """
        return repr(self)

    @property
    def restricted(self):
        return self._restricted

    @property
    def state(self):
        return self._state

    @property
    def charge(self):
        # compute the number of electrons
        molecule = self.model.molecule
        natom = molecule.natom()
        charge = round(sum([molecule.Z(i) for i in range(natom)])) - self.state.na() - self.state.nb()
        return charge

    @property
    def multiplicity(self):
        return self.state.multiplicity()

    @property
    def e_convergence(self):
        return self._e_convergence

    @property
    def d_convergence(self):
        return self._d_convergence

    def run(self):
        """Compute the energy using psi4"""
        import psi4
        if not isinstance(self.model, MolecularModel):
            raise RuntimeError('HF.energy() is implemented only for MolecularModel objects')

        self.model.molecule.set_molecular_charge(self.charge)
        self.model.molecule.set_multiplicity(self.multiplicity)

        options = {
            'BASIS': self.model.basis,
            'REFERENCE': 'RHF' if self._restricted else 'UHF',
            'SCF_TYPE': 'pk',
            'E_CONVERGENCE': self.e_convergence,
            'D_CONVERGENCE': self.d_convergence
        }

        # set the options
        psi4.set_options(options)

        # pipe output to the file self._output_file
        psi4.core.set_output_file(self._output_file, True)

        # run scf and return the energy and a wavefunction object
        energy, psi_wfn = psi4.energy('scf', return_wfn=True)

        # add the energy to the results
        self._results.add('hf energy', energy, 'Hartree-Fock energy', 'Eh')

        # add objects to the collection
        self.psi_wfn = psi_wfn
        self.scf_info = SCFInfo(psi_wfn)

        return self
