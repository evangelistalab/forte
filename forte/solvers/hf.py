from forte.solvers.solver import Solver
from forte.model import MolecularModel
from forte.forte import SCFInfo


class HF(Solver):
    """
    A class to run Hartree-Fock computations
    """
    def __init__(
        self,
        parent_solver,
        state,
        restricted=True,
        e_convergence=1.0e-10,
        d_convergence=1.0e-6,
        int_type='conventional',
        docc=None,
        socc=None,
        options=None
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
        int_type: str
            the type of integrals used in the HF procedure (conventional = pk, direct, df, ...)
        docc: list(int)
            The number of doubly occupied orbitals per irrep
        socc: list(int)
            The number of singly occupied orbitals per irrep
        options: dict()
            Additional options passed to control psi4
        """
        # initialize common objects
        super().__init__()
        self._data = parent_solver.data
        self._state = state
        self._restricted = restricted
        self._e_convergence = e_convergence
        self._d_convergence = d_convergence
        self._int_type = int_type
        self._docc = docc
        self._socc = socc
        self._options = {} if options is None else options

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
        molecule = self.data.model.molecule
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

    @property
    def docc(self):
        return self._docc

    @property
    def socc(self):
        return self._socc

    def run(self):
        """Compute the Hartree-Fock energy"""
        import psi4
        # currently limited to molecules
        if not isinstance(self.data.model, MolecularModel):
            raise RuntimeError('HF.energy() is implemented only for MolecularModel objects')

        molecule = self.data.model.molecule
        molecule.set_molecular_charge(self.charge)
        molecule.set_multiplicity(self.multiplicity)

        # prepare options for psi4
        scf_type_dict = {
            'CONVENTIONAL': 'PK',
            'STD': 'PK',
        }

        if self._int_type.upper() in scf_type_dict:
            scf_type = scf_type_dict[self._int_type.upper()]
        else:
            scf_type = self._int_type.upper()

        options = {
            'BASIS': self.data.model.basis,
            'REFERENCE': 'RHF' if self._restricted else 'UHF',
            'SCF_TYPE': scf_type,
            'E_CONVERGENCE': self.e_convergence,
            'D_CONVERGENCE': self.d_convergence
        }

        # optionally specify docc/socc
        if self.docc is not None:
            options['DOCC'] = self.docc
        if self.socc is not None:
            options['SOCC'] = self.socc

        full_options = {**options, **self._options}

        # set the options
        psi4.set_options(full_options)

        # pipe output to the file self._output_file
        psi4.core.set_output_file(self._output_file, True)

        # run scf and return the energy and a wavefunction object
        energy, psi_wfn = psi4.energy('scf', molecule=molecule, return_wfn=True)

        # add the energy to the results
        self._results.add('hf energy', energy, 'Hartree-Fock energy', 'Eh')

        # store calculation outputs in the Data object
        self.data.psi_wfn = psi_wfn
        self.data.scf_info = SCFInfo(psi_wfn)

        # set executed flag
        self._executed = True

        return self
