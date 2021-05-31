from forte.method import Method
from forte.model import Model
from forte.results import Results


class MCSCF(Method):
    """
    A class to run Hartree-Fock computations

    Attributes
    ----------
    restricted : bool
        is this a restricted computation
    """
    def __init__(self, e_convergence=1.0e-10, d_convergence=1.0e-6):
        """
        initialize a Basis object

        Parameters
        ----------
        restricted : bool
            a basis object
        """
        self._e_convergence = e_convergence
        self._d_convergence = d_convergence

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'MCSCF(e_convergence={self._e_convergence},d_convergence={self._d_convergence})'

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

    def energy(self, model: Model, results: Results):
        import psi4
        """Compute the energy"""
        options = {
            'BASIS': model.basis,
            'REFERENCE': 'RHF' if self._restricted else 'UHF',
            'SCF_TYPE': 'pk',
            'E_CONVERGENCE': self._e_convergence,
            'D_CONVERGENCE': self._d_convergence
        }

        # set the options
        psi4.set_options(options)

        # pipe output to the file psi4.hf.output.txt
        psi4.core.set_output_file('psi4.hf.output.txt', True)

        # run scf and return the energy and a wavefunction object
        E_scf, wfn = psi4.energy('scf', return_wfn=True)

        results.add('hf energy', E_scf, 'Hartree-Fock energy', 'Eh')
