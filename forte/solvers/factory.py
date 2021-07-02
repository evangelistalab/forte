from forte.core import flog
from forte.model import MolecularModel
from forte.molecule import Molecule
from forte.basis import Basis
from forte.solvers.input import Input


def solver_factory(molecule, basis, int_type=None, jkfit_aux_basis=None, rifit_aux_basis=None):
    """A factory to build a basic solver object"""
    flog('info', 'Calling solver factory')

    # TODO: generalize to other type of models (e.g. if molecule/basis are not provided)

    # convert string arguments to objects if necessary
    if isinstance(molecule, str):
        molecule = Molecule.from_geom(molecule)
    if isinstance(basis, str):
        basis = Basis(basis)
    if isinstance(jkfit_aux_basis, str):
        jkfit_aux_basis = Basis(jkfit_aux_basis)
    if isinstance(rifit_aux_basis, str):
        rifit_aux_basis = Basis(rifit_aux_basis)

    # create an empty solver and pass the model in
    solver = Input()
    solver.data.model = MolecularModel(
        molecule=molecule,
        int_type=int_type,
        basis=basis,
        jkfit_aux_basis=jkfit_aux_basis,
        rifit_aux_basis=rifit_aux_basis
    )
    return solver
