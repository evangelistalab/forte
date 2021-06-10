"""Test the MCSCF solver."""

import forte
import pytest
from forte import Molecule, Basis
from forte.solvers import HF, MCSCF, molecular_model


def test_casscf():
    """Test RHF."""

    ref_energy = -1.1271993998799024

    # create a molecule from a string
    mol = Molecule.from_geom("""
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """)

    # create a basis object
    basis = Basis('cc-pVDZ')

    # create a molecular model
    root = molecular_model(molecule=mol, basis=basis)

    # specify the electronic state
    state = root.model.state(charge=0, multiplicity=1, sym='ag')

    # compute HF orbitals
    hf = HF(root,state=state)

    fci = FCI(states=state, active=[1, 0, 0, 0, 0, 1, 0, 0])
    # create an MCSCF object
    mcscf = MCSCF()
    mcscf.run()

    def run_casscf()

        state = root.model.state(charge=0, multiplicity=1, sym='ag')

        # compute HF orbitals
        hf = HF(root, state=state)

        fci = FCI(states=state, active=[1, 0, 0, 0, 0, 1, 0, 0])
        # create an MCSCF object
        mcscf = MCSCF()
        mcscf.run()
    

    # compute HF orbitals
    hf = HF(job, state=state)
    job2 = hf.run()

    avas = AVAS(job2)
    job3 = avas.run()

    # create an MCSCF object
    mcscf = MCSCF(job3, states=state, active=[1, 0, 0, 0, 0, 1, 0, 0])
    obj4 = mcscf.run()

    assert mcscf.value('mcscf energy')[0] == pytest.approx(ref_energy, 1.0e-10)


if __name__ == "__main__":
    test_casscf()
