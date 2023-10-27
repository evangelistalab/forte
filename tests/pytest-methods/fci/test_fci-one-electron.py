import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci_one_electron():
    """Test FCI on a one-electron system"""

    ref_hf_energy = -0.600480545551890
    ref_fci_energy = -0.600480545551890

    xyz = """
    H
    H 1 1.0
    """
    input = solver_factory(molecule=xyz, basis='aug-cc-pVDZ')
    state = input.state(charge=1, multiplicity=2, sym='ag')
    hf = HF(input, state=state, e_convergence=1.0e-12)
    # define an active space
    mo_spaces = input.mo_spaces(
        frozen_docc=[0, 0, 0, 0, 0, 0, 0, 0], restricted_docc=[0, 0, 0, 0, 0, 0, 0, 0], active=[1, 0, 0, 0, 0, 0, 0, 0]
    )
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, e_convergence=1.0e-12, mo_spaces=mo_spaces)
    fci.run()

    # check results
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert fci.value('active space energy')[state] == pytest.approx([ref_fci_energy], 1.0e-10)


if __name__ == "__main__":
    test_fci_one_electron()
