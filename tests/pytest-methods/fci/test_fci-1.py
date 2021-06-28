import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci_1():
    """Test FCI on Li2/STO-3G"""

    ref_hf_energy = -14.54873910108353
    ref_fci_energy = -14.595808852754054

    # setup job
    xyz = """
    Li
    Li 1 3.0
    units bohr
    """
    input = solver_factory(molecule=xyz, basis='sto-3g')
    state = input.state(charge=0, multiplicity=1, sym='ag')
    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state)
    fci.run()

    # check results
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert fci.value('active space energy')[state] == pytest.approx([ref_fci_energy], 1.0e-10)


if __name__ == "__main__":
    test_fci_1()
