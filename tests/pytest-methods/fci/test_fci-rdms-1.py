import pytest
import psi4

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci_rdms_1():
    """Test the FCI 2- and 3-RDMs on LiH/STO-3G"""

    # setup job
    xyz = """
    Li
    H 1 3.0
    units bohr
    """
    input = solver_factory(molecule=xyz, basis='sto-3g')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, options={'fci_test_rdms': True})
    fci.run()

    # check results
    print("Testing AAAA 2-RDM")
    assert psi4.core.variable("AAAA 2-RDM ERROR") == pytest.approx(0.0, 1.0e-10)
    print("Testing BBBB 2-RDM")
    assert psi4.core.variable("BBBB 2-RDM ERROR") == pytest.approx(0.0, 1.0e-10)
    print("Testing ABAB 2-RDM")
    assert psi4.core.variable("ABAB 2-RDM ERROR") == pytest.approx(0.0, 1.0e-10)

    print("Testing AAAAAA 3-RDM")
    assert psi4.core.variable("AAAAAA 3-RDM ERROR") == pytest.approx(0.0, 1.0e-10)
    print("Testing AABAAB 3-RDM")
    assert psi4.core.variable("AABAAB 3-RDM ERROR") == pytest.approx(0.0, 1.0e-10)
    print("Testing ABBABB 3-RDM")
    assert psi4.core.variable("ABBABB 3-RDM ERROR") == pytest.approx(0.0, 1.0e-10)
    print("Testing BBBBBB 3-RDM")
    assert psi4.core.variable("BBBBBB 3-RDM ERROR") == pytest.approx(0.0, 1.0e-10)


if __name__ == "__main__":
    test_fci_rdms_1()
