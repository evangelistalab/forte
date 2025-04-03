import pytest
import forte.apps


def test_mcscf_app1():
    """
    Test the MCSCF app with an active space is defined by the number of orbitals and electrons.
    """
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    """
    data = forte.apps.hf(geom=xyz, basis="cc-pVDZ", state={"charge": 0, "multiplicity": 1, "sym": "ag"})

    data = forte.apps.mcscf(
        data,
        active_space={"norb": 2, "nel": 2},
        solver_type="GENCI",
    )
    assert data.results.value("mcscf energy [0]") == pytest.approx(-1.0561253825629822, 1.0e-10)


def test_mcscf_app2():
    """
    Test the MCSCF app with an active space is defined by the number of orbitals per irrep.
    """

    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    """
    data = forte.apps.hf(geom=xyz, basis="cc-pVDZ", state={"charge": 0, "multiplicity": 1, "sym": "ag"})

    data = forte.apps.mcscf(
        data,
        states={"charge": 0, "multiplicity": 1, "sym": "ag"},
        active_space={"active": [1, 0, 0, 0, 0, 1, 0, 0]},
        solver_type="GENCI",
    )
    assert data.results.value("mcscf energy [0]") == pytest.approx(-1.0561253825629822, 1.0e-10)


def test_mcscf_app3():
    """
    Test the MCSCF app with an active space is defined by the specific irrep and orbital index (within the irrep).
    """
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    """
    data = forte.apps.hf(geom=xyz, basis="cc-pVDZ", state={"charge": 0, "multiplicity": 1, "sym": "ag"})

    data = forte.apps.mcscf(
        data,
        active_space={"active_orbitals": ["1 Ag", "1-B1u"], "nel": 2},
        solver_type="GENCI",
    )
    assert data.results.value("mcscf energy [0]") == pytest.approx(-1.0561253825629822, 1.0e-10)
    data.scf_info.reorder_orbitals([[0, 1, 2], [], [0], [0], [], [1, 0, 2], [0], [0]])


def test_mcscf_app4():
    """
    Test the MCSCF app on a state-averaged computation.
    """
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    """
    data = forte.apps.hf(geom=xyz, basis="cc-pVDZ", state={"charge": 0, "multiplicity": 1, "sym": "ag"})

    data = forte.apps.mcscf(
        data,
        states=[{"charge": 0, "multiplicity": 1, "sym": "ag", "weights": [0.5, 0.5]}],
        active_space={"active_orbitals": ["1 Ag", "1 B1u"], "nel": 2},
        solver_type="GENCI",
    )
    assert data.results.value("mcscf energy [0]") == pytest.approx(-1.045347524713, 1.0e-10)
    assert data.results.value("mcscf energy [1]") == pytest.approx(-0.541334565331, 1.0e-10)


if __name__ == "__main__":
    test_mcscf_app1()
    test_mcscf_app2()
    test_mcscf_app3()
    test_mcscf_app4()
