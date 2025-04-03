import pytest
import forte.apps


def test_mcscf_app1():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    """
    data = forte.apps.mcscf(
        geom=xyz,
        basis="cc-pVDZ",
        state={"charge": 0, "multiplicity": 1, "sym": "ag"},
        active_space={"norb": 2, "nel": 2},
        solver_type="GENCI",
    )
    assert data.results.value("mcscf energy") == pytest.approx(-1.0561253825629822, 1.0e-10)


def test_mcscf_app2():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    """
    data = forte.apps.mcscf(
        geom=xyz,
        basis="cc-pVDZ",
        state={"charge": 0, "multiplicity": 1, "sym": "ag"},
        active_space={"active": [1, 0, 0, 0, 0, 1, 0, 0]},
        solver_type="GENCI",
    )
    assert data.results.value("mcscf energy") == pytest.approx(-1.0561253825629822, 1.0e-10)


def test_mcscf_app3():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.5
    """
    data = forte.apps.mcscf(
        geom=xyz,
        basis="cc-pVDZ",
        state={"charge": 0, "multiplicity": 1, "sym": "ag"},
        active_space={"active_orbitals": ["1 Ag", "1 B1u"], "nel": 2},
        solver_type="GENCI",
    )
    assert data.results.value("mcscf energy") == pytest.approx(-1.0561253825629822, 1.0e-10)
    data.scf_info.reorder_orbitals([[0, 1, 2], [], [0], [0], [], [1, 0, 2], [0], [0]])


# def test_mcscf_app4():
#     xyz = """
#     H 0.0 0.0 0.0
#     H 0.0 0.0 1.5
#     """
#     data = forte.apps.run_mcscf(
#         geom=xyz,
#         basis="cc-pVDZ",
#         states=[{"charge": 0, "multiplicity": 1, "sym": "ag", "weights": [0.5, 0.5]}],
#         active_space={"active_orbitals": ["1 Ag", "1 B1u"], "nel": 2},
#         solver_type="GENCI",
#     )
#     assert data.results.value("mcscf energy") == pytest.approx(-1.0561253825629822, 1.0e-10)
#     data.scf_info.reorder_orbitals([[0, 1, 2], [], [0], [0], [], [1, 0, 2], [0], [0]])


if __name__ == "__main__":
    test_mcscf_app1()
    test_mcscf_app2()
    test_mcscf_app3()
