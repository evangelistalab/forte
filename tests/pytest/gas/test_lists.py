import forte
import pytest
import psi4
from forte import det


def test_gas_lists1():
    nmopi = psi4.core.Dimension([1, 2, 0, 0])
    mo_space_map = {"GAS1": [1, 0, 0, 0], "GAS2": [0, 2, 0, 0]}
    mo_space_info = forte.make_mo_space_info_from_map(nmopi, "C2V", mo_space_map)
    na = 2
    nb = 2
    gas_min = [0, 0]
    gas_max = [2, 2]
    symmetry = 0
    gas_lists = forte.GenCIStringLists(mo_space_info, na, nb, symmetry, forte.PrintLevel.Debug, gas_min, gas_max)
    gas_vector = forte.GenCIVector(gas_lists)
    gas_vector.set_to(1.0)
    # gas_vector.print()
    state_vector = gas_vector.as_state_vector()
    assert len(state_vector) == 4
    state_vector_test = forte.StateVector({det("220"): 1.0, det("202"): 1.0, det("2+-"): 1.0, det("2-+"): 1.0})
    assert state_vector == state_vector_test


def test_gas_lists2():
    nmopi = psi4.core.Dimension([2, 1, 2, 0])
    mo_space_map = {"GAS1": [1, 0, 1, 0], "GAS2": [1, 1, 1, 0]}
    mo_space_info = forte.make_mo_space_info_from_map(nmopi, "C2V", mo_space_map)
    na = 3
    nb = 3
    gas_min = [3, 3]
    gas_max = [3, 3]
    symmetry = 0
    gas_lists = forte.GenCIStringLists(mo_space_info, na, nb, symmetry, forte.PrintLevel.Default, gas_min, gas_max)
    dets = gas_lists.make_determinants()
    dets.sort()
    test_dets = [
        det("+-022"),
        det("220+-"),
        det("+-220"),
        det("202+-"),
        det("-+022"),
        det("220-+"),
        det("-+220"),
        det("202-+"),
    ]
    test_dets.sort()
    assert dets == test_dets

    gas_vector = forte.GenCIVector(gas_lists)
    gas_vector.set_to(1.0)
    state_vector = gas_vector.as_state_vector()
    assert len(state_vector) == 8
    state_vector_test = forte.StateVector(
        {
            det("+-022"): 1.0,
            det("220+-"): 1.0,
            det("+-220"): 1.0,
            det("202+-"): 1.0,
            det("-+022"): 1.0,
            det("220-+"): 1.0,
            det("-+220"): 1.0,
            det("202-+"): 1.0,
        }
    )
    assert state_vector == state_vector_test

    gas_vector.print(0.0)


if __name__ == "__main__":
    test_gas_lists1()
    test_gas_lists2()
