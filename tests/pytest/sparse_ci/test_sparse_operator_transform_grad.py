import math
import numpy as np
import forte


def test_sparse_operator_transform_grad_1():
    O = forte.sparse_operator("[0a+ 0a-]", 1.0)
    T = forte.operator_list("[1a+ 0a-]", 0.1)
    forte.fact_unitary_trans_antiherm(O, T, 0)
    assert np.isclose(O["[0a+ 0a-]"], 0.9900332889206208)
    assert np.isclose(O["[1a+ 0a-]"], -0.0993346653975306)
    assert np.isclose(O["[0a+ 1a-]"], -0.0993346653975306)
    assert np.isclose(O["[1a+ 1a-]"], 0.00996671107937919)

    O1 = forte.sparse_operator("[0a+ 0a-]", 1.0)
    T = forte.operator_list("[1a+ 0a-]", 0.0001)
    forte.fact_unitary_trans_antiherm(O1, T, 0)
    O2 = forte.sparse_operator("[0a+ 0a-]", 1.0)
    T = forte.operator_list("[1a+ 0a-]", -0.0001)
    forte.fact_unitary_trans_antiherm(O2, T, 0)
    grad = (O1 - O2) / 0.0002
    print(f"{grad = }")

    O = forte.sparse_operator("[0a+ 0a-]", 1.0)
    T = forte.operator_list("[1a+ 0a-]", 0.0)
    forte.fact_unitary_trans_antiherm_grad(O, T, 0)

    print(f"{O = }")

    assert np.isclose(O["[0a+ 0a-]"], 0)
    assert np.isclose(O["[1a+ 0a-]"], -1)
    assert np.isclose(O["[0a+ 1a-]"], -1)
    assert np.isclose(O["[1a+ 1a-]"], 0)


if __name__ == "__main__":
    test_sparse_operator_transform_grad_1()
