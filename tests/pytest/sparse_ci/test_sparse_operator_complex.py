import math
import forte
import numpy as np


def st_taylor(O, S):
    """Compute stO = exp(-S) O exp(S) numerically using the Taylor expansion"""
    stO = forte.SparseOperator(O)
    C = forte.SparseOperator(O)
    for i in range(1, 50):
        C = complex(1.0 / i) * C.commutator(S)
        stO += C
        if abs(C.norm()) < 1e-16:
            break
    return stO


def op_exp(S):
    """Compute the exponential of a sparse operator numerically using the Taylor expansion"""
    expS = forte.sparse_operator("[]", 1.0)
    powS = forte.SparseOperator(S)
    for k in range(20):
        powS /= complex(k + 1, 0)
        expS += powS
        if powS.norm() < 1e-15:
            break
        powS = powS @ S
    return expS


def test_sparse_operator_complex_1():
    """test product of two sparse operators"""
    A = forte.sparse_operator("[0a+ 0a-]", 2.0 + 3.0j)
    B = forte.sparse_operator("[0b+ 0b-]", 5.0 - 7.0j)
    C = A @ B
    assert C["[0a+ 0b+ 0b- 0a-]"] == 31.0 + 1.0j


def test_sparse_operator_complex_2():
    """test exponentiation of a sparse operator numerically"""
    theta = 0.1
    A = forte.sparse_operator([("[1a+ 0a-]", theta * 1j), ("[0a+ 1a-]", theta * 1j)])
    expA = op_exp(A)
    assert expA["[]"] == 1.0
    assert np.isclose(expA["[1a+ 0a-]"], 0.09983341664682815j, atol=1e-15)
    assert np.isclose(expA["[0a+ 1a-]"], 0.09983341664682815j, atol=1e-15)
    assert np.isclose(expA["[1a+ 1a-]"], -0.0049958347219741794, atol=1e-15)
    assert np.isclose(expA["[0a+ 0a-]"], -0.0049958347219741794, atol=1e-15)
    assert np.isclose(expA["[0a+ 1a+ 1a- 0a-]"], 0.0049958347219741794 * 2.0, atol=1e-15)

    mA = forte.sparse_operator([("[1a+ 0a-]", -theta * 1j), ("[0a+ 1a-]", -theta * 1j)])
    expmA = op_exp(mA)
    assert expmA["[]"] == 1.0
    assert np.isclose(expmA["[1a+ 0a-]"], -0.09983341664682815j, atol=1e-15)
    assert np.isclose(expmA["[0a+ 1a-]"], -0.09983341664682815j, atol=1e-15)
    assert np.isclose(expmA["[1a+ 1a-]"], -0.0049958347219741794, atol=1e-15)
    assert np.isclose(expmA["[0a+ 0a-]"], -0.0049958347219741794, atol=1e-15)
    assert np.isclose(expmA["[0a+ 1a+ 1a- 0a-]"], 0.0049958347219741794 * 2.0, atol=1e-15)

    # test exp(-i theta S) exp(i theta S) = 1
    AmA = expA @ expmA
    assert AmA["[]"] == 1.0
    mAA = expmA @ expA
    assert mAA["[]"] == 1.0

    # test exp(i theta S) a+(g1) a-(g0) exp(-i theta S) against the analytical result
    O = forte.sparse_operator("[1a+ 0a-]", 1.0)
    STO1 = expA @ O @ expmA
    STO2 = st_taylor(O, mA)

    sint = math.sin(theta)
    costm1 = math.cos(theta) - 1.0
    # O
    STO_test = forte.sparse_operator("[1a+ 0a-]", 1.0)
    # -i sin(theta) [O, S]
    STO_test += forte.sparse_operator("[1a+ 1a-]", -1j * sint)
    STO_test += forte.sparse_operator("[0a+ 0a-]", +1j * sint)
    # + (cos(theta) - 1) (SSO + OSS)
    STO_test += forte.sparse_operator("[1a+ 0a-]", 2 * costm1)
    # + sin(theta)^2 SOS
    STO_test += forte.sparse_operator("[0a+ 1a-]", sint**2)
    # i sin(theta) (cos(theta) - 1) S[O,S]S
    STO_test += forte.sparse_operator("[0a+ 0a-]", 1j * sint * costm1)
    STO_test += forte.sparse_operator("[1a+ 1a-]", -1j * sint * costm1)
    # + cos(theta)^2 NON
    STO_test += forte.sparse_operator("[1a+ 0a-]", costm1**2)

    assert np.isclose((STO1 - STO_test).norm(), 0.0, atol=1e-15)
    assert np.isclose((STO2 - STO_test).norm(), 0.0, atol=1e-15)

    STO3 = forte.SparseOperator(O)
    T = forte.SparseOperatorList()
    T.add("[1a+ 0a-]", theta)
    forte.fact_unitary_trans_imagherm(STO3, T)

    assert np.isclose((STO3 - STO_test).norm(), 0.0, atol=1e-15)


def test_sparse_operator_complex_3():
    """test exponentiation of a sparse operator numerically"""

    theta = 0.1
    A = forte.sparse_operator([("[1a+ 0a-]", theta * 1j), ("[0a+ 1a-]", theta * 1j)])
    expA = op_exp(A)
    mA = forte.sparse_operator([("[1a+ 0a-]", -theta * 1j), ("[0a+ 1a-]", -theta * 1j)])
    expmA = op_exp(mA)

    # test exp(-i theta S) exp(i theta S) = 1
    AmA = expA @ expmA
    assert AmA["[]"] == 1.0
    mAA = expmA @ expA
    assert mAA["[]"] == 1.0

    # test exp(i theta S) a+(g1) a-(g1) exp(-i theta S) against the analytical result
    O = forte.sparse_operator("[1a+ 1a-]", 1.0)
    STO1 = expA @ O @ expmA
    STO2 = st_taylor(O, mA)

    sint = math.sin(theta)
    costm1 = math.cos(theta) - 1.0

    STO_test = forte.SparseOperator()
    # O
    STO_test += forte.sparse_operator("[1a+ 1a-]", 1.0)
    # -i sin(theta) [O, S]
    STO_test += forte.sparse_operator("[1a+ 0a-]", -1j * sint)
    STO_test += forte.sparse_operator("[0a+ 1a-]", +1j * sint)
    # + (cos(theta) - 1) (SSO + OSS)
    STO_test += forte.sparse_operator("[0a+ 1a+ 1a- 0a-]", -2 * costm1)
    STO_test += forte.sparse_operator("[1a+ 1a-]", 2 * costm1)
    # + sin(theta)^2 SOS
    STO_test += forte.sparse_operator("[0a+ 1a+ 1a- 0a-]", -(sint**2))
    STO_test += forte.sparse_operator("[0a+ 0a-]", +(sint**2))
    # # i sin(theta) (cos(theta) - 1) S[O,S]S
    STO_test += forte.sparse_operator("[0a+ 1a-]", 1j * sint * costm1)
    STO_test += forte.sparse_operator("[1a+ 0a-]", -1j * sint * costm1)
    # + cos(theta)^2 NON
    STO_test += forte.sparse_operator("[0a+ 1a+ 1a- 0a-]", -(costm1**2))
    STO_test += forte.sparse_operator("[1a+ 1a-]", +(costm1**2))

    # assert np.isclose((STO1 - STO_test).norm(), 0.0, atol=1e-15)
    # assert np.isclose((STO2 - STO_test).norm(), 0.0, atol=1e-15)

    STO3 = forte.SparseOperator(O)
    T = forte.SparseOperatorList()
    T.add("[1a+ 0a-]", theta)
    forte.fact_unitary_trans_imagherm(STO3, T)

    assert np.isclose((STO3 - STO_test).norm(), 0.0, atol=1e-15)


def make_O():
    """Make a random sparse operator"""
    O = forte.SparseOperator()
    O = forte.sparse_operator("[0a+ 0a-]", 2.0 + 3.0j)
    O += forte.sparse_operator("[1a+ 1a-]", 0.3 - 0.5j)
    O += forte.sparse_operator("[2a+ 2a-]", 0.35 - 0.7j)
    O += forte.sparse_operator("[1a+ 0a-]", -0.9 + 0.11j)
    O += forte.sparse_operator("[0a+ 1a-]", -0.19 - 0.31j)
    O += forte.sparse_operator("[0a+ 1a+ 1a- 0a-]", -0.21 + 0.1j)
    O += forte.sparse_operator("[0a+ 1a+ 2a- 1a-]", -0.21 + 0.1j)
    O += forte.sparse_operator("[0a+ 2a+ 1a- 0a-]", +0.23 + 0.3j)
    O += forte.sparse_operator("[0a+ 2a+ 2a- 1a-]", -0.21 + 0.5j)
    O += forte.sparse_operator("[0a+ 1a+ 3a- 2a-]", -0.11 + 0.7j)
    O += forte.sparse_operator("[0a+ 2a+ 3a- 1a-]", -0.21 + 1.1j)
    O += forte.sparse_operator("[0a+ 1a+ 2a+ 2a- 1a- 0a-]", -0.21 + 1.13j)
    O += forte.sparse_operator("[0a+ 1a+ 3a+ 2a- 1a- 0a-]", -0.41 + 1.17j)
    O += forte.sparse_operator("[0a+ 2a+ 3a+ 2a- 1a- 0a-]", -0.52 + 0.98j)
    O += forte.sparse_operator("[0a+ 1a+ 3a+ 3a- 1a- 0a-]", -0.21 + 0.89j)
    return O


def test_sparse_operator_complex_transform1():
    """Test exponentiation of a sparse operator numerically"""
    O = make_O()
    op = "[1a+ 0a-]"
    theta = 0.35711
    run_sparse_operator_test(O, theta, op)


def test_sparse_operator_complex_transform2():
    """Test exponentiation of a sparse operator numerically"""
    O = make_O()
    op = "[1a+ 2a+ 2a- 0a-]"
    theta = 0.35711
    run_sparse_operator_test(O, theta, op)


def test_sparse_operator_complex_transform3():
    """test exponentiation of a sparse operator numerically"""
    O = make_O()
    op = "[1a+ 2a+ 3a- 0a-]"
    theta = 0.35711
    run_sparse_operator_test(O, theta, op)


def test_sparse_operator_complex_transform4():
    """test exponentiation of a sparse operator numerically"""
    O = make_O()
    op = "[0a+ 0a-]"
    theta = 0.35711
    run_sparse_operator_test(O, theta, op)


def test_sparse_operator_complex_transform5():
    """test exponentiation of a sparse operator numerically"""
    O = make_O()
    op = "[0a+ 1a+ 1a- 0a-]"
    theta = 0.35711
    run_sparse_operator_test(O, theta, op)


def test_sparse_operator_complex_transform5():
    """test exponentiation of a sparse operator numerically"""
    O = make_O()
    op = "[0a+ 1a+ 2a+ 2a- 1a- 0a-]"
    theta = 0.35711
    run_sparse_operator_test(O, theta, op)


def run_sparse_operator_test(O, theta, op):
    """test exponentiation of a single sparse operator numerically"""
    sqop, _ = forte.sqop(op)
    sqop_dagger = sqop.adjoint()

    T = forte.SparseOperatorList()
    T.add(op, theta)
    A = forte.sparse_operator([(sqop, theta), (sqop_dagger, -theta)])
    S = forte.sparse_operator([(sqop, -1j * theta), (sqop_dagger, -1j * theta)])

    # test transformations generated by the anti-Hermitian combination of two generators
    # A = T - T^dagger
    # exp(-theta A) O exp(theta A)
    STO_antiherm_taylor = st_taylor(O, A)
    STO_antiherm_analytical = forte.SparseOperator(O)
    forte.fact_unitary_trans_antiherm(STO_antiherm_analytical, T)
    assert np.isclose((STO_antiherm_taylor - STO_antiherm_analytical).norm(), 0.0, atol=1e-12)

    # test the gradient of the transformation generated by the anti-Hermitian combination of two generators
    # A = T - T^dagger
    # d/d theta exp(-theta A) O exp(theta A)
    dtheta = 1e-6
    Ap = forte.sparse_operator([(sqop, theta + dtheta), (sqop_dagger, -theta - dtheta)])
    Am = forte.sparse_operator([(sqop, theta - dtheta), (sqop_dagger, -theta + dtheta)])
    STO_antiherm_taylor_p = st_taylor(O, Ap)
    STO_antiherm_taylor_m = st_taylor(O, Am)
    STO_antiherm_grad_taylor = (STO_antiherm_taylor_p - STO_antiherm_taylor_m) / (2 * dtheta)
    STO_antiherm_grad_analytical = forte.SparseOperator(O)
    forte.fact_unitary_trans_antiherm_grad(STO_antiherm_grad_analytical, T, 0)
    assert np.isclose((STO_antiherm_grad_taylor - STO_antiherm_grad_analytical).norm(), 0.0, atol=1e-6)

    # test transformations generated by the Hermitian combination of two generators
    # S = (T + T^dagger)
    # exp(i theta S) O exp(-i theta S)
    STO_imagherm_taylor = st_taylor(O, S)
    STO_imagherm_analytical = forte.SparseOperator(O)
    forte.fact_unitary_trans_imagherm(STO_imagherm_analytical, T)

    error_norm = (STO_imagherm_taylor - STO_imagherm_analytical).norm()
    if not np.isclose(error_norm, 0.0, atol=1e-12):
        print(f"|STO_antiherm_taylor - STO_antiherm_analytical| = {error_norm}")
        print(f"\nSTO_imagherm_taylor = {STO_imagherm_taylor}")
        print(f"\nSTO_imagherm_analytical = {STO_imagherm_analytical}")
        print(f"\ndiff = {STO_imagherm_analytical - STO_imagherm_taylor}")

    assert np.isclose(error_norm, 0.0, atol=1e-12)


if __name__ == "__main__":
    test_sparse_operator_complex_1()
    test_sparse_operator_complex_2()
    test_sparse_operator_complex_3()
    test_sparse_operator_complex_transform1()
    test_sparse_operator_complex_transform2()
    test_sparse_operator_complex_transform3()
    test_sparse_operator_complex_transform4()
    test_sparse_operator_complex_transform5()
