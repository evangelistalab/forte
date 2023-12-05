from psi4.core import Matrix, print_out


def check_mo_orthonormality(S: Matrix, Ca: Matrix):
    """
    Return whether the MO overlap matrix is identity or not.
    S_MO = Ca^T S_SO Ca.
    The MO overlap is the identity if and only if the orbitals are orthonormal.
    Most electronic structure methods are derived assuming orthonormal orbitals.

    :param S: a Psi4 Matrix of overlap integrals in the SO basis
    :param Ca: a Psi4 Matrix that holds orbital coefficients

    :return: S_MO == I
    """
    p4print = print_out

    p4print("\n  Checking orbital orthonormality against current geometry ...")

    S = S.clone()
    S.transform(Ca)  # S = Ca^T S Ca

    # test orbital orthonormality
    identity = S.clone()
    identity.identity()
    S.subtract(identity)
    absmax = S.absmax()

    if absmax > 1.0e-8:
        p4print("\n\n  Forte Warning: ")
        p4print("Input orbitals are NOT from the current geometry!")
        p4print(f"\n  Max value of MO overlap: {absmax:.15f}\n")
        return False
    else:
        p4print(" Done (OK)\n\n")
        return True
