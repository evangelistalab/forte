import forte
import psi4
import numpy as np
import pytest

# Error Tests
# - check for errors when the API is used incorrectly    

def test_dl_error_1():
    """Test RuntimeError when the guesses are not linearly independent"""
    size = 4
    nroot = 1
    solver = forte.DavidsonLiuSolver(size, nroot)
    h_diag = psi4.core.Vector("h_diag",4)
    h_diag.zero()
    h_diag.set(0,-1.0)
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,0.0)]])
    solver.add_test_sigma_builder([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    with pytest.raises(RuntimeError):
        solver.solve()

def test_dl_error_2():
    """Test RuntimeError when not enough guesses are provided"""
    size = 4
    nroot = 2
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,0.1)]])
    solver.add_test_sigma_builder(matrix.tolist())
    with pytest.raises(RuntimeError):
        solver.solve()

def test_dl_no_builder_set():
    # Calling Functions Out of Order:
    # What to test: Call functions in a different order than the correct usage. For instance, call solver.solve() before solver.startup().
    # Expected: The solver should either handle these scenarios gracefully or raise informative errors.
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,0.1)]])
    with pytest.raises(RuntimeError):
        solver.solve()
    solver.add_test_sigma_builder(matrix.tolist())

def test_dl_no_h_diag_set():
    # Calling Functions Out of Order:
    # What to test: Call functions in a different order than the correct usage. For instance, call solver.solve() before solver.startup().
    # Expected: The solver should either handle these scenarios gracefully or raise informative errors.
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_test_sigma_builder(matrix.tolist())
    with pytest.raises(RuntimeError):
        solver.solve()

if __name__ == '__main__':
    test_dl_error_1()
    test_dl_error_2()
    test_dl_no_builder_set()
    test_dl_no_h_diag_set()
    