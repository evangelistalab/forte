import forte
import psi4
import numpy as np
import pytest

def solve_dl(size, nroot):
    # create a 100 x 100 numpy array
    matrix = np.zeros((size, size))
    # fill the matrix with random values
    for i in range(size):
        matrix[i][i] = -1.0 + i * 1.0
        for j in range(i):
            matrix[i][j] = 0.05 / (1. + abs(i - j))
            matrix[j][i] = matrix[i][j]
    
    solver = forte.DavidsonLiuSolver2(size, nroot)
    solver.startup()
    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver.add_h_diag(h_diag)
    guesses = [[(i,1.0)] for i in range(nroot)]
    solver.add_guesses(guesses)
    solver.add_test_sigma_builder(matrix.tolist())
    solver.solve()
    test_evals = np.linalg.eigh(matrix)[0][:nroot]
    dl_evals = [solver.eigenvalues().get(i) for i in range(nroot)]
    assert np.allclose(dl_evals,test_evals)        

def test_dl_1():
    """Test the Davidson-Liu solver with a 4x4 matrix"""
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver2(size, nroot)
    solver.startup()
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,1.0)],[(1,1.0)],[(2,1.0)],[(3,1.0)]])
    solver.add_test_sigma_builder(matrix.tolist())
    solver.solve()
    eval = min(np.linalg.eigh(matrix)[0])
    assert np.isclose(solver.eigenvalues().get(0),eval)    

def test_dl_2():
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver2(size, nroot)
    solver.startup()
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,0.1)]])
    solver.add_test_sigma_builder(matrix.tolist())
    solver.solve()
    eval = min(np.linalg.eigh(matrix)[0])
    assert np.isclose(solver.eigenvalues().get(0),eval)    

def test_dl_3():
    solve_dl(10, 1)    
    solve_dl(100, 1)
    solve_dl(1000, 1)

    solve_dl(10, 2)    
    solve_dl(100, 2)
    solve_dl(1000, 2)

    solve_dl(10, 4)    
    solve_dl(100, 4)
    solve_dl(1000, 4)

def test_dl_range():
    """Test the Davidson-Liu solver with matrices of different sizes and different number of roots"""
    for size in range(1,40):
        for nroot in range(1,size + 1):
            solve_dl(size, nroot)

def test_dl_error_1():
    """Test RuntimeError when the guesses are not linearly independent"""
    size = 4
    nroot = 1
    solver = forte.DavidsonLiuSolver2(size, nroot)
    solver.startup()
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

    solver = forte.DavidsonLiuSolver2(size, nroot)
    solver.startup()
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,0.1)]])
    solver.add_test_sigma_builder(matrix.tolist())
    with pytest.raises(RuntimeError):
        solver.solve()

if __name__ == '__main__':
    test_dl_1()
    test_dl_2()
    test_dl_3()
    test_dl_range()
    test_dl_error_1()
    test_dl_error_2()