import forte
import psi4
import numpy as np
import pytest

# Basic Functionality Tests
# Here we test for:
# - the correct way of using the API expecting successful solution without any errors.
# - initialization with different sizes and roots
# - the correct way of using the API expecting errors.

def solve_dl(size, nroot):
    # create a 100 x 100 numpy array
    matrix = np.zeros((size, size))
    # fill the matrix with random values
    for i in range(size):
        matrix[i][i] = -1.0 + i * 0.1
        for j in range(i):
            matrix[i][j] = 0.05 / (1. + abs(i - j))
            matrix[j][i] = matrix[i][j]

    evals, evecs = np.linalg.eigh(matrix)
    
    solver = forte.DavidsonLiuSolver(size, nroot)
    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver.add_h_diag(h_diag)
    guesses = [[(i,1.0)] for i in range(nroot)]
    solver.add_guesses(guesses)
    solver.add_test_sigma_builder(matrix.tolist())
    solver.solve()
    
    test_evals = evals[:nroot]
    dl_evals = [solver.eigenvalues().get(i) for i in range(nroot)]
    assert np.allclose(dl_evals,test_evals)        

def test_dl_1():
    """Test the Davidson-Liu solver with a 4x4 matrix"""
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    evals, evecs = np.linalg.eigh(matrix)

    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,1.0)],[(1,1.0)],[(2,1.0)],[(3,1.0)]])
    solver.add_test_sigma_builder(matrix.tolist())
    solver.solve()

    assert np.isclose(solver.eigenvalues().get(0),evals[0])    

def test_dl_2():
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    evals, evecs = np.linalg.eigh(matrix)

    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,0.1)]])
    solver.add_test_sigma_builder(matrix.tolist())
    solver.solve()

    assert np.isclose(solver.eigenvalues().get(0),evals[0])    

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

def test_dl_no_guess():
    # Calling the solver with no guess
    # Random guesses will be generated
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    evals, evecs = np.linalg.eigh(matrix)

    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_test_sigma_builder(matrix.tolist())
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,0.1)]])            
    solver.solve()

    assert np.isclose(solver.eigenvalues().get(0),evals[0])        

def test_project_out():
    # Testing the project out vectors function
    # Random guesses will be generated
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    evals, evecs = np.linalg.eigh(matrix)
    proj_out_evec = evecs[:,0]

    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_test_sigma_builder(matrix.tolist())
    solver.add_h_diag(h_diag) 
    solver.add_project_out_vectors([[(i,v) for i,v in enumerate(proj_out_evec)]])       
    solver.solve()    

    assert np.isclose(solver.eigenvalues().get(0),evals[1])        

def test_dl_restart_1():
    # Calling the solver twice
    # Random guesses will be generated
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    evals, evecs = np.linalg.eigh(matrix)

    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_test_sigma_builder(matrix.tolist())
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,1.0)]])            
    solver.solve()

    solver.solve()

    assert np.isclose(solver.eigenvalues().get(0),evals[0])         

def test_dl_restart_2():
    # Calling the solver twice
    # Random guesses will be generated
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    evals, evecs = np.linalg.eigh(matrix)

    h_diag = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag.set(i,matrix[i][i])

    solver = forte.DavidsonLiuSolver(size, nroot)
    solver.add_test_sigma_builder(matrix.tolist())
    solver.add_h_diag(h_diag)
    solver.add_guesses([[(0,1.0)]])            
    solver.solve()

    matrix2 = np.array([[-2, 1, 1, 1],[1, 0, 1, 1],[1, 1, 0, 1],[1, 1, 1, 0]])
    evals2, evecs2 = np.linalg.eigh(matrix2)

    h_diag2 = psi4.core.Vector("h_diag",size)
    for i in range(size):
        h_diag2.set(i,matrix[i][i])

    solver.add_test_sigma_builder(matrix2.tolist())
    solver.add_h_diag(h_diag2)
    solver.solve()

    assert np.isclose(solver.eigenvalues().get(0),evals2[0])

if __name__ == '__main__':
    # test_dl_1()
    # test_dl_2()
    # test_dl_3()
    # test_dl_range()
    # test_dl_no_guess()
    # test_project_out()
    # test_dl_restart_1()
    test_dl_restart_2()