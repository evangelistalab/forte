import pytest

from forte.solvers import input_factory, HF, ActiveSpaceSolver, SpinAnalysis
from forte.solvers.solver import Node


def test_solver_2():
    """Test RHF on H2."""

    # define a molecule
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    # create a molecular model
    input = input_factory(molecule=xyz, basis='cc-pVDZ')

    # specify the electronic state
    state = input.state(charge=0, multiplicity=1, sym='ag')

    # create a HF object and run
    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, state, 'FCI')
    spin = SpinAnalysis(fci)
    test_graph = """SpinAnalysis
└──ActiveSpaceSolver
   └──HF
      └──Input"""
    assert spin.computational_graph() == test_graph


def test_solver_2_computational_graph():
    # here we define some new nodes derived from Node to test the computational_graph function
    class A(Node):
        pass

    class B(Node):
        pass

    class C(Node):
        pass

    class D(Node):
        pass

    # create a complicated graph
    test_graph = """D
├──C
│  ├──B
│  │  └──A
│  └──B
│     └──A
└──B
   └──A"""
    a = A([], [])
    b = B([], [], input_nodes=a)
    c = C([], [], input_nodes=[b, b])
    d = D([], [], input_nodes=[c, b])
    assert d.computational_graph() == test_graph


if __name__ == "__main__":
    test_solver_2()
    test_solver_2_computational_graph()
