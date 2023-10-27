import pytest
from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci_ex_1():
    """Test FCI on the ground and first singlet A1 states of acetone"""

    ref_hf_energy = -190.88631411404435
    ref_fci_energies = [-190.903043353477869, -190.460579059045074]

    # setup job
    xyz = """
    H   0.000000   2.136732  -0.112445
    H   0.000000  -2.136732  -0.112445
    H  -0.881334   1.333733  -1.443842
    H   0.881334  -1.333733  -1.443842
    H  -0.881334  -1.333733  -1.443842
    H   0.881334   1.333733  -1.443842
    C   0.000000   0.000000   0.000000
    C   0.000000   1.287253  -0.795902
    C   0.000000  -1.287253  -0.795902
    O   0.000000   0.000000   1.227600
    units angstrom
    """
    input = solver_factory(molecule=xyz, basis='3-21g')
    state = input.state(charge=0, multiplicity=1, sym='a1')

    hf = HF(input, state=state, docc=[8, 1, 2, 5])
    hf.run()
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)

    # define an active space
    mo_spaces = input.mo_spaces(frozen_docc=[3, 0, 0, 1], restricted_docc=[4, 1, 1, 3], active=[2, 0, 2, 1])
    # compute the FCI energy for the double B1 (M_S =  1/2) solution
    fci = ActiveSpaceSolver(hf, type='FCI', states={state: 2}, mo_spaces=mo_spaces)
    fci.run()

    # check results
    assert fci.value('active space energy')[state] == pytest.approx(ref_fci_energies, 1.0e-10)


if __name__ == "__main__":
    test_fci_ex_1()
