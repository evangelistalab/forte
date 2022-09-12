import pytest
from forte.solvers import HF, ActiveSpaceSolver, MCSCF, input_factory


def test_casscf_1():
    """CASSCF on BeH2 with no symmetry and conventional integrals"""
    ref_hf_energy = -15.5049032510
    ref_mcscf_energy = -15.5107025722

    xyz = """
    Be        0.000000000000     0.000000000000     0.000000000000
    H         0.000000000000     1.390000000000     2.500000000000
    H         0.000000000000    -1.390000000000     2.500000000000
    units bohr
    no_reorient
    """

    input = input_factory(molecule=xyz, basis='3-21g')
    state = input.state(charge=0, multiplicity=1,sym='a1')
    mo_spaces = input.mo_spaces(restricted_docc=[2,0,0,0], active=[1,0,0,1])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()

    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


if __name__ == "__main__":
    test_casscf_1()
