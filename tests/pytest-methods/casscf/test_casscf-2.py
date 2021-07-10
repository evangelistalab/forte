import pytest
from forte.solvers import HF, ActiveSpaceSolver, MCSCF, input_factory


def test_casscf_2():
    """CASSCF on HF with no symmetry and conventional integrals"""
    ref_hf_energy = -99.87285247289
    ref_mcscf_energy = -99.939316381644

    xyz = """
    F
    H  1 R
    R = 1.50
    symmetry c1
    """

    input = input_factory(molecule=xyz, basis='cc-pvdz')
    state = input.state(charge=0, multiplicity=1)
    mo_spaces = input.mo_spaces(restricted_docc=[4], active=[2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()

    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


if __name__ == "__main__":
    test_casscf_2()
