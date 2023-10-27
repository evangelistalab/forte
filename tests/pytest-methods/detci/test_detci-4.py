import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_detci_4():
    """CASCI test of Forte DETCI using the SparseList algorithm to build the sigma vector"""

    ref_hf_energy = -99.977636678461636
    ref_fci_energy = -100.113732484560970

    xyz = """
    F
    H 1 1.0
    """
    input = solver_factory(molecule=xyz, basis='6-31g')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    hf = HF(input, state=state, e_convergence=1.0e-12, d_convergence=1.0e-8)
    # create a detci solver
    fci = ActiveSpaceSolver(
        hf,
        type='detci',
        states=state,
        mo_spaces=input.mo_spaces(frozen_docc=[1, 0, 0, 0]),
        options={'active_ref_type': 'cas'}
    )
    fci.run()

    # check results
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert fci.value('active space energy')[state] == pytest.approx([ref_fci_energy], 1.0e-10)


if __name__ == "__main__":
    test_detci_4()
