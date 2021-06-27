import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_aci_1():
    """Test FCI on Li2/STO-3G. Reproduces the test aci-1"""

    ref_hf_energy = -14.839846512738
    ref_aci_energy = -14.889166993726

    # setup job
    xyz = """
    Li
    Li 1 2.0
    """
    input = solver_factory(molecule=xyz, basis='DZ')
    state = input.state(charge=0, multiplicity=1, sym='ag')
    hf = HF(input, state=state)
    options = {
        'sigma': 0.001,
        'sci_enforce_spin_complete': False,
        'sci_project_out_spin_contaminants': False,
        'active_ref_type': 'hf'
    }
    aci = ActiveSpaceSolver(hf, type='ACI', states=state, options=options)
    aci.run()

    # check results
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert aci.value('active space energy')[state] == pytest.approx([ref_aci_energy], 1.0e-10)


if __name__ == "__main__":
    test_aci_1()
