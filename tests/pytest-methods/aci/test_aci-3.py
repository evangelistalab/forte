import pytest
import psi4

from forte.solvers import solver_factory, HF, ActiveSpaceSolver, SpinAnalysis


def test_aci_3():
    """Test FCI on H4/STO-3G. Reproduces the test aci-1"""

    ref_hf_energy = -2.0310813811962456
    ref_aci_energy = -2.115455548674
    ref_acipt2_energy = -2.116454734743
    spin_val = 1.02027340

    # setup job
    xyz = """
    H -0.4  0.0 0.0
    H  0.4  0.0 0.0
    H  0.1 -0.3 1.0
    H -0.1  0.5 1.0
    """
    input = solver_factory(molecule=xyz, basis='cc-pVDZ')
    state = input.state(charge=0, multiplicity=1, sym='a')
    hf = HF(input, state=state, e_convergence=1.0e-12, d_convergence=1.0e-6)
    options = {
        'sigma': 0.001,
        'active_ref_type': 'hf',
        'diag_algorithm': 'SPARSE',
        'active_guess_size': 300,
        'aci_screen_alg': 'BATCH_HASH',
        'aci_nbatch': 2
    }
    aci = ActiveSpaceSolver(hf, type='ACI', states=state, e_convergence=1.0e-11, r_convergence=1.0e-7, options=options)
    spin = SpinAnalysis(aci, options={'SPIN_TEST': True})
    spin.run()

    # check results
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert aci.value('active space energy')[state] == pytest.approx([ref_aci_energy], 1.0e-9)
    assert psi4.core.variable("ACI+PT2 ENERGY") == pytest.approx(ref_acipt2_energy, 1.0e-8)
    assert psi4.core.variable("SPIN CORRELATION TEST") == pytest.approx(spin_val, 1.0e-7)


if __name__ == "__main__":
    test_aci_3()
