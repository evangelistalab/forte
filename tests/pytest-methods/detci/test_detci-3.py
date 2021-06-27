import pytest
import psi4
from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_detci_3():
    """Test DETCI transition dipole and oscillator strengths"""

    ref_hf_energy = -154.80914322697598

    ref_ag_energies = [-154.84695193645672, -154.59019912152513, -154.45363253270600]
    ref_bu_energies = [-154.54629332287075]
    ref_osc_0ag_0bu = 1.086716030595
    ref_osc_1ag_0bu = 0.013542791095
    ref_osc_2ag_0bu = 0.041762051475

    # setup job
    butadiene = """
    H  1.080977 -2.558832  0.000000
    H -1.080977  2.558832  0.000000
    H  2.103773 -1.017723  0.000000
    H -2.103773  1.017723  0.000000
    H -0.973565 -1.219040  0.000000
    H  0.973565  1.219040  0.000000
    C  0.000000  0.728881  0.000000
    C  0.000000 -0.728881  0.000000
    C  1.117962 -1.474815  0.000000
    C -1.117962  1.474815  0.000000
    """
    input = solver_factory(
        molecule=butadiene,
        basis='def2-svp',
        scf_aux_basis='def2-universal-jkfit',
        corr_aux_basis='def2-universal-jkfit',
        int_type='df'
    )

    state = input.state(charge=0, multiplicity=1, sym='ag')
    hf = HF(input, state=state, d_convergence=1.0e-12)
    hf.run()

    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)

    # define two states with different number of GAS constraints
    ag_state = input.state(charge=0, multiplicity=1, sym='ag')
    bu_state = input.state(charge=0, multiplicity=1, sym='bu')

    # define the active space
    mo_spaces = input.mo_spaces(frozen_docc=[2, 0, 0, 2], restricted_docc=[5, 0, 0, 4], active=[0, 2, 2, 0])
    # create a detci solver
    fci = ActiveSpaceSolver(
        hf, type='detci', states={
            ag_state: 3,
            bu_state: 1
        }, mo_spaces=mo_spaces, options={'transition_dipoles': True}
    )
    fci.run()

    # check results
    assert fci.value('active space energy')[ag_state] == pytest.approx(ref_ag_energies, 1.0e-10)
    assert fci.value('active space energy')[bu_state] == pytest.approx(ref_bu_energies, 1.0e-10)
    print("Oscillator strength singlet 0Ag -> 0Bu")
    assert psi4.core.variable("OSC. SINGLET 0AG -> 0BU") == pytest.approx(ref_osc_0ag_0bu, 1.0e-8)
    print("Oscillator strength singlet 1Ag -> 0Bu")
    assert psi4.core.variable("OSC. SINGLET 1AG -> 0BU") == pytest.approx(ref_osc_1ag_0bu, 1.0e-8)
    print("Oscillator strength singlet 2Ag -> 0Bu")
    assert psi4.core.variable("OSC. SINGLET 2AG -> 0BU") == pytest.approx(ref_osc_2ag_0bu, 1.0e-8)


if __name__ == "__main__":
    test_detci_3()
