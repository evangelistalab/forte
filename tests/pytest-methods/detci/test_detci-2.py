import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_detci_2():
    """Test GAS CI calculation with multi-gas algorithm"""

    ref_hf_energy = -76.0172965561
    ref_gas1_energy = -76.029945793736
    ref_gas2_energy = -55.841944166803

    # setup job
    xyz = """
    O
    H 1 1.00
    H 1 1.00 2 103.1
    """
    input = solver_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    hf = HF(input, state=state, d_convergence=1.0e-10)
    hf.run()

    # define two states with different number of GAS constraints
    gas_state_1 = input.state(charge=0, multiplicity=1, sym='a1', gasmin=[0], gasmax=[2])
    gas_state_2 = input.state(charge=0, multiplicity=1, sym='a1', gasmin=[0], gasmax=[1])

    # define the active space
    mo_spaces = input.mo_spaces(gas1=[1, 0, 0, 0], gas2=[3, 0, 1, 2])
    # create a detci solver
    fci = ActiveSpaceSolver(
        hf, type='detci', states=[gas_state_1, gas_state_2], mo_spaces=mo_spaces, options={'active_ref_type': 'gas'}
    )
    fci.run()

    # check results
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert fci.value('active space energy')[gas_state_1] == pytest.approx([ref_gas1_energy], 1.0e-10)
    assert fci.value('active space energy')[gas_state_2] == pytest.approx([ref_gas2_energy], 1.0e-10)


if __name__ == "__main__":
    test_detci_2()
