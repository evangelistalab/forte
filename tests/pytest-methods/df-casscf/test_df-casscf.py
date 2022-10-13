import pytest
from forte.solvers import HF, ActiveSpaceSolver, MCSCF, input_factory


def test_df_casscf_1():

    ref_mcscf_energy_1_5 = -99.939310302576473 # @1.5
    ref_mcscf_energy_1_6 = -99.924066063941879 # @1.6
    xyz = """
    F
    H 1 1.5
    """

    input = input_factory(molecule=xyz, basis='cc-pVDZ', jkfit_aux_basis='cc-pVDZ-jkfit', int_type='DF')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(frozen_docc=[1, 0, 0, 0], restricted_docc=[1, 0, 1, 1], active=[2, 0, 0, 0])
    # create a HF object
    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='DETCI', states=state, mo_spaces=mo_spaces)
    # pass the FCI object to MCSCF
    mcscf = MCSCF(fci, e_convergence=1.0e-11)  # <- use information in fci to get active space, etc.
    # run the computation
    mcscf.run()

    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy_1_5], 1.0e-10)

    xyz = """
    F
    H 1 1.6
    """

    input = input_factory(molecule=xyz, basis='cc-pVDZ', jkfit_aux_basis='cc-pVDZ-jkfit', int_type='DF')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(frozen_docc=[1, 0, 0, 0], restricted_docc=[1, 0, 1, 1], active=[2, 0, 0, 0])
    # create a HF object
    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='DETCI', states=state, mo_spaces=mo_spaces)
    # pass the FCI object to MCSCF
    mcscf = MCSCF(fci, e_convergence=1.0e-11)  # <- use information in fci to get active space, etc.
    # run the computation
    mcscf.run()

    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy_1_6], 1.0e-10)


if __name__ == "__main__":
    test_df_casscf_1()
